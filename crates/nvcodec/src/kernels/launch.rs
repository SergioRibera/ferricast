//! The compiled kernels: the PTX image, its entry points, their parameter
//! structs (mirrors of `kernels.cu`) and one typed launch per kernel.

use std::ffi::{CStr, c_void};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use super::geometry::Lowered;
use crate::cuda::{Context, Function, Launch, Module, Stream};
use crate::error::Result;

/// `kernels.cu` compiled by `scripts/kernels.sh` (PTX ISA 6.0, `sm_52`),
/// NUL-terminated as `cuModuleLoadDataEx` wants it.
const PTX: &CStr =
    match CStr::from_bytes_with_nul(concat!(include_str!("kernels.ptx"), "\0").as_bytes()) {
        Ok(ptx) => ptx,
        Err(_) => panic!("the PTX image contains a NUL byte"),
    };

/// Threads per block of every kernel (`BLOCK` / `__launch_bounds__`).
const BLOCK: [u32; 2] = [32, 8];

/// A texture object or a device pointer, and its pitch (`struct Source`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourceRef {
    pub(crate) handle: u64,
    pub(crate) pitch: u64,
}

/// Pitched device memory (`struct Plane`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PlaneRef {
    pub(crate) ptr: u64,
    pub(crate) pitch: u64,
}

/// `struct RgbParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RgbParams {
    pub(crate) geometry: Lowered,
    pub(crate) src: SourceRef,
    pub(crate) dst: PlaneRef,
    pub(crate) swap_rb: i32,
    pub(crate) reserved: i32,
}

/// `struct Nv12Params`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Nv12Params {
    pub(crate) geometry: Lowered,
    pub(crate) src: SourceRef,
    pub(crate) luma: PlaneRef,
    pub(crate) chroma: PlaneRef,
    pub(crate) coeff: [i32; 9],
    pub(crate) bias_y: i32,
    pub(crate) bias_uv: i32,
    pub(crate) src_rgba: i32,
    pub(crate) reserved: [i32; 2],
}

/// `struct PlaneParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PlaneParams {
    pub(crate) geometry: Lowered,
    pub(crate) src: PlaneRef,
    pub(crate) dst: PlaneRef,
    pub(crate) out_scale: f32,
    pub(crate) out_max: i32,
    pub(crate) out_shift: i32,
    pub(crate) fill0: u32,
    pub(crate) fill1: u32,
    pub(crate) reserved: i32,
}

// The C side has matching `static_assert`s; `ptx_parameters_match` checks
// these sizes against the compiled PTX.
const _: () = assert!(size_of::<Lowered>() == 72);
const _: () = assert!(size_of::<RgbParams>() == 112);
const _: () = assert!(size_of::<Nv12Params>() == 176);
const _: () = assert!(size_of::<PlaneParams>() == 128);

/// Where RGB pixels are read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RgbSource {
    /// A texture object over a CUDA array.
    Array,
    /// Pitched device memory.
    Pitched,
}

/// The sample types and channel count of a plane conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaneKind {
    /// 8-bit → 8-bit.
    U8ToU8,
    /// 16-bit → 16-bit.
    U16ToU16,
    /// 16-bit → 8-bit.
    U16ToU8,
}

/// Entry points, in the order of [`Kernels::functions`].
const ENTRIES: [&CStr; 10] = [
    c"ferricast_rgb_to_rgb_array",
    c"ferricast_rgb_to_rgb_pitched",
    c"ferricast_rgb_to_nv12_array",
    c"ferricast_rgb_to_nv12_pitched",
    c"ferricast_plane_u8_u8_c1",
    c"ferricast_plane_u8_u8_c2",
    c"ferricast_plane_u16_u16_c1",
    c"ferricast_plane_u16_u16_c2",
    c"ferricast_plane_u16_u8_c1",
    c"ferricast_plane_u16_u8_c2",
];

/// The kernel module loaded on one context, with its functions looked up.
/// Shared: [`Kernels::get`] loads it once per context.
#[derive(Debug)]
pub struct Kernels {
    module: Module,
    functions: [Function; 10],
}

/// Modules loaded so far, one per context; dropped with their last user.
static LOADED: Mutex<Vec<Weak<Kernels>>> = Mutex::new(Vec::new());

impl Kernels {
    /// The kernels on `context`, JIT-compiled on first use (tens of
    /// milliseconds) and shared while anyone holds them.
    ///
    /// # Errors
    ///
    /// [`Error::KernelLoad`](crate::Error::KernelLoad) when the driver
    /// cannot compile the PTX, [`Error::Cuda`](crate::Error::Cuda),
    /// [`Error::SymbolMissing`](crate::Error::SymbolMissing).
    pub fn get(context: &Arc<Context>) -> Result<Arc<Self>> {
        let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
        loaded.retain(|kernels| kernels.strong_count() > 0);
        if let Some(kernels) = loaded
            .iter()
            .filter_map(Weak::upgrade)
            .find(|kernels| kernels.module.context().raw() == context.raw())
        {
            return Ok(kernels);
        }
        let started = std::time::Instant::now();
        let module = Module::load_ptx(context, PTX)?;
        let mut functions = Vec::with_capacity(ENTRIES.len());
        for name in ENTRIES {
            functions.push(module.function(name)?);
        }
        let functions = functions
            .try_into()
            .unwrap_or_else(|_| unreachable!("one function per entry"));
        tracing::debug!(elapsed = ?started.elapsed(), "image kernels JIT-compiled");
        let kernels = Arc::new(Self { module, functions });
        loaded.push(Arc::downgrade(&kernels));
        Ok(kernels)
    }

    /// The context the kernels run on.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.module.context()
    }

    /// Queues RGB → RGB.
    ///
    /// # Safety
    ///
    /// The source handle must be a live texture object of this context
    /// ([`RgbSource::Array`]) or device memory holding every pixel the
    /// geometry maps to, 4 bytes each, 4-byte aligned; the destination must
    /// hold `canvas_w` × 4 bytes × `canvas_h` rows at its pitch, 4-byte
    /// aligned. Both must stay allocated until the stream has run the
    /// kernel.
    pub(crate) unsafe fn rgb_to_rgb(
        &self,
        source: RgbSource,
        params: &RgbParams,
        stream: &Stream,
    ) -> Result<()> {
        let function = self.functions[match source {
            RgbSource::Array => 0,
            RgbSource::Pitched => 1,
        }];
        let geometry = &params.geometry;
        let launch = grid(geometry.canvas_w, geometry.canvas_h);
        // SAFETY: `RgbParams` is the kernel's only parameter (layout checked
        // against the PTX); the memory contract is the caller's.
        unsafe { self.launch(function, launch, params, stream) }
    }

    /// Queues RGB → NV12 (one thread per 2 × 2 block).
    ///
    /// # Safety
    ///
    /// As [`Kernels::rgb_to_rgb`] for the source; the luma destination must
    /// hold `canvas_w` bytes × `canvas_h` rows, the chroma one
    /// `2 × ceil(canvas_w / 2)` bytes × `ceil(canvas_h / 2)` rows.
    pub(crate) unsafe fn rgb_to_nv12(
        &self,
        source: RgbSource,
        params: &Nv12Params,
        stream: &Stream,
    ) -> Result<()> {
        let function = self.functions[match source {
            RgbSource::Array => 2,
            RgbSource::Pitched => 3,
        }];
        let geometry = &params.geometry;
        let launch = grid((geometry.canvas_w + 1) / 2, (geometry.canvas_h + 1) / 2);
        // SAFETY: as in `rgb_to_rgb`.
        unsafe { self.launch(function, launch, params, stream) }
    }

    /// Queues one plane of `channels` (1 or 2) interleaved samples.
    ///
    /// # Safety
    ///
    /// The source must hold every sample the geometry maps to (`channels`
    /// samples per pixel), the destination `canvas_w` × `channels` samples ×
    /// `canvas_h` rows, both aligned to their sample size, allocated until
    /// the stream has run the kernel.
    pub(crate) unsafe fn plane(
        &self,
        kind: PlaneKind,
        channels: Channels,
        params: &PlaneParams,
        stream: &Stream,
    ) -> Result<()> {
        let base = match kind {
            PlaneKind::U8ToU8 => 4,
            PlaneKind::U16ToU16 => 6,
            PlaneKind::U16ToU8 => 8,
        };
        let function = self.functions[base
            + match channels {
                Channels::One => 0,
                Channels::Two => 1,
            }];
        let geometry = &params.geometry;
        let launch = grid(geometry.canvas_w, geometry.canvas_h);
        // SAFETY: as in `rgb_to_rgb`.
        unsafe { self.launch(function, launch, params, stream) }
    }

    /// # Safety
    ///
    /// `P` must be the kernel's single parameter type; the memory contract
    /// of the typed wrappers above.
    unsafe fn launch<P: Copy>(
        &self,
        function: Function,
        launch: Launch,
        params: &P,
        stream: &Stream,
    ) -> Result<()> {
        let mut copy = *params;
        let mut pointers = [(&raw mut copy).cast::<c_void>()];
        // SAFETY: one pointer to a live value of the parameter's type; the
        // driver copies it before returning. Memory: the caller's contract.
        unsafe { self.module.launch(function, launch, stream, &mut pointers) }
    }
}

/// Interleaved samples per pixel of a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Channels {
    /// Luma.
    One,
    /// Interleaved chroma (Cb, Cr).
    Two,
}

/// Enough blocks of [`BLOCK`] to cover `width` × `height` threads.
fn grid(width: i32, height: i32) -> Launch {
    let blocks = |threads: i32, per_block: u32| {
        u32::try_from(threads)
            .unwrap_or(0)
            .div_ceil(per_block)
            .max(1)
    };
    Launch {
        grid: [blocks(width, BLOCK[0]), blocks(height, BLOCK[1]), 1],
        block: [BLOCK[0], BLOCK[1], 1],
        shared_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte size of the single `.param` of `entry` in the PTX.
    fn param_bytes(ptx: &str, entry: &str) -> Option<usize> {
        let marker = format!(".visible .entry {entry}(");
        let start = ptx.find(&marker)? + marker.len();
        let declaration = ptx[start..].lines().nth(1)?;
        let name = format!("{entry}_param_0[");
        let size = declaration.split(&name).nth(1)?.split(']').next()?;
        // Exactly one parameter: the next line closes the list.
        let closing = ptx[start..].lines().nth(2)?;
        (closing.trim() == ")").then(|| size.parse().ok())?
    }

    #[test]
    fn ptx_parameters_match() {
        let ptx = PTX.to_str().unwrap();
        assert!(ptx.contains(".version 6.0"));
        assert!(ptx.contains(".target sm_52"));
        let sizes = [
            size_of::<RgbParams>(),
            size_of::<RgbParams>(),
            size_of::<Nv12Params>(),
            size_of::<Nv12Params>(),
            size_of::<PlaneParams>(),
            size_of::<PlaneParams>(),
            size_of::<PlaneParams>(),
            size_of::<PlaneParams>(),
            size_of::<PlaneParams>(),
            size_of::<PlaneParams>(),
        ];
        for (entry, size) in ENTRIES.iter().zip(sizes) {
            let entry = entry.to_str().unwrap();
            assert_eq!(param_bytes(ptx, entry), Some(size), "{entry}");
            assert!(
                ptx.contains(&format!(".param .align 8 .b8 {entry}_param_0[")),
                "{entry} parameter alignment"
            );
        }
        assert_eq!(ptx.matches(".visible .entry").count(), ENTRIES.len());
    }

    #[test]
    fn grid_covers_the_canvas() {
        let launch = grid(1281, 9);
        assert_eq!(launch.grid, [41, 2, 1]);
        assert_eq!(grid(0, 0).grid, [1, 1, 1]);
    }
}
