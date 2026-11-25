//! What the VA-API encoders (H.264, HEVC) share: the render node they open
//! (the configuration's GPU preference) and the video processor (VPP) that
//! takes captured DMA-BUFs into the encoder's NV12 surface, scaling,
//! letterboxing, converting colour and turning rotated or flipped captures
//! on the GPU.
//!
//! The VPP rotation and mirror path is compiled and unit-tested only: the
//! machines it was written on have no VA-API encoder or VPP (NVIDIA's VA-API
//! driver decodes only). The orientation mapping is checked against
//! [`ferricast_core::RawFrame::oriented`] by simulating `va_vpp.h`'s order
//! of operations; whether a driver follows that documentation is not.

use std::os::fd::RawFd;
use std::path::Path;
use std::rc::Rc;

use cros_libva::{
    BufferType, Config, Context, Display, ExternalBufferDescriptor, MemoryType, Picture,
    ProcPipelineParameterBuffer, Surface, VA_MIRROR_HORIZONTAL, VA_MIRROR_NONE, VA_MIRROR_VERTICAL,
    VA_ROTATION_90, VA_ROTATION_180, VA_ROTATION_270, VA_ROTATION_NONE,
    VADRMPRIMESurfaceDescriptor, VAEntrypoint, VAProcPipelineCaps, VAProfile, VARectangle,
    vaQueryVideoProcPipelineCaps,
};
use ferricast_codec::GpuInput as _;
use ferricast_core::{
    BackendId, EncoderConfig, FerricastError, GpuAffinity, GpuFrame, Orientation, PixelFormat,
    Result, Rotation,
};
use tracing::{debug, info};

use crate::h264::yuv::{
    VPP_BACKGROUND_BLACK, VPP_INPUT_STANDARD, VPP_OUTPUT_STANDARD, vpp_input_colour,
    vpp_output_colour,
};

/// Tried in order when the GPU preference names no render node (no
/// sysfs, or no GPU that suits VA-API encode).
const RENDER_NODES: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
];

/// VA-API fourccs for the BGRA / RGBA input layouts the DMA-BUF path
/// accepts. They name the **byte order in memory** (byte 0 = B for BGRA).
const VA_FOURCC_BGRA: u32 = 0x4152_4742; // 'B','G','R','A'
const VA_FOURCC_RGBA: u32 = 0x4142_4752; // 'R','G','B','A'

/// DRM fourccs are the inverse of VA's: the four letters describe the
/// **integer** as little-endian, not the memory order. So memory B,G,R,A ==
/// integer ARGB == `DRM_FORMAT_ARGB8888`.
const DRM_FORMAT_ARGB8888: u32 = 0x3432_5241; // 'A','R','2','4'
const DRM_FORMAT_ABGR8888: u32 = 0x3432_4241; // 'A','B','2','4'

/// `VA_RT_FORMAT_RGB32`: the VPP input surface of BGRA / RGBA imports.
const VA_RT_FORMAT_RGB32: u32 = 0x04;

/// Opens the VA-API display the configuration's GPU preference puts the
/// encoder on: the render node of the first non-NVIDIA GPU it ranks (the
/// NVIDIA VA-API driver has no encoder), else the first render node that
/// opens.
///
/// # Errors
/// [`FerricastError::Codec`] (`BackendUnavailable`) when the preference
/// pins a GPU that is missing or NVIDIA's; [`FerricastError::Encoder`]
/// when no render node opens.
pub(crate) fn open_display(config: &EncoderConfig) -> Result<Rc<Display>> {
    if let Some(gpu) = config
        .gpu_for(BackendId::Vaapi, GpuAffinity::NotNvidia)
        .map_err(FerricastError::Codec)?
    {
        let display = Display::open_drm_display(Path::new(&gpu.render_node)).map_err(|e| {
            FerricastError::Encoder(format!("VA-API: cannot open {}: {e}", gpu.render_node))
        })?;
        debug!(node = %gpu.render_node, gpu = %gpu.label(), "opened VA-API DRM display");
        return Ok(display);
    }
    for node in RENDER_NODES {
        let path = Path::new(node);
        if !path.exists() {
            continue;
        }
        match Display::open_drm_display(path) {
            Ok(display) => {
                debug!(node, "opened VA-API DRM display");
                return Ok(display);
            }
            Err(e) => debug!(node, error = %e, "render node open failed"),
        }
    }
    Display::open()
        .ok_or_else(|| FerricastError::Encoder("VA-API: no usable DRM render node".into()))
}

/// The `rotation_state` / `mirror_state` pair of a VPP pipeline.
///
/// `va_vpp.h` (`VAProcPipelineParameterBuffer::mirror_state`): "It is
/// assumed that the rotation operation is always performed before the
/// mirroring operation"; `VA_ROTATION_*` turn clockwise
/// (`VA_ROTATION_90`: the output's top-left is the source's bottom-left);
/// `VA_MIRROR_HORIZONTAL` swaps left and right, `VA_MIRROR_VERTICAL` top
/// and bottom. ferricast's [`Orientation`] is the other way round: a
/// left-to-right mirror first, then the clockwise rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VaOrientation {
    /// `VA_ROTATION_*`.
    pub rotation: u32,
    /// `VA_MIRROR_*`.
    pub mirror: u32,
}

impl VaOrientation {
    /// The VA operations that turn a picture like `orientation`, the
    /// preferred one first. A mirror then a rotation R equals R then the
    /// mirror conjugated by R: still left-to-right for 0° / 180°, top-to-
    /// bottom for 90° / 270°. A top-to-bottom mirror is also a half turn
    /// plus a left-to-right one, so each mirrored orientation has a second
    /// form, for drivers that offer only one mirror direction.
    #[must_use]
    pub(crate) fn of(orientation: Orientation) -> Vec<Self> {
        let (rotation, opposite) = match orientation.rotation {
            Rotation::None => (VA_ROTATION_NONE, VA_ROTATION_180),
            Rotation::Clockwise90 => (VA_ROTATION_90, VA_ROTATION_270),
            Rotation::Half => (VA_ROTATION_180, VA_ROTATION_NONE),
            Rotation::Clockwise270 => (VA_ROTATION_270, VA_ROTATION_90),
        };
        if !orientation.mirrored {
            return vec![Self {
                rotation,
                mirror: VA_MIRROR_NONE,
            }];
        }
        let (same, other) = if orientation.swaps_axes() {
            (VA_MIRROR_VERTICAL, VA_MIRROR_HORIZONTAL)
        } else {
            (VA_MIRROR_HORIZONTAL, VA_MIRROR_VERTICAL)
        };
        vec![
            Self {
                rotation,
                mirror: same,
            },
            Self {
                rotation: opposite,
                mirror: other,
            },
        ]
    }
}

/// The rotations and mirrors a driver's VPP performs
/// (`VAProcPipelineCaps::rotation_flags` / `mirror_flags`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct VppCaps {
    /// Bit `1 << VA_ROTATION_x` per supported rotation.
    pub rotation_flags: u32,
    /// `VA_MIRROR_HORIZONTAL` / `VA_MIRROR_VERTICAL` bits.
    pub mirror_flags: u32,
}

impl VppCaps {
    /// The VA operations for `orientation` this driver supports, if any.
    #[must_use]
    pub(crate) fn pick(self, orientation: Orientation) -> Option<VaOrientation> {
        VaOrientation::of(orientation).into_iter().find(|va| {
            let rotates =
                va.rotation == VA_ROTATION_NONE || self.rotation_flags & (1 << va.rotation) != 0;
            let mirrors = va.mirror == VA_MIRROR_NONE || self.mirror_flags & va.mirror != 0;
            rotates && mirrors
        })
    }

    /// Whether every one of the eight orientations is supported.
    #[must_use]
    pub(crate) fn orients_all(self) -> bool {
        Orientation::ALL
            .into_iter()
            .all(|orientation| self.pick(orientation).is_some())
    }
}

/// Descriptor passed to `Display::create_surfaces` to import a single-plane
/// DMA-BUF (every capture path emits BGRA / RGBA).
struct DmaBufImport {
    fd: RawFd,
    width: u32,
    height: u32,
    va_fourcc: u32,
    drm_fourcc: u32,
    modifier: u64,
    offset: u32,
    stride: u32,
    size: u32,
}

impl ExternalBufferDescriptor for DmaBufImport {
    const MEMORY_TYPE: MemoryType = MemoryType::DrmPrime2;
    type DescriptorAttribute = VADRMPRIMESurfaceDescriptor;

    fn va_surface_attribute(&mut self) -> Self::DescriptorAttribute {
        // A plain C struct of integers and arrays of integers: zero is a
        // valid value of every field, and the slots past `num_objects` /
        // `num_layers` stay zero, which the driver ignores.
        let mut d = VADRMPRIMESurfaceDescriptor {
            fourcc: self.va_fourcc,
            width: self.width,
            height: self.height,
            num_objects: 1,
            num_layers: 1,
            ..Default::default()
        };
        d.objects[0].fd = self.fd;
        d.objects[0].size = self.size;
        d.objects[0].drm_format_modifier = self.modifier;
        d.layers[0].drm_format = self.drm_fourcc;
        d.layers[0].num_planes = 1;
        d.layers[0].object_index[0] = 0;
        d.layers[0].offset[0] = self.offset;
        d.layers[0].pitch[0] = self.stride;
        d
    }
}

/// The video processor (`VAEntrypointVideoProc`) of one encoder session.
pub(crate) struct Vpp {
    /// Held so the config outlives its context (cros-libva drops the config
    /// first otherwise).
    _config: Config,
    context: Rc<Context>,
    caps: VppCaps,
}

impl Vpp {
    /// Brings up the VPP for surfaces of `width` × `height` and reads its
    /// rotation / mirror capabilities.
    ///
    /// # Errors
    /// [`FerricastError::Encoder`] when the driver has no VPP.
    pub(crate) fn new(display: &Rc<Display>, width: u32, height: u32) -> Result<Self> {
        let config = display
            .create_config(
                vec![],
                VAProfile::VAProfileNone,
                VAEntrypoint::VAEntrypointVideoProc,
            )
            .map_err(|e| FerricastError::Encoder(format!("vaCreateConfig(VPP): {e}")))?;
        // VPP contexts have no render-target list: the destination surface
        // is passed per picture.
        let context = display
            .create_context::<()>(&config, width, height, None, true)
            .map_err(|e| FerricastError::Encoder(format!("vaCreateContext(VPP): {e}")))?;
        let caps = query_caps(display, &context);
        info!(
            rotation_flags = caps.rotation_flags,
            mirror_flags = caps.mirror_flags,
            "VA-API VPP up; DMA-BUF input will be zero-copy"
        );
        Ok(Self {
            _config: config,
            context,
            caps,
        })
    }

    /// The driver's rotation / mirror support.
    #[must_use]
    pub(crate) fn caps(&self) -> VppCaps {
        self.caps
    }

    /// Whether `frame` can go through the VPP: a BGRA / RGBA buffer whose
    /// orientation the driver can apply. Others are read back and turned on
    /// the CPU, frame by frame.
    #[must_use]
    pub(crate) fn takes(&self, frame: &GpuFrame) -> bool {
        matches!(frame.format, PixelFormat::Bgra | PixelFormat::Rgba)
            && self.caps.pick(frame.orientation).is_some()
    }

    /// Imports `frame`'s DMA-BUF and blits it, turned upright, scaled into
    /// `output` (a region of the encoder's surface, sized for the upright
    /// picture) and converted to NV12, into `dest`.
    ///
    /// # Errors
    /// [`FerricastError::Encoder`] naming the VA-API step that failed, or
    /// for a frame [`Self::takes`] refuses.
    pub(crate) fn blit(
        &self,
        display: &Rc<Display>,
        frame: &GpuFrame,
        dest: &Surface<()>,
        output: VARectangle,
    ) -> Result<()> {
        let fail = |step: &str, e: &dyn std::fmt::Display| {
            FerricastError::Encoder(format!("VA-API VPP: {step}: {e}"))
        };
        let (va_fourcc, drm_fourcc) = match frame.format {
            PixelFormat::Bgra => (VA_FOURCC_BGRA, DRM_FORMAT_ARGB8888),
            PixelFormat::Rgba => (VA_FOURCC_RGBA, DRM_FORMAT_ABGR8888),
            other => {
                return Err(FerricastError::Encoder(format!(
                    "VA-API VPP: unsupported source pixel format {other:?}"
                )));
            }
        };
        let turn = self.caps.pick(frame.orientation).ok_or_else(|| {
            FerricastError::Encoder(format!(
                "VA-API VPP: the driver cannot apply {:?}",
                frame.orientation
            ))
        })?;
        let import = DmaBufImport {
            fd: frame.plane.fd.as_raw_fd(),
            width: frame.width,
            height: frame.height,
            va_fourcc,
            drm_fourcc,
            modifier: frame.plane.modifier,
            offset: frame.plane.offset,
            stride: frame.plane.stride,
            size: frame.plane.size,
        };
        // Per frame: the producer's buffers come and go with its pool, and
        // the import is a few `vaCreateSurfaces` calls.
        let mut surfaces = display
            .create_surfaces(
                VA_RT_FORMAT_RGB32,
                Some(va_fourcc),
                frame.width,
                frame.height,
                None,
                vec![import],
            )
            .map_err(|e| fail("create_surfaces(import DMA-BUF)", &e))?;
        let imported = surfaces
            .pop()
            .ok_or_else(|| FerricastError::Encoder("VA-API VPP: no surface imported".into()))?;
        // The source region is the whole buffer (unrotated); the driver
        // turns it, then scales it into `output` (`va_vpp.h`: "first rotated
        // by the specified angle and then scaled to fit the output_region").
        let pipe = ProcPipelineParameterBuffer::new(
            imported.id(),
            None,
            VPP_INPUT_STANDARD,
            Some(output),
            VPP_BACKGROUND_BLACK,
            VPP_OUTPUT_STANDARD,
            0,
            0,
            None,
            None,
            None,
            turn.rotation,
            None,
            turn.mirror,
            None,
            0,
            0,
            vpp_input_colour(),
            vpp_output_colour(),
            0,
            None,
        );
        let buffer = self
            .context
            .create_buffer(BufferType::ProcPipelineParameter(pipe))
            .map_err(|e| fail("vaCreateBuffer", &e))?;
        let mut picture = Picture::new(frame.timestamp_us, Rc::clone(&self.context), dest);
        picture.add_buffer(buffer);
        let picture = picture
            .begin::<()>()
            .map_err(|e| fail("vaBeginPicture", &e))?;
        let picture = picture.render().map_err(|e| fail("vaRenderPicture", &e))?;
        let picture = picture.end().map_err(|e| fail("vaEndPicture", &e))?;
        picture
            .sync::<()>()
            .map_err(|(e, _)| fail("vaSyncSurface", &e))?;
        // The driver is done with the per-frame surface.
        drop(imported);
        Ok(())
    }
}

/// `vaQueryVideoProcPipelineCaps` without filters; no rotation or mirror
/// when the query fails.
fn query_caps(display: &Display, context: &Context) -> VppCaps {
    let mut caps = VAProcPipelineCaps::default();
    // SAFETY: `display` and `context` are live for the call; no filter
    // buffers are passed (null, 0), as `va_vpp.h` allows; `caps` is a valid,
    // zeroed out-struct whose pointer fields stay null with zero counts, so
    // the driver writes no arrays (the same call FFmpeg's transpose_vaapi
    // makes).
    let status = unsafe {
        vaQueryVideoProcPipelineCaps(
            display.handle(),
            context.id(),
            std::ptr::null_mut(),
            0,
            &raw mut caps,
        )
    };
    if status != 0 {
        debug!(
            status,
            "vaQueryVideoProcPipelineCaps failed; no VPP rotation"
        );
        return VppCaps::default();
    }
    VppCaps {
        rotation_flags: caps.rotation_flags,
        mirror_flags: caps.mirror_flags,
    }
}

/// Whether the VA-API encoder for `config` would turn rotated GPU frames
/// itself: its render node's VPP rotates and mirrors every way. Opens the
/// display and a VPP context once per render node and process.
#[must_use]
pub(crate) fn orients_gpu_frames(config: &EncoderConfig) -> bool {
    use std::sync::Mutex;
    static KNOWN: Mutex<Vec<(Option<String>, bool)>> = Mutex::new(Vec::new());
    let node = config
        .gpu_for(BackendId::Vaapi, GpuAffinity::NotNvidia)
        .ok()
        .flatten()
        .map(|gpu| gpu.render_node);
    let mut known = KNOWN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, orients)) = known.iter().find(|(key, _)| *key == node) {
        return *orients;
    }
    let orients = open_display(config)
        .and_then(|display| Vpp::new(&display, 64, 64))
        .is_ok_and(|vpp| vpp.caps().orients_all());
    known.push((node, orients));
    orients
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use ferricast_core::RawFrame;

    use super::*;

    /// A row-major picture of pixel ids.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Grid {
        width: usize,
        height: usize,
        cells: Vec<u8>,
    }

    impl Grid {
        fn new(width: usize, height: usize) -> Self {
            let cells = (0..width * height)
                .map(|i| u8::try_from(i).unwrap())
                .collect();
            Self {
                width,
                height,
                cells,
            }
        }

        fn at(&self, x: usize, y: usize) -> u8 {
            self.cells[y * self.width + x]
        }

        fn build(width: usize, height: usize, pixel: impl Fn(usize, usize) -> u8) -> Self {
            let mut cells = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    cells.push(pixel(x, y));
                }
            }
            Self {
                width,
                height,
                cells,
            }
        }

        /// `VA_ROTATION_*` as `va_vpp.h` describes the output's corner
        /// (0, 0): the source's top-left (none), bottom-left (90),
        /// bottom-right (180), top-right (270): clockwise turns.
        fn rotate(&self, rotation: u32) -> Self {
            let (w, h) = (self.width, self.height);
            match rotation {
                VA_ROTATION_NONE => self.clone(),
                VA_ROTATION_90 => Self::build(h, w, |x, y| self.at(y, h - 1 - x)),
                VA_ROTATION_180 => Self::build(w, h, |x, y| self.at(w - 1 - x, h - 1 - y)),
                VA_ROTATION_270 => Self::build(h, w, |x, y| self.at(w - 1 - y, x)),
                other => panic!("rotation {other}"),
            }
        }

        fn mirror(&self, mirror: u32) -> Self {
            let (w, h) = (self.width, self.height);
            match mirror {
                VA_MIRROR_NONE => self.clone(),
                VA_MIRROR_HORIZONTAL => Self::build(w, h, |x, y| self.at(w - 1 - x, y)),
                VA_MIRROR_VERTICAL => Self::build(w, h, |x, y| self.at(x, h - 1 - y)),
                other => panic!("mirror {other}"),
            }
        }

        /// What the VPP does: rotation first, then the mirror.
        fn va(&self, va: VaOrientation) -> Self {
            self.rotate(va.rotation).mirror(va.mirror)
        }

        /// The CPU reference: [`RawFrame::oriented`].
        fn oriented(&self, orientation: Orientation) -> Self {
            let raw = RawFrame {
                width: u32::try_from(self.width).unwrap(),
                height: u32::try_from(self.height).unwrap(),
                stride: u32::try_from(self.width * 4).unwrap(),
                format: PixelFormat::Bgra,
                data: Bytes::from(
                    self.cells
                        .iter()
                        .flat_map(|&id| [id, 0, 0, 255])
                        .collect::<Vec<u8>>(),
                ),
                timestamp_us: 0,
                colour: None,
                hdr: None,
            };
            let turned = raw.oriented(orientation).unwrap();
            Self {
                width: turned.width as usize,
                height: turned.height as usize,
                cells: turned.data.chunks_exact(4).map(|pixel| pixel[0]).collect(),
            }
        }
    }

    #[test]
    fn every_va_form_matches_the_cpu_orientation() {
        let source = Grid::new(4, 3);
        for orientation in Orientation::ALL {
            let expected = source.oriented(orientation);
            let forms = VaOrientation::of(orientation);
            assert_eq!(forms.len(), if orientation.mirrored { 2 } else { 1 });
            for va in forms {
                assert_eq!(
                    source.va(va),
                    expected,
                    "{orientation:?} as rotation {} then mirror {}",
                    va.rotation,
                    va.mirror
                );
            }
        }
    }

    #[test]
    fn capabilities_pick_a_supported_form_or_none() {
        let all_rotations = (1 << VA_ROTATION_90) | (1 << VA_ROTATION_180) | (1 << VA_ROTATION_270);
        let full = VppCaps {
            rotation_flags: all_rotations,
            mirror_flags: VA_MIRROR_HORIZONTAL | VA_MIRROR_VERTICAL,
        };
        assert!(full.orients_all());
        let flipped_90 = Orientation {
            mirrored: true,
            rotation: Rotation::Clockwise90,
        };
        assert_eq!(
            full.pick(flipped_90),
            Some(VaOrientation {
                rotation: VA_ROTATION_90,
                mirror: VA_MIRROR_VERTICAL
            })
        );
        // Only a horizontal mirror: the half-turned form.
        let horizontal_only = VppCaps {
            mirror_flags: VA_MIRROR_HORIZONTAL,
            ..full
        };
        assert!(horizontal_only.orients_all());
        assert_eq!(
            horizontal_only.pick(flipped_90),
            Some(VaOrientation {
                rotation: VA_ROTATION_270,
                mirror: VA_MIRROR_HORIZONTAL
            })
        );
        // Rotations without mirrors: flipped frames go through the CPU.
        let no_mirror = VppCaps {
            mirror_flags: 0,
            ..full
        };
        assert!(!no_mirror.orients_all());
        assert!(no_mirror.pick(flipped_90).is_none());
        assert!(
            no_mirror
                .pick(Orientation {
                    mirrored: false,
                    rotation: Rotation::Clockwise270
                })
                .is_some()
        );
        // Nothing: only upright frames.
        let none = VppCaps::default();
        assert!(none.pick(Orientation::IDENTITY).is_some());
        assert!(none.pick(Orientation::FLIP_VERTICAL).is_none());
    }
}
