//! Plugin frames as ferricast frames, without copying their pixels.

#[cfg(target_os = "linux")]
use std::os::fd::AsFd as _;
#[cfg(target_os = "linux")]
use std::sync::Arc;

use bytes::Bytes;
#[cfg(target_os = "linux")]
use ferricast_core::{
    BufferLease, DmaBufFd, DmaBufImporter, DmaBufPlane, DrmDevice, FrameSync, GpuFrame,
};
use ferricast_core::{CapturedFrame, FerricastError, Orientation, RawFrame, Result, Rotation};
use istmo_core::NativeBuffer;
use istmo_screen_capture::{
    BufferMemory, FrameMeta, Orientation as PluginOrientation, Rotation as PluginRotation,
};
#[cfg(target_os = "linux")]
use istmo_screen_capture::{GpuIdentity, SyncKind};
use tracing::warn;

use super::formats::byte_order;
#[cfg(target_os = "linux")]
use crate::vulkan::VulkanImporter;

/// The plugin's orientation (mirror, then clockwise quarter turns) in
/// ferricast's terms, which use the same model.
pub(crate) const fn orientation(orientation: PluginOrientation) -> Orientation {
    Orientation {
        mirrored: orientation.mirrored,
        rotation: match orientation.rotation {
            PluginRotation::R0 => Rotation::None,
            PluginRotation::R90 => Rotation::Clockwise90,
            PluginRotation::R180 => Rotation::Half,
            PluginRotation::R270 => Rotation::Clockwise270,
        },
    }
}

/// A converted frame, and whether it must still be turned upright on
/// the CPU before it goes out.
#[derive(Debug)]
pub(crate) struct Converted {
    pub(crate) frame: CapturedFrame,
    /// The turn still owed: a CPU frame of a rotated source, or a GPU
    /// frame of one for a consumer that does not turn GPU frames.
    pub(crate) pending: Option<Orientation>,
}

impl Converted {
    /// The frame as the consumer gets it: read back and turned when it
    /// still has to be. Blocking (a GPU readback, a full-frame shuffle).
    pub(crate) fn finish(self) -> Result<CapturedFrame> {
        let Some(orientation) = self.pending else {
            return Ok(self.frame);
        };
        match self.frame {
            // Turned by its own orientation on the way back.
            #[cfg(target_os = "linux")]
            CapturedFrame::Gpu(gpu) => gpu
                .into_cpu()
                .map(CapturedFrame::Cpu)
                .map_err(|e| FerricastError::Capture(format!("capture readback: {e}"))),
            CapturedFrame::Cpu(raw) => Ok(CapturedFrame::Cpu(turned(raw, orientation))),
            native @ CapturedFrame::Native(_) => Ok(native),
        }
    }
}

/// `raw` turned upright; a picture that cannot be turned (short data)
/// goes out as it is rather than not at all.
fn turned(raw: RawFrame, orientation: Orientation) -> RawFrame {
    match raw.clone().oriented(orientation) {
        Ok(upright) => upright,
        Err(e) => {
            warn!(%e, "capture: emitting the frame unoriented");
            raw
        }
    }
}

/// The bytes of a CPU frame: the plugin's buffer itself (its shared
/// memory), held until the last `Bytes` clone drops, which also hands
/// the slot back to the producer.
struct CpuPixels(NativeBuffer);

impl AsRef<[u8]> for CpuPixels {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes().unwrap_or_default()
    }
}

/// Turns plugin frames into ferricast frames for one session.
pub(crate) struct FrameMapper {
    /// Reads DMA-BUF frames back for CPU consumers.
    #[cfg(target_os = "linux")]
    pub(crate) importer: Option<Arc<VulkanImporter>>,
    /// The consumer turns GPU frames itself (`CaptureConfig::consumer_orients`).
    pub(crate) orients: bool,
    /// The pool generation the importer's cache belongs to.
    #[cfg(target_os = "linux")]
    generation: Option<u64>,
}

impl FrameMapper {
    #[cfg(target_os = "linux")]
    pub(crate) const fn new(importer: Option<Arc<VulkanImporter>>, orients: bool) -> Self {
        Self {
            importer,
            orients,
            generation: None,
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub(crate) const fn new(orients: bool) -> Self {
        Self { orients }
    }

    /// The plugin frame `(meta, buffer)` as a ferricast frame.
    ///
    /// # Errors
    /// A frame in memory or a format this crate never asks for, or a
    /// buffer that does not hold what its metadata says.
    pub(crate) fn convert(&mut self, meta: &FrameMeta, buffer: NativeBuffer) -> Result<Converted> {
        let turn = orientation(meta.orientation);
        match meta.memory {
            BufferMemory::Cpu => Ok(Converted {
                frame: CapturedFrame::Cpu(cpu_frame(meta, buffer)?),
                pending: (!turn.is_identity()).then_some(turn),
            }),
            #[cfg(target_os = "linux")]
            BufferMemory::DmaBuf => {
                let gpu = self.gpu_frame(meta, buffer, turn)?;
                Ok(Converted {
                    frame: CapturedFrame::Gpu(gpu),
                    pending: (!self.orients && !turn.is_identity()).then_some(turn),
                })
            }
            other => Err(FerricastError::Capture(format!(
                "capture: frames in {other:?} memory are not supported"
            ))),
        }
    }

    #[cfg(target_os = "linux")]
    fn gpu_frame(
        &mut self,
        meta: &FrameMeta,
        buffer: NativeBuffer,
        orientation: Orientation,
    ) -> Result<GpuFrame> {
        let format = byte_order(meta.format).ok_or_else(|| unsupported(meta))?;
        let dmabuf = buffer.dmabuf().ok_or_else(|| {
            FerricastError::Capture("capture: DMA-BUF frame without planes".into())
        })?;
        let [plane] = dmabuf.planes.as_slice() else {
            return Err(FerricastError::Capture(format!(
                "capture: {} planes in a packed RGB DMA-BUF",
                dmabuf.planes.len()
            )));
        };
        // The pool keeps the slot's fds; the frame gets a duplicate it
        // owns, so an import cache holding it never sees a recycled
        // descriptor (`DmaBufFd`).
        let fd = DmaBufFd::dup(plane.fd.as_fd())
            .map_err(|e| FerricastError::Capture(format!("capture: dup(DMA-BUF fd): {e}")))?;
        let size = plane.size.unwrap_or_else(|| {
            u64::from(plane.offset) + u64::from(plane.stride) * u64::from(dmabuf.height)
        });
        let plane = DmaBufPlane {
            fd,
            offset: plane.offset,
            stride: plane.stride,
            modifier: dmabuf.modifier,
            size: u32::try_from(size).unwrap_or(u32::MAX),
        };
        // Same slot of a new generation = a new buffer: cached imports
        // keyed on the old ones go.
        if self.generation.replace(meta.buffer.pool_generation) != Some(meta.buffer.pool_generation)
            && let Some(importer) = &self.importer
        {
            importer.reset_cache();
        }
        Ok(GpuFrame {
            width: meta.size.width,
            height: meta.size.height,
            stride: plane.stride,
            format,
            timestamp_us: meta.wall_us,
            plane,
            importer: self
                .importer
                .clone()
                .map(|importer| importer as Arc<dyn DmaBufImporter>),
            // The whole plugin buffer: dropping the last clone of the
            // frame hands the slot back to the producer.
            lease: Some(BufferLease::new(buffer)),
            // The importer runs on the GPU the backends report producing
            // on: the best guess for a frame that does not say.
            producer: meta
                .producer
                .and_then(|producer| match producer {
                    GpuIdentity::Drm(dev) => Some(DrmDevice::from_dev(dev)),
                    GpuIdentity::Luid(_) | GpuIdentity::RegistryId(_) => None,
                })
                .or_else(|| {
                    let importer = self.importer.as_ref()?;
                    importer.render_node().map(DrmDevice::from_dev)
                }),
            sync: match meta.sync {
                SyncKind::Ready => FrameSync::Ready,
                SyncKind::Implicit => FrameSync::Implicit,
            },
            orientation,
        })
    }
}

fn unsupported(meta: &FrameMeta) -> FerricastError {
    FerricastError::Capture(format!(
        "capture: {:?} frames are not supported",
        meta.format
    ))
}

/// A CPU frame over the plugin's own buffer: no copy.
fn cpu_frame(meta: &FrameMeta, buffer: NativeBuffer) -> Result<RawFrame> {
    let format = byte_order(meta.format).ok_or_else(|| unsupported(meta))?;
    let plane = meta
        .planes
        .first()
        .ok_or_else(|| FerricastError::Capture("capture: CPU frame without a plane".into()))?;
    let (width, height) = (meta.size.width, meta.size.height);
    let row = u64::from(format.packed_row_bytes(width));
    // A trimmed last row is fine: every row but the last needs its full
    // stride, the last only its pixels.
    let needed = match height {
        0 => 0,
        rows => u64::from(plane.stride) * u64::from(rows - 1) + row,
    };
    let len = buffer.as_bytes().map_or(0, <[u8]>::len) as u64;
    let start = u64::from(plane.offset);
    let end = start.saturating_add(plane.size.max(needed)).min(len);
    if u64::from(plane.stride) < row || end.saturating_sub(start) < needed {
        return Err(FerricastError::Capture(format!(
            "capture: CPU frame {width}x{height} stride {} holds {} bytes, needs {needed}",
            plane.stride,
            end.saturating_sub(start)
        )));
    }
    let (Ok(start), Ok(end)) = (usize::try_from(start), usize::try_from(end)) else {
        return Err(FerricastError::Capture(
            "capture: CPU frame too large".into(),
        ));
    };
    Ok(RawFrame {
        width,
        height,
        stride: plane.stride,
        format,
        data: Bytes::from_owner(CpuPixels(buffer)).slice(start..end),
        timestamp_us: meta.wall_us,
        colour: None,
        hdr: None,
    })
}

#[cfg(test)]
mod tests {
    use istmo_core::{BufferIdentity, NativeBuffer};
    use istmo_screen_capture::pixels::CpuView;
    use istmo_screen_capture::{PixelFormat as Format, PlaneLayout, Size};

    use super::*;

    const ROTATIONS: [PluginRotation; 4] = [
        PluginRotation::R0,
        PluginRotation::R90,
        PluginRotation::R180,
        PluginRotation::R270,
    ];

    fn every_orientation() -> impl Iterator<Item = PluginOrientation> {
        [false, true].into_iter().flat_map(|mirrored| {
            ROTATIONS
                .into_iter()
                .map(move |rotation| PluginOrientation { mirrored, rotation })
        })
    }

    /// Distinct pixels, `stride` bytes per row (padding after `width`).
    fn pattern(width: u32, height: u32, stride: u32) -> Vec<u8> {
        let mut bytes = vec![0xee; (stride * height) as usize];
        for (y, row) in bytes.chunks_exact_mut(stride as usize).enumerate() {
            for (x, px) in row[..(width * 4) as usize].chunks_exact_mut(4).enumerate() {
                px.copy_from_slice(&[u8::try_from(x).unwrap(), u8::try_from(y).unwrap(), 7, 255]);
            }
        }
        bytes
    }

    fn meta(width: u32, height: u32, stride: u32, orientation: PluginOrientation) -> FrameMeta {
        let mut meta = test_meta();
        meta.size = Size::new(width, height);
        meta.planes = vec![PlaneLayout {
            offset: 0,
            stride,
            size: u64::from(stride) * u64::from(height),
        }];
        meta.orientation = orientation;
        meta.wall_us = 1_700_000_000_000_000;
        meta
    }

    fn test_meta() -> FrameMeta {
        istmo_screen_capture::FrameMeta {
            seq: 0,
            pts_ns: 0,
            pts_clock: istmo_screen_capture::PtsClock::Monotonic,
            wall_us: 0,
            size: Size::new(0, 0),
            content_rect: istmo_screen_capture::Rect::new(0, 0, 0, 0),
            format: Format::Bgrx8,
            memory: BufferMemory::Cpu,
            planes: Vec::new(),
            modifier: None,
            buffer: istmo_screen_capture::BufferIdentity::default(),
            tier: 0,
            copied: None,
            producer: None,
            colour: istmo_screen_capture::Colour {
                space: istmo_screen_capture::ColourSpace::Srgb,
                transfer: istmo_screen_capture::TransferFunction::Srgb,
                range: istmo_screen_capture::ColourRange::Full,
            },
            hdr: None,
            orientation: PluginOrientation::UPRIGHT,
            damage: istmo_screen_capture::Damage::Full,
            cursor: None,
            sync: istmo_screen_capture::SyncKind::Ready,
            scale_permille: 1000,
            source_id: istmo_screen_capture::SourceId::new(
                istmo_screen_capture::BackendId::LinuxWlrScreencopy,
                "DP-3",
            ),
            repeat: false,
            skipped_before: 0,
            protected_content: false,
            composite: None,
        }
    }

    /// Every orientation the plugin can report turns a CPU frame exactly
    /// as the plugin's own pixel helper turns it: the orientation a GPU
    /// frame carries means what the CPU path applies.
    #[test]
    fn frame_orientation_matches_the_image_path() {
        let (width, height, stride) = (5, 3, 24);
        let bytes = pattern(width, height, stride);
        for plugin in every_orientation() {
            let raw = cpu_frame(
                &meta(width, height, stride, plugin),
                NativeBuffer::from_bytes(bytes.clone()),
            )
            .unwrap();
            let ours = turned(raw, orientation(plugin));
            let view =
                CpuView::new(&bytes, Size::new(width, height), stride, Format::Bgrx8).unwrap();
            let (size, theirs_stride, theirs) = view.orient(plugin).unwrap();
            assert_eq!(
                (ours.width, ours.height),
                (size.width, size.height),
                "{plugin:?}"
            );
            // Identity keeps the producer's stride; compare the rows.
            let row = (ours.width * 4) as usize;
            let rows = |data: &[u8], stride: u32| -> Vec<u8> {
                data.chunks(stride as usize)
                    .flat_map(|line| line[..row].to_vec())
                    .collect()
            };
            assert_eq!(
                rows(&ours.data, ours.stride),
                rows(&theirs, theirs_stride),
                "{plugin:?}"
            );
        }
    }

    #[test]
    fn identity_orientation_keeps_frame_untouched() {
        let bytes = pattern(4, 2, 20);
        let mut mapper = mapper(false);
        let frame = frame(
            meta(4, 2, 20, PluginOrientation::UPRIGHT),
            NativeBuffer::from_bytes(bytes.clone()),
        );
        let converted = mapper.convert_parts(frame).unwrap();
        assert!(converted.pending.is_none());
        let CapturedFrame::Cpu(raw) = converted.finish().unwrap() else {
            panic!("a CPU frame");
        };
        assert_eq!(raw.stride, 20, "the producer's stride, not repacked");
        assert_eq!(raw.data.as_ref(), bytes.as_slice());
        assert_eq!(raw.timestamp_us, 1_700_000_000_000_000, "wall clock µs");
    }

    #[test]
    fn y_invert_flips_rows_and_drops_stride_padding() {
        let bytes = pattern(2, 2, 12);
        let y_invert = PluginOrientation::Y_INVERT;
        let converted = mapper(false)
            .convert_parts(frame(
                meta(2, 2, 12, y_invert),
                NativeBuffer::from_bytes(bytes.clone()),
            ))
            .unwrap();
        assert_eq!(converted.pending, Some(orientation(y_invert)));
        let CapturedFrame::Cpu(raw) = converted.finish().unwrap() else {
            panic!("a CPU frame");
        };
        assert_eq!(raw.stride, 8);
        assert_eq!(&raw.data[..8], &bytes[12..20]);
        assert_eq!(&raw.data[8..], &bytes[..8]);
    }

    #[test]
    fn rotation_by_90_swaps_axes() {
        let turn = PluginOrientation {
            mirrored: false,
            rotation: PluginRotation::R90,
        };
        let raw = mapper(false)
            .convert_parts(frame(
                meta(4, 2, 16, turn),
                NativeBuffer::from_bytes(pattern(4, 2, 16)),
            ))
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!((raw.width(), raw.height()), (2, 4));
    }

    #[test]
    fn short_buffer_is_rejected() {
        let short = vec![0u8; 20];
        let error = mapper(false)
            .convert_parts(frame(
                meta(4, 2, 16, PluginOrientation::UPRIGHT),
                NativeBuffer::from_bytes(short),
            ))
            .unwrap_err();
        assert!(error.to_string().contains("needs 32"), "{error}");
    }

    #[test]
    fn frame_range_accepts_trimmed_last_row() {
        // Two rows of 16-byte stride, the last one cut after its pixels.
        let mut bytes = pattern(3, 2, 16);
        bytes.truncate(16 + 12);
        let raw = cpu_frame(
            &meta(3, 2, 16, PluginOrientation::UPRIGHT),
            NativeBuffer::from_bytes(bytes),
        )
        .unwrap();
        assert_eq!(raw.data.len(), 28);
    }

    #[test]
    fn unorientable_picture_is_emitted_unoriented() {
        // NV12 cannot be turned on the CPU; the frame still goes out.
        let raw = RawFrame {
            width: 2,
            height: 2,
            stride: 2,
            format: ferricast_core::PixelFormat::Nv12,
            data: Bytes::from_static(&[0; 6]),
            timestamp_us: 1,
            colour: None,
            hdr: None,
        };
        let out = turned(raw, orientation(PluginOrientation::Y_INVERT));
        assert_eq!((out.width, out.height, out.data.len()), (2, 2, 6));
    }

    #[test]
    fn cpu_frames_hold_the_producer_slot_until_the_last_clone() {
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&released);
        let buffer = NativeBuffer::from_bytes(pattern(2, 2, 8))
            .with_identity(BufferIdentity::new(1, 0))
            .with_release(move || flag.store(true, std::sync::atomic::Ordering::SeqCst));
        let raw = cpu_frame(&meta(2, 2, 8, PluginOrientation::UPRIGHT), buffer).unwrap();
        let clone = raw.clone();
        drop(raw);
        assert!(!released.load(std::sync::atomic::Ordering::SeqCst));
        drop(clone);
        assert!(released.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn ten_bit_frames_are_refused() {
        let mut meta = meta(2, 2, 8, PluginOrientation::UPRIGHT);
        meta.format = Format::Xr30;
        assert!(cpu_frame(&meta, NativeBuffer::from_bytes(pattern(2, 2, 8))).is_err());
    }

    #[cfg(target_os = "linux")]
    fn mapper(orients: bool) -> FrameMapper {
        FrameMapper::new(None, orients)
    }

    #[cfg(not(target_os = "linux"))]
    fn mapper(orients: bool) -> FrameMapper {
        FrameMapper::new(orients)
    }

    fn frame(meta: FrameMeta, buffer: NativeBuffer) -> (FrameMeta, NativeBuffer) {
        (meta, buffer)
    }

    impl FrameMapper {
        fn convert_parts(
            &mut self,
            (meta, buffer): (FrameMeta, NativeBuffer),
        ) -> Result<Converted> {
            self.convert(&meta, buffer)
        }
    }

    #[cfg(target_os = "linux")]
    mod gpu {
        use std::os::fd::{FromRawFd as _, OwnedFd};

        use istmo_core::{DmaBuf, DmaBufPlane as PluginPlane};

        use super::*;

        fn memfd() -> OwnedFd {
            // SAFETY: a valid NUL-terminated name; the fd is new and ours.
            unsafe {
                let fd = libc::memfd_create(c"ferricast-capture-test".as_ptr(), libc::MFD_CLOEXEC);
                assert!(fd >= 0);
                OwnedFd::from_raw_fd(fd)
            }
        }

        fn dmabuf_frame(
            rotation: PluginRotation,
        ) -> (
            (FrameMeta, NativeBuffer),
            Arc<std::sync::atomic::AtomicBool>,
        ) {
            let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = Arc::clone(&released);
            let dmabuf = DmaBuf {
                width: 64,
                height: 32,
                fourcc: Format::Bgrx8.drm_fourcc(),
                modifier: 0x0300_0000_0060_6014,
                planes: vec![PluginPlane {
                    fd: memfd(),
                    offset: 0,
                    stride: 256,
                    size: Some(256 * 32),
                }],
            };
            let buffer = NativeBuffer::from_dmabuf(dmabuf)
                .with_identity(BufferIdentity::new(1, 2))
                .with_release(move || flag.store(true, std::sync::atomic::Ordering::SeqCst));
            let mut meta = meta(
                64,
                32,
                256,
                PluginOrientation {
                    mirrored: false,
                    rotation,
                },
            );
            meta.memory = BufferMemory::DmaBuf;
            meta.modifier = Some(0x0300_0000_0060_6014);
            meta.producer = Some(GpuIdentity::Drm(0xe280));
            meta.buffer = istmo_screen_capture::BufferIdentity {
                pool_generation: 1,
                slot: 2,
            };
            (frame(meta, buffer), released)
        }

        #[test]
        fn dmabuf_frames_go_downstream_as_gpu_frames_with_their_lease() {
            let (frame, released) = dmabuf_frame(PluginRotation::R0);
            let converted = mapper(false).convert_parts(frame).unwrap();
            assert!(converted.pending.is_none());
            let CapturedFrame::Gpu(gpu) = converted.frame else {
                panic!("a GPU frame");
            };
            assert_eq!((gpu.width, gpu.height, gpu.stride), (64, 32, 256));
            assert_eq!(gpu.plane.modifier, 0x0300_0000_0060_6014);
            assert_eq!(gpu.plane.size, 256 * 32);
            assert_eq!(gpu.producer, Some(DrmDevice::from_dev(0xe280)));
            assert_eq!(gpu.sync, FrameSync::Ready);
            assert_eq!(gpu.timestamp_us, 1_700_000_000_000_000);
            let clone = gpu.clone();
            drop(gpu);
            assert!(
                !released.load(std::sync::atomic::Ordering::SeqCst),
                "a clone holds the lease"
            );
            drop(clone);
            assert!(released.load(std::sync::atomic::Ordering::SeqCst));
        }

        #[test]
        fn rotated_dmabufs_carry_the_orientation_for_a_consumer_that_turns_them() {
            let (frame, _) = dmabuf_frame(PluginRotation::R90);
            let converted = mapper(true).convert_parts(frame).unwrap();
            assert!(converted.pending.is_none());
            let CapturedFrame::Gpu(gpu) = converted.frame else {
                panic!("a GPU frame");
            };
            assert_eq!(gpu.orientation.rotation, Rotation::Clockwise90);
            assert_eq!(gpu.oriented_size(), (32, 64));
        }

        #[test]
        fn rotated_dmabufs_are_read_back_for_any_other_consumer() {
            let (frame, _) = dmabuf_frame(PluginRotation::R270);
            let converted = mapper(false).convert_parts(frame).unwrap();
            assert!(converted.pending.is_some());
            // No importer in a unit test: the readback is what fails.
            assert!(converted.finish().is_err());
        }
    }
}
