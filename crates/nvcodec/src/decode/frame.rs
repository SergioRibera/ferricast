//! Decoded frames: host memory or pooled device memory, each returning its
//! buffer to the decoder's pool on drop.

use std::fmt;
use std::sync::Arc;

use super::format::{FrameInfo, Layout};
use super::pool::Pooled;
use super::sei::{SeiMessages, SeiStore};
use crate::cuda::{DeviceBuffer, Event, PinnedBuffer, PinnedCopy, Point, Region, Stream};
use crate::error::Result;

/// A host buffer, the event its copy completes, and the SEI of the
/// picture in it.
#[derive(Debug, Default)]
pub(super) struct HostSlot {
    pub(super) data: HostMemory,
    /// Recorded on the decoder's stream after an asynchronous copy into
    /// page-locked `data`; `None` when the copy was synchronous.
    pub(super) ready: Option<Event>,
    /// The copy into `data` may still be running: wait for `ready`.
    pub(super) copy_pending: bool,
    pub(super) sei: SeiStore,
}

impl HostSlot {
    /// Waits for the copy into the slot, if one is pending.
    pub(super) fn wait(&mut self) -> Result<()> {
        if self.copy_pending {
            if let Some(ready) = &self.ready {
                ready.synchronize()?;
            }
            self.copy_pending = false;
        }
        Ok(())
    }
}

/// Where a host frame's planes live.
#[derive(Debug, Default)]
pub(super) enum HostMemory {
    /// Nothing yet.
    #[default]
    Empty,
    /// Page-locked (`cuMemHostAlloc`): copies into it run asynchronously.
    Pinned(PinnedBuffer),
    /// Ordinary memory, when page-locked memory is not available.
    Pageable(Vec<u8>),
}

impl HostMemory {
    pub(super) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::Pinned(buffer) => buffer.as_slice(),
            Self::Pageable(data) => data,
        }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Empty => &mut [],
            Self::Pinned(buffer) => buffer.as_mut_slice(),
            Self::Pageable(data) => data,
        }
    }

    pub(super) fn is_pinned(&self) -> bool {
        matches!(self, Self::Pinned(_))
    }
}

/// A device buffer, the events ordering its uses, and the SEI of the
/// picture in it.
#[derive(Debug)]
pub(super) struct DeviceSlot {
    pub(super) buffer: Arc<DeviceBuffer>,
    /// Recorded on the decoder's stream after the copy into `buffer`.
    pub(super) ready: Event,
    /// Recorded on the consumer's stream by [`DeviceFrame::release_after`];
    /// the decoder's stream waits for it before writing `buffer` again.
    pub(super) released: Event,
    pub(super) release_pending: bool,
    pub(super) sei: SeiStore,
}

/// A decoded picture in host memory.
///
/// The planes are packed back to back with no row padding (each plane's
/// pitch is its row width), luma first: exactly what FFmpeg's `rawvideo`
/// muxer writes for the same pixel format (`nv12`, `p010le` for 10-bit
/// [`PixelFormat::P016`](super::PixelFormat::P016), `yuv444p`, ...). The
/// buffer goes back to the decoder's pool when the frame drops.
pub struct HostFrame {
    pub(super) info: FrameInfo,
    pub(super) layout: Layout,
    pub(super) slot: Pooled<HostSlot>,
}

impl HostFrame {
    /// Size, format, timestamp, colour and status of the picture.
    #[must_use]
    pub fn info(&self) -> &FrameInfo {
        &self.info
    }

    /// The timestamp of the packet that carried the picture.
    #[must_use]
    pub fn timestamp(&self) -> i64 {
        self.info.timestamp
    }

    /// Where each plane sits in [`HostFrame::data`].
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// All planes, packed.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.slot
            .data
            .as_slice()
            .get(..self.layout.total_bytes())
            .unwrap_or_default()
    }

    /// Plane `index` (0 luma, then chroma), `None` past the last one.
    #[must_use]
    pub fn plane(&self, index: usize) -> Option<&[u8]> {
        let plane = self.layout.planes().get(index)?;
        self.slot
            .data
            .as_slice()
            .get(plane.offset..plane.offset.checked_add(plane.len())?)
    }

    /// The picture's SEI messages: empty when it has none or the driver
    /// predates the SEI callback (NVDEC API 12.0).
    #[must_use]
    pub fn sei(&self) -> SeiMessages<'_> {
        self.slot.sei.messages()
    }
}

impl fmt::Debug for HostFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostFrame")
            .field("info", &self.info)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// A decoded picture in a pooled, pitched CUDA buffer on the decoder's
/// context.
///
/// The planes are stacked in one [`DeviceBuffer`] at its pitch, luma first:
/// plane `n + 1` starts on the row after plane `n`'s last row (so NV12 /
/// P016 chroma starts at `pitch * height`, the layout NVENC takes). See
/// [`DeviceFrame::layout`].
///
/// The copy into the buffer runs on the decoder's stream: wait for
/// [`DeviceFrame::ready`] (on a stream with [`DeviceFrame::wait`], or on the
/// host with [`DeviceFrame::synchronize`]) before reading it. The NVDEC
/// surface it came from is already released.
///
/// When the frame drops its buffer goes back to the pool and the decoder may
/// write the next picture into it. Dropping therefore means "done with it":
/// GPU work that reads the buffer must have completed, or be ordered with
/// [`DeviceFrame::release_after`].
pub struct DeviceFrame {
    pub(super) info: FrameInfo,
    pub(super) layout: Layout,
    pub(super) slot: Pooled<DeviceSlot>,
}

impl DeviceFrame {
    /// Size, format, timestamp, colour and status of the picture.
    #[must_use]
    pub fn info(&self) -> &FrameInfo {
        &self.info
    }

    /// The timestamp of the packet that carried the picture.
    #[must_use]
    pub fn timestamp(&self) -> i64 {
        self.info.timestamp
    }

    /// Where each plane sits in [`DeviceFrame::buffer`]: offsets from its
    /// device pointer, all at its pitch.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The buffer holding the planes: `width_bytes` is the widest plane row,
    /// `rows` the rows of all planes; [`DeviceFrame::layout`] says where each
    /// plane is.
    #[must_use]
    pub fn buffer(&self) -> &DeviceBuffer {
        &self.slot.buffer
    }

    /// The buffer as an `Arc`, for consumers that cache it by identity, such
    /// as [`encode::DeviceFrame`](crate::encode::DeviceFrame): the pool lends
    /// the same buffer to later pictures, so NVENC registers it once.
    ///
    /// Holding a clone past dropping the frame is safe but costs an
    /// allocation: the decoder will not write a buffer someone else still
    /// holds and gives the slot a new one instead.
    #[must_use]
    pub fn shared_buffer(&self) -> &Arc<DeviceBuffer> {
        &self.slot.buffer
    }

    /// Completes when the picture is in [`DeviceFrame::buffer`].
    #[must_use]
    pub fn ready(&self) -> &Event {
        &self.slot.ready
    }

    /// Makes work queued on `stream` after this call wait for the picture.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`](crate::Error::Cuda).
    pub fn wait(&self, stream: &Stream) -> Result<()> {
        stream.wait(&self.slot.ready)
    }

    /// Waits on the host for the picture.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`](crate::Error::Cuda), including errors of the copy.
    pub fn synchronize(&self) -> Result<()> {
        self.slot.ready.synchronize()
    }

    /// Returns the buffer to the pool once the work queued on `stream` so
    /// far has run: the decoder's stream waits for that point before
    /// writing into the buffer again. For consumers that read the frame
    /// asynchronously (a copy or kernel queued on their own stream).
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`](crate::Error::Cuda) when the event cannot be recorded; the buffer
    /// still goes back to the pool, unordered.
    pub fn release_after(mut self, stream: &Stream) -> Result<()> {
        self.slot.released.record(stream)?;
        self.slot.release_pending = true;
        Ok(())
    }

    /// Copies the picture to page-locked host memory, packed as in
    /// [`HostFrame`], and returns it with that layout: every plane queued
    /// after the picture is ready, one wait. For tests and diagnostics; a
    /// host consumer should decode in host mode instead.
    ///
    /// # Errors
    ///
    /// [`Error::Cuda`](crate::Error::Cuda).
    pub fn download(&self) -> Result<(PinnedBuffer, Layout)> {
        let context = self.slot.buffer.context();
        let packed = Layout::packed(self.info.format, self.info.width, self.info.height);
        let mut host = PinnedBuffer::new(context, packed.total_bytes())?;
        let stream = Stream::new(context)?;
        let mut row = 0;
        let copies: Vec<PinnedCopy> = self
            .layout
            .planes()
            .iter()
            .zip(packed.planes())
            .map(|(source, target)| {
                let src =
                    Region::new(source.width_bytes, source.rows).at(Point { x_bytes: 0, row });
                row += source.rows;
                PinnedCopy {
                    src,
                    offset: target.offset,
                    pitch: target.pitch,
                }
            })
            .collect();
        host.download(&self.slot.buffer, &copies, &stream, Some(&self.slot.ready))?;
        Ok((host, packed))
    }

    /// The picture's SEI messages: empty when it has none or the driver
    /// predates the SEI callback (NVDEC API 12.0).
    #[must_use]
    pub fn sei(&self) -> SeiMessages<'_> {
        self.slot.sei.messages()
    }
}

impl fmt::Debug for DeviceFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceFrame")
            .field("info", &self.info)
            .field("layout", &self.layout)
            .field("buffer", &self.slot.buffer)
            .finish_non_exhaustive()
    }
}
