//! DRM synchronisation objects (`drm_syncobj`): the explicit-sync
//! primitive behind PipeWire's `SPA_META_SyncTimeline` and Wayland's
//! `wp_linux_drm_syncobj_v1`.
//!
//! A timeline syncobj carries a monotonically increasing sequence of
//! points, each backed by a GPU fence. The producer of a DMA-BUF names
//! the point after which the buffer may be read (*acquire*); the
//! consumer signals another point once it stopped reading (*release*).
//! Unlike implicit sync this works on every driver, NVIDIA's included:
//! the proprietary driver does not attach its rendering fences to the
//! DMA-BUF, so polling the buffer returns before the GPU finished.
//!
//! The ioctls are core DRM (`drm.h`), so any DRM node of the system
//! works for waiting and signalling a syncobj another process exported;
//! the render node of the producer's GPU is the natural choice.

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd as _, BorrowedFd, FromRawFd as _, OwnedFd};
use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use super::DrmDevice;

/// The request type of `ioctl`: `c_int` in Bionic, `c_ulong` in glibc and
/// musl. DRM request numbers use all 32 bits; Bionic takes them as the
/// same bits in a signed int.
#[cfg(target_os = "android")]
type IoctlRequest = libc::c_int;
#[cfg(not(target_os = "android"))]
type IoctlRequest = libc::c_ulong;

/// `DRM_IOWR(nr, size)`: `_IOWR('d', nr, size)`.
const fn drm_iowr(nr: u8, size: usize) -> u32 {
    // Read + write direction (3) in the top two bits, then the size
    // (14 bits), the type ('d') and the number. Every DRM struct used
    // here is far below the 14-bit size limit.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ioctl sizes fit 14 bits by construction (const-evaluated)"
    )]
    let size = size as u32;
    (3u32 << 30) | (size << 16) | ((b'd' as u32) << 8) | nr as u32
}

#[repr(C)]
#[derive(Default)]
struct SyncobjCreate {
    handle: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Default)]
struct SyncobjDestroy {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct SyncobjHandle {
    handle: u32,
    flags: u32,
    fd: i32,
    pad: u32,
}

/// `struct drm_syncobj_timeline_wait` without the `deadline_nsec`
/// field Linux 6.4 appended: DRM zero-extends shorter structs, so this
/// layout works on every kernel with timeline syncobjs (5.2+).
#[repr(C)]
#[derive(Default)]
struct SyncobjTimelineWait {
    handles: u64,
    points: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct SyncobjTimelineArray {
    handles: u64,
    points: u64,
    count_handles: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Default)]
struct SyncobjTransfer {
    src_handle: u32,
    dst_handle: u32,
    src_point: u64,
    dst_point: u64,
    flags: u32,
    pad: u32,
}

const IOCTL_CREATE: u32 = drm_iowr(0xBF, size_of::<SyncobjCreate>());
const IOCTL_DESTROY: u32 = drm_iowr(0xC0, size_of::<SyncobjDestroy>());
const IOCTL_HANDLE_TO_FD: u32 = drm_iowr(0xC1, size_of::<SyncobjHandle>());
const IOCTL_FD_TO_HANDLE: u32 = drm_iowr(0xC2, size_of::<SyncobjHandle>());
const IOCTL_TIMELINE_WAIT: u32 = drm_iowr(0xCA, size_of::<SyncobjTimelineWait>());
const IOCTL_QUERY: u32 = drm_iowr(0xCB, size_of::<SyncobjTimelineArray>());
const IOCTL_TRANSFER: u32 = drm_iowr(0xCC, size_of::<SyncobjTransfer>());
const IOCTL_TIMELINE_SIGNAL: u32 = drm_iowr(0xCD, size_of::<SyncobjTimelineArray>());

const FD_TO_HANDLE_IMPORT_SYNC_FILE: u32 = 1;
const HANDLE_TO_FD_EXPORT_SYNC_FILE: u32 = 1;
/// Also wait for the point's fence to be *submitted* (a producer may
/// queue the buffer before its GPU work reached the kernel).
const WAIT_FOR_SUBMIT: u32 = 1 << 1;
const WAIT_ALL: u32 = 1;

/// An open DRM node used for syncobj ioctls. Cheap to clone.
#[derive(Debug, Clone)]
pub struct SyncobjDevice {
    fd: Arc<OwnedFd>,
    node: DrmDevice,
}

impl SyncobjDevice {
    /// Opens the DRM node at `path` (`/dev/dri/renderD128`).
    ///
    /// # Errors
    /// The OS error from `open` / `stat`; `InvalidInput` when `path` is
    /// not a character device.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let metadata = file.metadata()?;
        if !metadata.file_type().is_char_device() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} is not a DRM device node", path.display()),
            ));
        }
        Ok(Self {
            node: DrmDevice::from_dev(metadata.rdev()),
            fd: Arc::new(OwnedFd::from(file)),
        })
    }

    /// Opens the render node `node`, or, without one (or when it cannot
    /// be found), the first render node of the system: syncobj ioctls
    /// are core DRM, so any node can wait on a syncobj fd.
    ///
    /// # Errors
    /// `NotFound` when no render node can be opened.
    pub fn open_render_node(node: Option<DrmDevice>) -> std::io::Result<Self> {
        let candidates = DrmDevice::render_node_paths();
        let wanted = node.and_then(|node| {
            candidates
                .iter()
                .find(|path| std::fs::metadata(path).is_ok_and(|m| m.rdev() == node.dev()))
                .and_then(|path| Self::open(path).ok())
        });
        if let Some(device) = wanted {
            return Ok(device);
        }
        candidates
            .iter()
            .find_map(|path| Self::open(path).ok())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no DRM render node can be opened for syncobj ioctls",
                )
            })
    }

    /// The node this device is.
    #[must_use]
    pub fn node(&self) -> DrmDevice {
        self.node
    }

    /// Creates a new (timeline-capable) syncobj with no point signalled.
    ///
    /// # Errors
    /// The OS error of `DRM_IOCTL_SYNCOBJ_CREATE`.
    pub fn create(&self) -> std::io::Result<Syncobj> {
        let mut args = SyncobjCreate::default();
        self.ioctl(IOCTL_CREATE, &mut args)?;
        Ok(Syncobj {
            device: self.clone(),
            handle: args.handle,
        })
    }

    /// Imports a syncobj another process exported (PipeWire's
    /// `SPA_DATA_SyncObj`, `wp_linux_drm_syncobj` timelines). The
    /// descriptor stays the caller's.
    ///
    /// # Errors
    /// The OS error of `DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE` (`EINVAL` for a
    /// descriptor that is no syncobj).
    pub fn import(&self, fd: BorrowedFd<'_>) -> std::io::Result<Syncobj> {
        let mut args = SyncobjHandle {
            fd: fd.as_raw_fd(),
            ..SyncobjHandle::default()
        };
        self.ioctl(IOCTL_FD_TO_HANDLE, &mut args)?;
        Ok(Syncobj {
            device: self.clone(),
            handle: args.handle,
        })
    }

    fn ioctl<T>(&self, request: u32, args: &mut T) -> std::io::Result<()> {
        loop {
            // SAFETY: `self.fd` is a live DRM node; `args` is the
            // `#[repr(C)]` struct `request` encodes (its size is part of
            // the request number), valid for reads and writes.
            #[allow(
                clippy::cast_lossless,
                clippy::cast_possible_wrap,
                reason = "the same 32 request bits in the libc's request type"
            )]
            let rc = unsafe {
                libc::ioctl(
                    self.fd.as_raw_fd(),
                    request as IoctlRequest,
                    std::ptr::from_mut(args).cast::<libc::c_void>(),
                )
            };
            if rc == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if !matches!(error.raw_os_error(), Some(libc::EINTR | libc::EAGAIN)) {
                return Err(error);
            }
        }
    }
}

/// What [`Syncobj::wait`] saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncWait {
    /// The point signalled.
    Signalled,
    /// The timeout passed first.
    TimedOut,
}

/// One syncobj handle on a [`SyncobjDevice`]; destroyed on drop.
#[derive(Debug)]
pub struct Syncobj {
    device: SyncobjDevice,
    handle: u32,
}

impl Syncobj {
    /// Waits until timeline `point` has been submitted and signalled, or
    /// `timeout` passes. Point 0 of a binary syncobj is its fence.
    ///
    /// # Errors
    /// The OS error of `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` other than the
    /// timeout (`ETIME`).
    pub fn wait(&self, point: u64, timeout: Duration) -> std::io::Result<SyncWait> {
        let mut handle = self.handle;
        let mut point = point;
        let mut args = SyncobjTimelineWait {
            handles: std::ptr::from_mut(&mut handle).addr() as u64,
            points: std::ptr::from_mut(&mut point).addr() as u64,
            timeout_nsec: monotonic_deadline(timeout),
            count_handles: 1,
            flags: WAIT_ALL | WAIT_FOR_SUBMIT,
            ..SyncobjTimelineWait::default()
        };
        match self.device.ioctl(IOCTL_TIMELINE_WAIT, &mut args) {
            Ok(()) => Ok(SyncWait::Signalled),
            Err(error) if error.raw_os_error() == Some(libc::ETIME) => Ok(SyncWait::TimedOut),
            Err(error) => Err(error),
        }
    }

    /// Signals timeline `point` from the CPU (a consumer's release).
    ///
    /// # Errors
    /// The OS error of `DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL`.
    pub fn signal(&self, point: u64) -> std::io::Result<()> {
        let mut handle = self.handle;
        let mut point = point;
        let mut args = SyncobjTimelineArray {
            handles: std::ptr::from_mut(&mut handle).addr() as u64,
            points: std::ptr::from_mut(&mut point).addr() as u64,
            count_handles: 1,
            flags: 0,
        };
        self.device.ioctl(IOCTL_TIMELINE_SIGNAL, &mut args)
    }

    /// The last signalled point.
    ///
    /// # Errors
    /// The OS error of `DRM_IOCTL_SYNCOBJ_QUERY`.
    pub fn signalled_point(&self) -> std::io::Result<u64> {
        let mut handle = self.handle;
        let mut point = 0u64;
        let mut args = SyncobjTimelineArray {
            handles: std::ptr::from_mut(&mut handle).addr() as u64,
            points: std::ptr::from_mut(&mut point).addr() as u64,
            count_handles: 1,
            flags: 0,
        };
        self.device.ioctl(IOCTL_QUERY, &mut args)?;
        Ok(point)
    }

    /// Attaches the fence of `sync_file` (a GPU job's completion, such
    /// as an `EGL_ANDROID_native_fence_sync` export) as timeline `point`.
    ///
    /// # Errors
    /// The OS error of the import or the transfer.
    pub fn import_sync_file(&self, point: u64, sync_file: BorrowedFd<'_>) -> std::io::Result<()> {
        let binary = self.device.create()?;
        let mut import = SyncobjHandle {
            handle: binary.handle,
            flags: FD_TO_HANDLE_IMPORT_SYNC_FILE,
            fd: sync_file.as_raw_fd(),
            pad: 0,
        };
        self.device.ioctl(IOCTL_FD_TO_HANDLE, &mut import)?;
        let mut transfer = SyncobjTransfer {
            src_handle: binary.handle,
            dst_handle: self.handle,
            src_point: 0,
            dst_point: point,
            ..SyncobjTransfer::default()
        };
        self.device.ioctl(IOCTL_TRANSFER, &mut transfer)
    }

    /// Exports timeline `point` as a `sync_file` (pollable; readable once
    /// the point signalled). The point must have been submitted.
    ///
    /// # Errors
    /// The OS error of the transfer or the export (`EINVAL` / `ENOENT`
    /// for a point without a fence yet).
    pub fn export_sync_file(&self, point: u64) -> std::io::Result<OwnedFd> {
        let binary = self.device.create()?;
        let mut transfer = SyncobjTransfer {
            src_handle: self.handle,
            dst_handle: binary.handle,
            src_point: point,
            dst_point: 0,
            ..SyncobjTransfer::default()
        };
        self.device.ioctl(IOCTL_TRANSFER, &mut transfer)?;
        let mut export = SyncobjHandle {
            handle: binary.handle,
            flags: HANDLE_TO_FD_EXPORT_SYNC_FILE,
            fd: -1,
            pad: 0,
        };
        self.device.ioctl(IOCTL_HANDLE_TO_FD, &mut export)?;
        // SAFETY: the ioctl succeeded, so `fd` is a new descriptor the
        // caller owns.
        Ok(unsafe { OwnedFd::from_raw_fd(export.fd) })
    }

    /// Exports the syncobj itself (to hand it to another process or
    /// import it on another device).
    ///
    /// # Errors
    /// The OS error of `DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD`.
    pub fn export(&self) -> std::io::Result<OwnedFd> {
        let mut export = SyncobjHandle {
            handle: self.handle,
            fd: -1,
            ..SyncobjHandle::default()
        };
        self.device.ioctl(IOCTL_HANDLE_TO_FD, &mut export)?;
        // SAFETY: as in `export_sync_file`.
        Ok(unsafe { OwnedFd::from_raw_fd(export.fd) })
    }
}

impl Drop for Syncobj {
    fn drop(&mut self) {
        let mut args = SyncobjDestroy {
            handle: self.handle,
            pad: 0,
        };
        if let Err(error) = self.device.ioctl(IOCTL_DESTROY, &mut args) {
            tracing::warn!(%error, handle = self.handle, "destroying a DRM syncobj handle");
        }
    }
}

/// `CLOCK_MONOTONIC` now + `timeout`, in nanoseconds (the absolute
/// deadline DRM waits take), saturating.
fn monotonic_deadline(timeout: Duration) -> i64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid out-pointer; CLOCK_MONOTONIC always exists.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut now) };
    #[allow(
        clippy::useless_conversion,
        reason = "`time_t` / `c_long` are narrower than i64 on 32-bit targets"
    )]
    let now_ns = i64::from(now.tv_sec)
        .saturating_mul(1_000_000_000)
        .saturating_add(i64::from(now.tv_nsec));
    now_ns.saturating_add(i64::try_from(timeout.as_nanos()).unwrap_or(i64::MAX))
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsFd as _;
    use std::time::Instant;

    use super::*;

    fn device() -> Option<SyncobjDevice> {
        match SyncobjDevice::open_render_node(None) {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping syncobj test: {error}");
                None
            }
        }
    }

    #[test]
    fn ioctl_numbers_match_drm_h() {
        assert_eq!(IOCTL_CREATE, 0xC008_64BF);
        assert_eq!(IOCTL_FD_TO_HANDLE, 0xC010_64C2);
        assert_eq!(IOCTL_TIMELINE_WAIT, 0xC028_64CA);
        assert_eq!(IOCTL_TIMELINE_SIGNAL, 0xC018_64CD);
        assert_eq!(IOCTL_TRANSFER, 0xC020_64CC);
    }

    #[test]
    fn wait_blocks_until_another_thread_signals() {
        let Some(device) = device() else { return };
        let timeline = match device.create() {
            Ok(timeline) => timeline,
            Err(error) => {
                eprintln!("skipping: {:?} has no syncobjs: {error}", device.node());
                return;
            }
        };
        assert_eq!(
            timeline.wait(1, Duration::from_millis(20)).unwrap(),
            SyncWait::TimedOut
        );
        // A second handle of the same syncobj, as a producer would hold.
        let exported = timeline.export().unwrap();
        let producer = device.import(exported.as_fd()).unwrap();
        let start = Instant::now();
        let signaller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            producer.signal(1).unwrap();
        });
        assert_eq!(
            timeline.wait(1, Duration::from_secs(2)).unwrap(),
            SyncWait::Signalled
        );
        let waited = start.elapsed();
        signaller.join().unwrap();
        assert!(
            waited >= Duration::from_millis(140),
            "returned after {waited:?}"
        );
        assert_eq!(timeline.signalled_point().unwrap(), 1);
        // Signalled points return at once.
        let start = Instant::now();
        assert_eq!(
            timeline.wait(1, Duration::from_secs(1)).unwrap(),
            SyncWait::Signalled
        );
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    #[test]
    fn signalled_point_exports_a_ready_sync_file() {
        let Some(device) = device() else { return };
        let Ok(timeline) = device.create() else {
            return;
        };
        timeline.signal(3).unwrap();
        let sync_file = timeline.export_sync_file(3).unwrap();
        let mut poll_fd = libc::pollfd {
            fd: sync_file.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd.
        assert_eq!(unsafe { libc::poll(&raw mut poll_fd, 1, 0) }, 1);
        // Re-imported at another point, it signals that point too.
        timeline.import_sync_file(5, sync_file.as_fd()).unwrap();
        assert_eq!(
            timeline.wait(5, Duration::from_millis(100)).unwrap(),
            SyncWait::Signalled
        );
    }
}
