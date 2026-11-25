//! Waiting for the producer of a DMA-BUF.
//!
//! Mesa drivers attach their rendering fence to the DMA-BUF (implicit
//! synchronisation); `poll(POLLIN)` on the buffer waits for those write
//! fences, which is what `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` + waiting on the
//! sync file would do, without the extra descriptor. CUDA takes no part in
//! implicit sync and cannot import a `sync_file`, so the wait is on the CPU.
//!
//! The NVIDIA driver attaches **no** fence to the buffers it renders: on
//! driver 595 the poll returned in about 2 µs with 300 ms of GL work still
//! queued on the buffer (`tests/interop.rs`, `implicit_wait_ignores_pending_nvidia_work`).
//! [`FenceWait::Implicit`] is therefore best effort; correctness on NVIDIA
//! comes from the producer: an explicit acquire point waited before
//! [`FenceWait::AlreadySignalled`] (PipeWire's `SyncTimeline`, see
//! `ferricast_core::Syncobj`), or a protocol that reports the frame only
//! once rendering completed (wlr-screencopy `ready` on niri).

use std::os::fd::{AsRawFd as _, BorrowedFd};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};

/// How [`DmaBufImporter::map`](super::DmaBufImporter::map) makes sure the
/// producer finished writing before CUDA reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceWait {
    /// Wait up to this long for the DMA-BUF's implicit write fences.
    /// Returns at once when they have signalled, and also when the
    /// producer attached none (NVIDIA): best effort, see the module docs.
    Implicit(Duration),
    /// The caller already waited (explicit sync: a `wp_linux_drm_syncobj`
    /// acquire point, a PipeWire sync object); do not wait again.
    AlreadySignalled,
}

impl FenceWait {
    /// Waits on `fd` as configured.
    pub(super) fn wait(self, fd: BorrowedFd<'_>) -> Result<()> {
        match self {
            Self::AlreadySignalled => Ok(()),
            Self::Implicit(timeout) => wait_readable(fd, timeout),
        }
    }
}

/// Waits until `fd` polls readable (a DMA-BUF: its write fences have
/// signalled) or `timeout` passes.
fn wait_readable(fd: BorrowedFd<'_>, timeout: Duration) -> Result<()> {
    let start = Instant::now();
    loop {
        let remaining = timeout.saturating_sub(start.elapsed());
        // Round up so a sub-millisecond remainder still waits.
        let millis = remaining.as_micros().div_ceil(1000);
        let millis = libc::c_int::try_from(millis).unwrap_or(libc::c_int::MAX);
        let mut poll_fd = libc::pollfd {
            fd: fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid `pollfd`; the descriptor is borrowed for the call.
        let ready = unsafe { libc::poll(&raw mut poll_fd, 1, millis) };
        match ready {
            1.. if poll_fd.revents & libc::POLLNVAL != 0 => {
                return Err(Error::Os {
                    call: "poll",
                    errno: libc::EBADF,
                });
            }
            1.. => return Ok(()),
            0 => {
                return Err(Error::FenceTimeout {
                    waited: start.elapsed(),
                });
            }
            _ => {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                if errno != libc::EINTR {
                    return Err(Error::Os {
                        call: "poll",
                        errno,
                    });
                }
            }
        }
    }
}
