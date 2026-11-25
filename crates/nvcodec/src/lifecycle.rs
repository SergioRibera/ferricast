//! The thread that sets up and tears down driver sessions.
//!
//! The NVIDIA driver (595) crashes the process when an NVENC session is
//! destroyed while the driver's own libnvcuvid worker thread for it is still
//! busy: a SIGSEGV in that worker, or a glibc heap-corruption abort in
//! `cuArrayDestroy` under `nvEncDestroyEncoder`. A plain C program does the
//! same with nothing but "encode a few pictures, destroy the session" in a
//! loop, on one thread or several, with or without locks
//! (`crates/nvcodec/scripts/nvenc_churn.c`; `design/NVCODEC.md`, "NVENC and
//! context hardening"). Nothing here removes that race; it is the driver's.
//!
//! What this module guarantees is that ferricast adds no race of its own:
//! every session set-up and tear-down runs here ([`run`]), one at a time,
//! on one thread started on first use, which leaves the context of its last
//! job current (it pops it only to make another context current, and holds
//! no reference to it). Calls from that thread run inline. Per-picture calls
//! stay on the caller's thread.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::ptr;
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use crate::cuda::Context;
use crate::error::CudaStatus;
use crate::loader::CudaLibrary;
use crate::sys::cuda as sys;

/// A job. `'static` by construction in [`run`], which waits for it.
type Job = Box<dyn FnOnce() + Send>;

struct Request {
    context: sys::CUcontext,
    job: Job,
    done: SyncSender<std::thread::Result<()>>,
}

// SAFETY: the context handle is only compared and made current on the
// lifecycle thread, and the caller keeps the context alive until its job has
// run (`run` borrows its `Arc` and waits).
unsafe impl Send for Request {}

thread_local! {
    /// Set on the lifecycle thread.
    static ON_LIFECYCLE_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// The queue to the lifecycle thread, which is started on first use.
fn queue() -> &'static Mutex<Sender<Request>> {
    static QUEUE: OnceLock<Mutex<Sender<Request>>> = OnceLock::new();
    QUEUE.get_or_init(|| {
        let (sender, receiver) = channel();
        let spawned = std::thread::Builder::new()
            .name("nvcodec-lifecycle".into())
            .spawn(move || serve(&receiver));
        if let Err(error) = spawned {
            // Without the thread every send fails and `run` runs inline.
            tracing::error!(%error, "starting the NVIDIA session thread");
        }
        Mutex::new(sender)
    })
}

/// The lifecycle thread: runs jobs one at a time with their context current,
/// and leaves it current afterwards.
fn serve(requests: &Receiver<Request>) {
    ON_LIFECYCLE_THREAD.with(|on| on.set(true));
    let mut resident: sys::CUcontext = ptr::null_mut();
    for request in requests {
        if request.context != resident {
            resident = make_resident(resident, request.context);
        }
        let outcome = catch_unwind(AssertUnwindSafe(request.job));
        // The caller waits for this answer, so it is still there.
        let _ = request.done.send(outcome);
    }
}

/// Replaces `resident` with `context` as this thread's current context;
/// returns the context current now.
fn make_resident(resident: sys::CUcontext, context: sys::CUcontext) -> sys::CUcontext {
    let Ok(cuda) = CudaLibrary::get() else {
        return resident;
    };
    if !resident.is_null() {
        let mut popped = ptr::null_mut();
        // SAFETY: this thread pushed `resident`, the only entry on its stack
        // (jobs pop what they push); valid out-pointer. A failure (the
        // context was destroyed meanwhile) leaves nothing to undo.
        let _ = unsafe { (cuda.ctx_pop_current)(&raw mut popped) };
    }
    // SAFETY: the caller of `run` keeps `context` alive during the job.
    let status = CudaStatus(unsafe { (cuda.ctx_push_current)(context) });
    if status.code() == 0 {
        context
    } else {
        tracing::warn!(%status, "making a CUDA context current on the session thread");
        ptr::null_mut()
    }
}

/// Runs `job` on the lifecycle thread with `context` current there, and
/// returns its result; inline when called from that thread. A panic in `job`
/// is resumed here.
///
/// For the calls that set up or tear down driver sessions (see the module
/// docs). `job` may borrow from the caller: `run` returns only once it has
/// run.
pub fn run<T: Send>(context: &Arc<Context>, job: impl FnOnce() -> T + Send) -> T {
    if ON_LIFECYCLE_THREAD.with(Cell::get) {
        return job();
    }
    let (done, outcome) = sync_channel(1);
    let mut slot: Option<T> = None;
    let slot_ref = &mut slot;
    let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || *slot_ref = Some(job()));
    // SAFETY: the job borrows only from this frame, and this function does
    // not return or unwind before the job has run: it waits for the job's
    // outcome below, or runs it here when the thread cannot take it. That is
    // what a scoped thread relies on too.
    let job: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + '_>, Job>(job) };
    let request = Request {
        context: context.raw(),
        job,
        done,
    };
    let sent = queue()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .send(request);
    match sent {
        Ok(()) => match outcome.recv() {
            Ok(Ok(())) => {}
            Ok(Err(panic)) => resume_unwind(panic),
            Err(_) => unreachable!("the NVIDIA session thread answers every request it takes"),
        },
        // No thread: run here.
        Err(unsent) => (unsent.0.job)(),
    }
    slot.unwrap_or_else(|| unreachable!("the job ran and filled the slot"))
}
