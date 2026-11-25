//! DMA-BUF → CUDA import on real hardware, without a compositor: buffers are
//! allocated with GBM on the CUDA device's render node, filled by an
//! independent writer (GLES for tiled layouts, a separate CUDA registration
//! for LINEAR ones), imported, mapped, copied
//! to host memory and compared bit-exact.
//!
//! Every test prints why it skips when the machine has no NVIDIA GPU, EGL,
//! GBM or render node.

#![cfg(target_os = "linux")]
#![allow(
    clippy::missing_panics_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::print_stderr,
    reason = "hardware tests: panics are failures, sizes are small and positive"
)]

use std::os::fd::{AsFd as _, FromRawFd as _, OwnedFd};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferricast_nvcodec::Error;
use ferricast_nvcodec::cuda::{ArrayView, HostImageMut, Stream};
use ferricast_nvcodec::interop::{
    BufferKey, DmaBuf, DmaBufImporter, DrmFormat, DrmNode, FenceWait, FramePool, FrameView, Import,
    Modifier,
};

use support::{Bo, Gpu};

const FENCE: FenceWait = FenceWait::Implicit(Duration::from_secs(1));

/// The test pattern: four bytes per pixel, in memory order, distinct per
/// position and seed (including the fourth byte).
fn pattern(width: u32, height: u32, seed: u8) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            [
                (x as u8) ^ seed,
                (y as u8).wrapping_mul(3),
                ((x >> 8) as u8) ^ ((y >> 8) as u8).wrapping_add(seed),
                (x.wrapping_add(y) as u8) ^ 0x5a,
            ]
        })
        .collect()
}

/// Maps `key`, copies it through a pooled device buffer and returns the
/// pixels, tightly packed.
fn read_back(importer: &DmaBufImporter, key: BufferKey, stream: &Stream) -> Vec<u8> {
    let frame = importer.map(key, FENCE).unwrap();
    let pool = FramePool::new(
        importer.context(),
        frame.width() as usize * 4,
        frame.height() as usize,
    );
    let buffer = frame.copy_to_pool(&pool, stream).unwrap();
    stream.synchronize().unwrap();
    let mut host = vec![0; buffer.width_bytes() * buffer.rows()];
    let pitch = buffer.width_bytes();
    buffer
        .download(
            buffer.region(),
            HostImageMut {
                data: &mut host,
                pitch,
            },
        )
        .unwrap();
    host
}

fn assert_same(got: &[u8], want: &[u8], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(index) = got.iter().zip(want).position(|(a, b)| a != b) {
        let wrong = got.iter().zip(want).filter(|(a, b)| a != b).count();
        panic!(
            "{what}: {wrong} bytes differ, first at byte {index} (pixel {}): got {:?}, want {:?}",
            index / 4,
            &got[index / 4 * 4..index / 4 * 4 + 4],
            &want[index / 4 * 4..index / 4 * 4 + 4],
        );
    }
}

fn dma_buf(bo: &Bo) -> DmaBuf<'_> {
    DmaBuf {
        fd: bo.fd.as_fd(),
        width: bo.width,
        height: bo.height,
        format: bo.format,
        offset: bo.offset,
        stride: bo.stride,
        modifier: bo.modifier,
    }
}

/// Imports a buffer of every supported format and layout, written by an
/// independent writer, and reads it back bit-exact.
#[test]
fn imports_every_format_and_layout_bit_exact() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&gpu.context).unwrap();
    let (width, height) = (1920, 1080);
    let mut key = 0;
    for format in DrmFormat::ALL {
        let tiled = importer
            .modifiers(format)
            .iter()
            .copied()
            .find(|modifier| *modifier != Modifier::LINEAR);
        assert!(
            importer.modifiers(format).contains(&Modifier::LINEAR),
            "LINEAR is importable for {format:?}"
        );
        for modifier in [Some(Modifier::LINEAR), tiled].into_iter().flatten() {
            let Some(bo) = gpu.allocate(width, height, format, modifier) else {
                continue;
            };
            let want = pattern(width, height, 7);
            gpu.write(&bo, &want);
            key += 1;
            assert_eq!(
                importer.import(BufferKey(key), &dma_buf(&bo)).unwrap(),
                Import::Registered
            );
            {
                let frame = importer.map(BufferKey(key), FENCE).unwrap();
                let tiled_layout = modifier != Modifier::LINEAR;
                assert_eq!(
                    matches!(frame.view(), FrameView::Array(_)),
                    tiled_layout,
                    "{format:?} {modifier:x?}: arrays for tiled layouts, pitched memory for LINEAR"
                );
                if let Some(array) = frame.array() {
                    assert_eq!((array.width(), array.height()), (width, height));
                }
            }
            let got = read_back(&importer, BufferKey(key), &stream);
            assert_same(&got, &want, &format!("{format:?} {modifier:x?}"));
            eprintln!("{format:?} modifier {:#018x}: bit-exact", modifier.0);
        }
    }
    assert!(key >= 4, "at least LINEAR of every format was tested");
}

/// A cached import is reused and aliases the buffer: new content shows
/// through without registering again.
#[test]
fn cached_import_is_reused_and_zero_copy() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&gpu.context).unwrap();
    let (width, height) = (640, 360);
    for modifier in [Modifier::LINEAR, Gpu::tiled_modifier(&importer)] {
        let bo = gpu
            .allocate(width, height, DrmFormat::Xbgr8888, modifier)
            .unwrap();
        let key = BufferKey::from((1, modifier.0));
        gpu.write(&bo, &pattern(width, height, 1));
        assert_eq!(
            importer.import(key, &dma_buf(&bo)).unwrap(),
            Import::Registered
        );
        let array_before = importer
            .map(key, FENCE)
            .unwrap()
            .array()
            .map(ArrayView::raw);
        assert_same(
            &read_back(&importer, key, &stream),
            &pattern(width, height, 1),
            "first frame",
        );
        // The producer renders the next frame into the same buffer.
        gpu.write(&bo, &pattern(width, height, 2));
        assert_eq!(importer.import(key, &dma_buf(&bo)).unwrap(), Import::Cached);
        let array_after = importer
            .map(key, FENCE)
            .unwrap()
            .array()
            .map(ArrayView::raw);
        assert_eq!(array_before, array_after, "same registration");
        assert_same(
            &read_back(&importer, key, &stream),
            &pattern(width, height, 2),
            "second frame through the cached registration",
        );
        // The same key with another description is re-registered.
        let mut moved = dma_buf(&bo);
        moved.height -= 8;
        assert_eq!(importer.import(key, &moved).unwrap(), Import::Replaced);
        assert_eq!(importer.map(key, FENCE).unwrap().height(), height - 8);
    }
    assert_eq!(importer.len(), 2);
}

#[test]
fn forget_and_clear() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let bos: Vec<Bo> = (0..3)
        .map(|_| {
            gpu.allocate(64, 64, DrmFormat::Argb8888, Modifier::LINEAR)
                .unwrap()
        })
        .collect();
    for (index, bo) in (0_u128..).zip(&bos) {
        importer.import(BufferKey(index), &dma_buf(bo)).unwrap();
    }
    assert_eq!(importer.len(), 3);
    assert!(importer.forget(BufferKey(1)));
    assert!(!importer.forget(BufferKey(1)));
    assert!(!importer.contains(BufferKey(1)));
    assert!(matches!(
        importer.map(BufferKey(1), FENCE),
        Err(Error::UnknownBuffer { key: 1 })
    ));
    assert!(importer.map(BufferKey(0), FENCE).is_ok());
    importer.clear();
    assert!(importer.is_empty());
    assert!(matches!(
        importer.map(BufferKey(0), FENCE),
        Err(Error::UnknownBuffer { .. })
    ));
    // Re-importing after a clear registers again.
    assert_eq!(
        importer.import(BufferKey(0), &dma_buf(&bos[0])).unwrap(),
        Import::Registered
    );
}

/// The importer keeps what it needs: the producer may free its buffer, and
/// the context, importer, pool and pooled buffers may go in any order.
#[test]
fn drop_order_is_free() {
    let Some(gpu) = Gpu::get() else { return };
    let (width, height) = (320, 240);
    let want = pattern(width, height, 9);
    let context = Arc::clone(&gpu.context);
    let mut importer = DmaBufImporter::new(&context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&context).unwrap();
    let key = BufferKey(1);
    {
        let bo = gpu
            .allocate(
                width,
                height,
                DrmFormat::Abgr8888,
                Gpu::tiled_modifier(&importer),
            )
            .unwrap();
        gpu.write(&bo, &want);
        importer.import(key, &dma_buf(&bo)).unwrap();
        // The producer frees its buffer object and descriptor here.
    }
    assert_same(
        &read_back(&importer, key, &stream),
        &want,
        "after the producer freed the buffer",
    );
    let pool = FramePool::new(&context, width as usize * 4, height as usize);
    let pooled = importer
        .map(key, FENCE)
        .unwrap()
        .copy_to_pool(&pool, &stream)
        .unwrap();
    drop(context);
    drop(importer);
    drop(pool);
    stream.synchronize().unwrap();
    drop(pooled);
    drop(stream);
}

#[test]
fn rejects_bad_buffers() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let bo = gpu
        .allocate(256, 128, DrmFormat::Xrgb8888, Modifier::LINEAR)
        .unwrap();

    // Larger than the buffer.
    let mut too_big = dma_buf(&bo);
    too_big.height = 4096;
    let error = importer.import(BufferKey(1), &too_big).unwrap_err();
    assert!(matches!(error, Error::InvalidDmaBuf { .. }), "{error}");

    // Stride smaller than a row.
    let mut narrow = dma_buf(&bo);
    narrow.stride = 256 * 4 - 4;
    assert!(matches!(
        importer.import(BufferKey(1), &narrow),
        Err(Error::InvalidDmaBuf { .. })
    ));

    // A modifier the GPU does not know.
    let mut odd = dma_buf(&bo);
    odd.modifier = Modifier(0x0123_4567);
    assert!(matches!(
        importer.import(BufferKey(1), &odd),
        Err(Error::UnsupportedModifier {
            modifier: 0x0123_4567,
            ..
        })
    ));

    // A descriptor that is not a DMA-BUF: a memfd of the right size.
    // SAFETY: plain syscalls; the name is NUL-terminated.
    let memfd = unsafe {
        let raw = libc::memfd_create(c"not-a-dmabuf".as_ptr(), libc::MFD_CLOEXEC);
        assert!(raw >= 0);
        assert_eq!(libc::ftruncate(raw, 256 * 4 * 128), 0);
        OwnedFd::from_raw_fd(raw)
    };
    let mut bogus = dma_buf(&bo);
    bogus.fd = memfd.as_fd();
    let error = importer.import(BufferKey(1), &bogus).unwrap_err();
    assert!(matches!(error, Error::Egl { .. }), "{error}");

    // `/dev/null` reports no size: refused before EGL sees it.
    let null = std::fs::File::open("/dev/null").unwrap();
    let mut empty = dma_buf(&bo);
    empty.fd = null.as_fd();
    assert!(matches!(
        importer.import(BufferKey(1), &empty),
        Err(Error::InvalidDmaBuf { .. })
    ));

    // Nothing failed half-way into the cache.
    assert!(importer.is_empty());
    assert!(matches!(
        DrmFormat::try_from(u32::from_le_bytes(*b"NV12")),
        Err(Error::UnsupportedFormat { .. })
    ));
}

#[test]
fn refuses_another_gpu() {
    let Some(gpu) = Gpu::get() else { return };
    // A DRM node number no GPU uses (major 226, minor 250).
    let elsewhere = DrmNode::from_dev(libc::makedev(226, 250));
    let error = DmaBufImporter::new(&gpu.context, Some(elsewhere)).unwrap_err();
    assert!(matches!(error, Error::DeviceMismatch { .. }), "{error}");
    eprintln!("{error}");
    // The primary node of the same GPU is accepted like its render node.
    let pci = gpu.context.device().pci_bus_id().unwrap();
    if let Some(card) = std::fs::read_dir(format!("/sys/bus/pci/devices/{pci}/drm"))
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .find(|name| name.starts_with("card"))
        && let Ok(node) = DrmNode::from_path(format!("/dev/dri/{card}"))
    {
        DmaBufImporter::new(&gpu.context, Some(node)).unwrap();
    }
}

#[test]
fn fence_wait_times_out_on_unsignalled_buffers_only() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let bo = gpu
        .allocate(64, 64, DrmFormat::Argb8888, Modifier::LINEAR)
        .unwrap();
    importer.import(BufferKey(1), &dma_buf(&bo)).unwrap();
    // No producer work pending: an immediate wait succeeds.
    importer
        .map(BufferKey(1), FenceWait::Implicit(Duration::ZERO))
        .unwrap();
    importer
        .map(BufferKey(1), FenceWait::AlreadySignalled)
        .unwrap();
}

/// Per-frame cost of the steady state at 1920x1080: map (fence poll + frame
/// lookup), and map + device copy into a pooled buffer.
#[test]
fn per_frame_cost() {
    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&gpu.context).unwrap();
    let (width, height) = (1920, 1080);
    let pool = FramePool::new(&gpu.context, width as usize * 4, height as usize);
    for modifier in [Modifier::LINEAR, Gpu::tiled_modifier(&importer)] {
        let bo = gpu
            .allocate(width, height, DrmFormat::Xrgb8888, modifier)
            .unwrap();
        let key = BufferKey(u128::from(modifier.0));
        let start = Instant::now();
        importer.import(key, &dma_buf(&bo)).unwrap();
        let import = start.elapsed();

        let rounds = 1000;
        let start = Instant::now();
        for _ in 0..rounds {
            let frame = importer.map(key, FENCE).unwrap();
            std::hint::black_box(frame.view());
        }
        let map = start.elapsed() / rounds;

        let rounds = 300;
        let start = Instant::now();
        for _ in 0..rounds {
            let frame = importer.map(key, FENCE).unwrap();
            let buffer = frame.copy_to_pool(&pool, &stream).unwrap();
            stream.synchronize().unwrap();
            drop(buffer);
        }
        let copy = start.elapsed() / rounds;
        eprintln!(
            "1920x1080 modifier {:#018x}: import {import:?}, map {map:?}, map + device copy + sync {copy:?}",
            modifier.0
        );
    }
}

/// Whether `fd` (a `sync_file`) has signalled.
fn signalled(fd: &OwnedFd) -> bool {
    let mut poll_fd = libc::pollfd {
        fd: std::os::fd::AsRawFd::as_raw_fd(fd),
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one valid pollfd.
    unsafe { libc::poll(&raw mut poll_fd, 1, 0) == 1 }
}

/// Pending GPU work on an NVIDIA-rendered buffer: the implicit wait
/// (`FenceWait::Implicit`) returns at once although the work is still
/// queued (the driver attaches no fence to the DMA-BUF), while waiting on
/// the work's explicit fence through a DRM timeline syncobj — what the
/// PipeWire capture does with `SPA_META_SyncTimeline` — blocks until it
/// completed.
#[test]
fn implicit_wait_ignores_pending_nvidia_work_explicit_wait_does_not() {
    use ferricast_core::{SyncWait, SyncobjDevice};

    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let tiled = Gpu::tiled_modifier(&importer);
    let bo = gpu
        .allocate(3840, 2160, DrmFormat::Argb8888, tiled)
        .unwrap();
    let key = BufferKey(1);
    importer.import(key, &dma_buf(&bo)).unwrap();
    let clears = 3000;

    // Implicit: returns while the clears are still running.
    let work = match gpu.render_pending(&bo, clears) {
        Ok(work) => work,
        Err(reason) => {
            eprintln!("skipping: {reason}");
            return;
        }
    };
    let start = Instant::now();
    let _mapped = importer
        .map(key, FenceWait::Implicit(Duration::from_secs(2)))
        .unwrap();
    let implicit = start.elapsed();
    let pending_after_implicit = !signalled(&work);
    let device = SyncobjDevice::open_render_node(None).unwrap();
    let timeline = device.create().unwrap();
    timeline
        .import_sync_file(1, std::os::fd::AsFd::as_fd(&work))
        .unwrap();
    assert_eq!(
        timeline.wait(1, Duration::from_secs(5)).unwrap(),
        SyncWait::Signalled
    );
    eprintln!(
        "implicit wait returned after {implicit:?}, work still pending: {pending_after_implicit}"
    );
    assert!(
        pending_after_implicit,
        "the work finished within the implicit wait ({implicit:?}): too little work to tell"
    );

    // Explicit: the timeline wait lasts until the work is done.
    let work = gpu.render_pending(&bo, clears).unwrap();
    let timeline = device.create().unwrap();
    timeline
        .import_sync_file(1, std::os::fd::AsFd::as_fd(&work))
        .unwrap();
    let start = Instant::now();
    assert_eq!(
        timeline.wait(1, Duration::from_secs(5)).unwrap(),
        SyncWait::Signalled
    );
    let explicit = start.elapsed();
    assert!(
        signalled(&work),
        "the timeline signalled before the work's fence"
    );
    // Then the frame is lent without another wait.
    let _mapped = importer.map(key, FenceWait::AlreadySignalled).unwrap();
    eprintln!("explicit timeline wait lasted {explicit:?} ({clears} clears of 3840x2160)");
    assert!(
        explicit > implicit * 10 && explicit > Duration::from_millis(5),
        "explicit wait {explicit:?} vs implicit {implicit:?}"
    );
}

/// Unregistering after an event recorded behind the last read, in one
/// batch, then clearing the rest without a context-wide wait.
#[test]
fn forget_all_waits_for_an_event_and_clear_with_for_nothing() {
    use ferricast_nvcodec::cuda::Event;
    use ferricast_nvcodec::interop::Quiesce;

    let Some(gpu) = Gpu::get() else { return };
    let mut importer = DmaBufImporter::new(&gpu.context, Some(gpu.node)).unwrap();
    let stream = Stream::new(&gpu.context).unwrap();
    let tiled = Gpu::tiled_modifier(&importer);
    let bos: Vec<_> = (0..3)
        .map(|_| gpu.allocate(64, 32, DrmFormat::Argb8888, tiled).unwrap())
        .collect();
    for (index, bo) in bos.iter().enumerate() {
        gpu.write(bo, &pattern(64, 32, index as u8));
        importer
            .import(BufferKey(index as u128), &dma_buf(bo))
            .unwrap();
    }
    let pool = FramePool::new(importer.context(), 64 * 4, 32);
    let copied = {
        let frame = importer.map(BufferKey(0), FENCE).unwrap();
        frame.copy_to_pool(&pool, &stream).unwrap()
    };
    let event = Event::new(&gpu.context).unwrap();
    event.record(&stream).unwrap();
    let forgotten = importer.forget_all(
        [BufferKey(0), BufferKey(1), BufferKey(7)],
        Quiesce::after(&event),
    );
    assert_eq!(forgotten, 2);
    assert!(!importer.contains(BufferKey(0)) && importer.contains(BufferKey(2)));
    stream.synchronize().unwrap();
    drop(copied);
    // SAFETY: the only work that read a buffer (the copy) is done.
    importer.clear_with(unsafe { Quiesce::idle() });
    assert!(importer.is_empty());
}

/// GBM, EGL and GLES, loaded at runtime like the crate loads its drivers.
#[path = "support/dmabuf.rs"]
mod support;
