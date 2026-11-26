//! Screen and audio capture against the live desktop, through the istmo
//! screen-capture plugin: source listing and thumbnails, DMA-BUF frames
//! into NVENC without a copy (normal and rotated outputs), the Vulkan
//! readback against a compositor screenshot, CPU frames at the hold
//! limit, and audio on the wall clock.
//!
//! Needs a Wayland session with wlr-screencopy (niri here) and opens no
//! dialog. Gated on `FERRICAST_TEST_CAPTURE_LIVE=1`; the NVENC checks
//! also need the `nvenc` feature and an NVIDIA GPU. Run one at a time:
//!
//! ```text
//! FERRICAST_TEST_CAPTURE_LIVE=1 cargo test --test capture_live -- --nocapture --test-threads 1
//! ```
//!
//! `ISTMO_SCREEN_CAPTURE_FULL_COPY=1` asks for every frame instead of damaged
//! ones, which makes the frame counts independent of what the screen
//! shows. `FERRICAST_TEST_MONITORS=DP-3,HDMI-A-1` limits the checks to
//! those outputs; the readback check runs only on the still output named
//! by `FERRICAST_TEST_STILL_MONITOR`.

#![cfg(target_os = "linux")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "frame counts and PSNR over small, positive values"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ferricast::capture::{NativeAudioCapture, NativeCapture, auto_enumerator};
use ferricast::{
    AudioCapture, AudioCaptureConfig, AudioMuteHandle, AudioSource, CaptureConfig, CaptureSource,
    CapturedFrame, DmaBufImporter, DmaBufPlane, EnumerationCapability, MonitorInfo, PixelFormat,
    ScreenCapture,
};

fn enabled() -> bool {
    let on = std::env::var_os("FERRICAST_TEST_CAPTURE_LIVE").is_some_and(|v| v == "1");
    if !on {
        eprintln!("skipped: set FERRICAST_TEST_CAPTURE_LIVE=1 in a live Wayland session");
    }
    on
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros() as u64)
}

fn runtime() -> tokio::runtime::Runtime {
    // `RUST_LOG=ferricast_capture=debug,istmo_screen_capture=debug` shows
    // what the plugin negotiated.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Every test here runs alone: they share the screen, and the NVENC ones
/// the GPU (driver 595 can crash a process destroying an NVENC session
/// while another thread uses one, see `design/NVCODEC.md`).
fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static SCREEN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SCREEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

async fn monitors() -> Vec<MonitorInfo> {
    let enumerator = auto_enumerator();
    assert!(
        enumerator
            .capabilities()
            .contains(&EnumerationCapability::Monitors),
        "{:?}",
        enumerator.capabilities()
    );
    let mut monitors = enumerator.list_monitors().await.unwrap();
    // `FERRICAST_TEST_MONITORS=DP-3,HDMI-A-1` limits the checks to those.
    if let Ok(only) = std::env::var("FERRICAST_TEST_MONITORS") {
        monitors.retain(|m| only.split(',').any(|id| id == m.id));
    }
    monitors
}

/// Counts the readbacks of the frames it is attached to.
struct Counting {
    inner: Arc<dyn DmaBufImporter>,
    readbacks: Arc<AtomicUsize>,
}

impl DmaBufImporter for Counting {
    fn readback(
        &self,
        plane: &DmaBufPlane,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> ferricast::Result<Bytes> {
        self.readbacks.fetch_add(1, Ordering::SeqCst);
        self.inner.readback(plane, width, height, format)
    }
}

#[derive(Debug, Default)]
struct Tally {
    gpu: usize,
    cpu: usize,
    readbacks: usize,
    stale: usize,
}

#[test]
fn sources_are_listed_with_thumbnails_and_no_dialog() {
    if !enabled() {
        return;
    }
    let _screen = one_at_a_time();
    runtime().block_on(async {
        let enumerator = auto_enumerator();
        eprintln!(
            "backend {} capabilities {:?}",
            enumerator.backend_name(),
            enumerator.capabilities()
        );
        let monitors = enumerator.list_monitors().await.unwrap();
        assert!(!monitors.is_empty());
        for monitor in &monitors {
            eprintln!("monitor {monitor:?}");
            let png = enumerator
                .monitor_thumbnail(&monitor.id, 320, 320)
                .await
                .unwrap();
            assert!(png.starts_with(b"\x89PNG"), "{}", monitor.id);
        }
        if enumerator
            .capabilities()
            .contains(&EnumerationCapability::Windows)
        {
            let windows = enumerator.list_windows().await.unwrap();
            eprintln!("{} windows, e.g. {:?}", windows.len(), windows.first());
        }
    });
}

/// `count` frames of `monitor`, each re-stamped check against the wall
/// clock, with the frames' readbacks counted.
async fn capture(
    monitor: &str,
    config: CaptureConfig,
    count: usize,
    mut each: impl FnMut(CapturedFrame),
) -> Tally {
    let mut capture = NativeCapture::new();
    capture
        .start(
            CaptureSource::FullScreen {
                monitor: Some(monitor.into()),
            },
            config,
        )
        .await
        .unwrap();
    let readbacks = Arc::new(AtomicUsize::new(0));
    let mut tally = Tally::default();
    for _ in 0..count {
        let mut frame = tokio::time::timeout(Duration::from_secs(5), capture.next_frame())
            .await
            .expect("a frame within 5 s (move the mouse on an idle screen)")
            .unwrap();
        // Capture timestamps are UNIX-epoch µs, the segmenter's clock.
        if now_us().abs_diff(frame.timestamp_us()) > 200_000 {
            tally.stale += 1;
        }
        match &mut frame {
            CapturedFrame::Gpu(gpu) => {
                tally.gpu += 1;
                let inner = gpu.importer.take().expect("GPU frames carry the importer");
                gpu.importer = Some(Arc::new(Counting {
                    inner,
                    readbacks: Arc::clone(&readbacks),
                }));
            }
            CapturedFrame::Cpu(_) => tally.cpu += 1,
            CapturedFrame::Native(_) => panic!("no native frames on Linux"),
        }
        each(frame);
    }
    capture.stop().await.unwrap();
    tally.readbacks = readbacks.load(Ordering::SeqCst);
    tally
}

#[test]
fn rotated_outputs_reach_cpu_consumers_upright() {
    if !enabled() {
        return;
    }
    let _screen = one_at_a_time();
    runtime().block_on(async {
        for monitor in monitors().await {
            let config = CaptureConfig {
                fps: 30,
                ..CaptureConfig::default()
            };
            let mut size = None;
            let tally = capture(&monitor.id, config, 30, |frame| {
                size = Some((frame.width(), frame.height()));
                let raw = frame.into_cpu().unwrap();
                assert_eq!((raw.width, raw.height), size.unwrap());
                assert!(raw.data.iter().any(|b| *b != 0), "not a blank frame");
            })
            .await;
            eprintln!("{}: {tally:?}, upright {size:?}", monitor.id);
            let (w, h) = size.unwrap();
            // The listing's logical geometry is upright too.
            assert_eq!(
                w >= h,
                monitor.geometry.width >= monitor.geometry.height,
                "{}",
                monitor.id
            );
            assert_eq!(tally.stale, 0, "timestamps on the wall clock");
        }
    });
}

/// Needs an upright output whose picture holds still for a moment, named
/// in `FERRICAST_TEST_STILL_MONITOR` (a playing video differs between the
/// two copies by itself); skipped without it.
#[test]
fn vulkan_readback_matches_a_compositor_screenshot() {
    if !enabled() {
        return;
    }
    let Ok(still) = std::env::var("FERRICAST_TEST_STILL_MONITOR") else {
        eprintln!("skipped: set FERRICAST_TEST_STILL_MONITOR to an output whose picture is still");
        return;
    };
    let _screen = one_at_a_time();
    runtime().block_on(async {
        let monitors = monitors().await;
        let monitor = monitors
            .iter()
            .find(|m| m.id == still)
            .expect("FERRICAST_TEST_STILL_MONITOR names a listed monitor");
        assert!(
            monitor.geometry.width >= monitor.geometry.height,
            "{still} is rotated: the readback is compared unturned"
        );
        let config = CaptureConfig {
            show_cursor: false,
            ..CaptureConfig::default()
        };
        let mut last = None;
        let tally = capture(&monitor.id, config, 3, |frame| last = Some(frame)).await;
        assert_eq!(tally.cpu, 0, "{tally:?}");
        let raw = last.unwrap().into_cpu().unwrap();
        // The compositor's own copy into shared memory, no Vulkan.
        let png = auto_enumerator()
            .monitor_thumbnail(&monitor.id, raw.width, raw.height)
            .await
            .unwrap();
        let shot = image::load_from_memory(&png).unwrap().to_rgba8();
        assert_eq!((shot.width(), shot.height()), (raw.width, raw.height));
        let (mut same, mut total, mut squared) = (0usize, 0usize, 0f64);
        let order: [usize; 3] = match raw.format {
            PixelFormat::Bgra => [2, 1, 0],
            _ => [0, 1, 2],
        };
        for (y, row) in raw.data.chunks(raw.stride as usize).enumerate() {
            for (x, px) in row[..(raw.width * 4) as usize].chunks_exact(4).enumerate() {
                let theirs = shot.get_pixel(x as u32, y as u32).0;
                let ours = [px[order[0]], px[order[1]], px[order[2]]];
                total += 1;
                if ours == theirs[..3] {
                    same += 1;
                }
                squared += ours
                    .iter()
                    .zip(&theirs[..3])
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum::<f64>();
            }
        }
        let mse = squared / (total as f64 * 3.0);
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        eprintln!(
            "{}: readback vs screenshot: {:.3} % identical pixels, PSNR {psnr:.1} dB",
            monitor.id,
            same as f64 * 100.0 / total as f64
        );
        assert!(
            psnr > 35.0,
            "readback differs from the screenshot: {psnr} dB"
        );
    });
}

#[test]
fn held_frames_keep_flowing_as_cpu_frames() {
    if !enabled() {
        return;
    }
    let _screen = one_at_a_time();
    if std::env::var_os("ISTMO_SCREEN_CAPTURE_FULL_COPY").is_none_or(|v| v != "1") {
        eprintln!("note: without ISTMO_SCREEN_CAPTURE_FULL_COPY=1, keep the screen changing");
    }
    runtime().block_on(async {
        // An upright output: a rotated one is read back for this consumer
        // anyway, so its frames are CPU frames from the start.
        let monitors = monitors().await;
        let monitor = monitors
            .iter()
            .find(|m| m.geometry.width >= m.geometry.height)
            .expect("an upright monitor");
        let mut held = Vec::new();
        let tally = capture(&monitor.id, CaptureConfig::default(), 40, |frame| {
            held.push(frame);
        })
        .await;
        eprintln!("{}: holding every frame: {tally:?}", monitor.id);
        assert!(tally.gpu > 0 && tally.cpu > 0, "{tally:?}");
        let cpu = held
            .iter()
            .find_map(|frame| match frame {
                CapturedFrame::Cpu(raw) => Some(raw),
                _ => None,
            })
            .unwrap();
        assert!(cpu.data.iter().any(|b| *b != 0));
    });
}

#[cfg(feature = "nvenc")]
#[test]
fn nvenc_encodes_every_output_zero_copy() {
    use ferricast::encoder::h264::H264Encoder;
    use ferricast::{BackendId, BackendSelection, Codec, EncoderConfig, VideoEncoder};

    if !enabled() {
        return;
    }
    let _screen = one_at_a_time();
    runtime().block_on(async {
        for monitor in monitors().await {
            let (width, height) = (monitor.geometry.width, monitor.geometry.height);
            let encoder_config = EncoderConfig {
                codec: Codec::H264,
                width,
                height,
                bitrate_kbps: 8_000,
                fps: 60,
                pixel_format: PixelFormat::Bgra,
                backend: BackendSelection::Fixed(BackendId::Nvenc),
                ..EncoderConfig::default()
            };
            let mut encoder = H264Encoder::default();
            if let Err(e) = encoder.configure(&encoder_config) {
                eprintln!("skipped: no NVENC ({e})");
                return;
            }
            let config = CaptureConfig {
                fps: 60,
                consumer_modifiers: ferricast::encoder::dmabuf_modifiers(&encoder_config),
                consumer_orients: ferricast::encoder::orients_gpu_frames(&encoder_config),
                ..CaptureConfig::default()
            };
            let orients = config.consumer_orients;
            let started = Instant::now();
            let mut packets = 0;
            let tally = capture(&monitor.id, config, 180, |frame| {
                encoder.encode(frame).unwrap();
                packets += 1;
            })
            .await;
            let fps = 180.0 / started.elapsed().as_secs_f64();
            eprintln!(
                "{} ({width}x{height}, orients {orients}): {tally:?}, {fps:.1} fps",
                monitor.id
            );
            assert_eq!(packets, 180);
            assert_eq!(tally.stale, 0, "timestamps on the wall clock");
            if orients {
                assert_eq!(tally.cpu, 0, "{}: {tally:?}", monitor.id);
                assert_eq!(tally.readbacks, 0, "{}: {tally:?}", monitor.id);
            }
        }
    });
}

#[test]
fn audio_follows_the_wall_clock_and_mutes_on_time() {
    if !enabled() {
        return;
    }
    let _screen = one_at_a_time();
    runtime().block_on(async {
        let mute = AudioMuteHandle::default();
        let mut capture = NativeAudioCapture::new();
        capture
            .start(
                AudioSource::DefaultMonitor,
                AudioCaptureConfig::default(),
                mute.clone(),
            )
            .await
            .unwrap();
        let started = Instant::now();
        let (mut chunks, mut samples, mut first) = (0usize, 0u64, None);
        let mut muted_ok = true;
        while started.elapsed() < Duration::from_secs(3) {
            if started.elapsed() > Duration::from_secs(2) {
                mute.set_muted(true);
            }
            let frame = tokio::time::timeout(Duration::from_secs(2), capture.next_frame())
                .await
                .expect("audio within 2 s")
                .unwrap();
            assert_eq!(frame.codec, ferricast::AudioCodec::Pcm);
            assert_eq!(frame.channels, 2);
            assert_eq!(frame.sample_rate, 48_000);
            assert!(
                now_us().abs_diff(frame.timestamp_us) < 500_000,
                "audio timestamps are UNIX-epoch µs"
            );
            let first_us = *first.get_or_insert(frame.timestamp_us);
            // The clock follows the sample count (re-anchoring on gaps).
            let expected = first_us + samples * 1_000_000 / 48_000;
            if frame.timestamp_us.abs_diff(expected) > 100_000 {
                eprintln!(
                    "re-anchored by {} µs",
                    frame.timestamp_us.abs_diff(expected)
                );
            }
            if mute.is_muted() {
                muted_ok &= frame.data.iter().all(|b| *b == 0);
            }
            samples += (frame.data.len() / 4) as u64;
            chunks += 1;
        }
        capture.stop().await.unwrap();
        eprintln!(
            "audio: {chunks} chunks, {:.2} s of samples in 3 s",
            samples as f64 / 48_000.0
        );
        assert!(chunks > 10);
        assert!(muted_ok, "muted chunks are silent");
        assert!((2.5..3.5).contains(&(samples as f64 / 48_000.0)));
    });
}
