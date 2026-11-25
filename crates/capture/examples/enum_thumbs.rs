//! Exercises the source enumerator, thumbnail and wayland-direct
//! capture paths the GUI picker uses, against the live compositor.
//!
//! ```text
//! cargo run -p ferricast-capture --example enum_thumbs -- [iterations]
//! ```
//!
//! Reports per-phase timings, empty listings, failures and calls that
//! exceed their deadline. Opens no windows.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ferricast_core::{CaptureConfig, CaptureSource, ScreenCapture, SourceEnumerator};

const CALL_DEADLINE: Duration = Duration::from_secs(10);

#[derive(Default)]
struct Tally {
    ok: usize,
    empty: usize,
    failed: usize,
    hung: usize,
    slowest: Duration,
}

impl Tally {
    fn record<T>(&mut self, started: Instant, outcome: Option<&Result<T, String>>, empty: bool) {
        self.slowest = self.slowest.max(started.elapsed());
        match outcome {
            None => self.hung += 1,
            Some(Err(_)) => self.failed += 1,
            Some(Ok(_)) if empty => self.empty += 1,
            Some(Ok(_)) => self.ok += 1,
        }
    }

    fn print(&self, what: &str) {
        println!(
            "{what:<28} ok={:<4} empty={:<4} failed={:<4} hung={:<4} slowest={:?}",
            self.ok, self.empty, self.failed, self.hung, self.slowest
        );
    }
}

async fn bounded<T, E: std::fmt::Display>(
    what: &str,
    fut: impl Future<Output = Result<T, E>>,
) -> Option<Result<T, String>> {
    match tokio::time::timeout(CALL_DEADLINE, fut).await {
        Ok(Ok(v)) => Some(Ok(v)),
        Ok(Err(e)) => {
            println!("  {what}: error: {e}");
            Some(Err(e.to_string()))
        }
        Err(_) => {
            println!("  {what}: no answer after {CALL_DEADLINE:?}");
            None
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_thread_names(true)
        .init();
    let iterations: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    println!(
        "WAYLAND_DISPLAY={:?} XDG_RUNTIME_DIR={:?}",
        std::env::var_os("WAYLAND_DISPLAY"),
        std::env::var_os("XDG_RUNTIME_DIR")
    );

    fresh_enumerators(iterations).await;
    let enumerator = ferricast_capture::auto_enumerator();
    println!(
        "backend={} capabilities={:?}",
        enumerator.backend_name(),
        enumerator.capabilities()
    );
    thumbnails(&enumerator, iterations).await;
    direct_capture(&enumerator).await;
}

/// A freshly built enumerator must list every monitor right away: the
/// picker can open before any later event arrives.
async fn fresh_enumerators(iterations: usize) {
    let mut monitors = Tally::default();
    let mut counts = Vec::new();
    for _ in 0..iterations {
        let started = Instant::now();
        let enumerator = ferricast_capture::auto_enumerator();
        let listed = bounded("list_monitors", enumerator.list_monitors()).await;
        let n = listed
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .map_or(0, Vec::len);
        let zero_geometry = listed
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .is_some_and(|ms| ms.iter().any(|m| m.geometry.width == 0));
        counts.push((n, zero_geometry));
        monitors.record(started, listed.as_ref(), n == 0 || zero_geometry);
    }
    monitors.print("fresh enumerator monitors");
    counts.sort_unstable();
    counts.dedup();
    println!("  distinct (count, any 0-width) results: {counts:?}");
}

async fn thumbnails(enumerator: &Arc<dyn SourceEnumerator>, iterations: usize) {
    let mut lists = Tally::default();
    let mut monitor_thumbs = Tally::default();
    let mut window_thumbs = Tally::default();
    for _ in 0..iterations {
        let started = Instant::now();
        let monitors = bounded("list_monitors", enumerator.list_monitors()).await;
        let windows = bounded("list_windows", enumerator.list_windows()).await;
        let mons = monitors
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .cloned()
            .unwrap_or_default();
        let wins = windows
            .as_ref()
            .and_then(|r| r.as_ref().ok())
            .cloned()
            .unwrap_or_default();
        lists.record(started, monitors.as_ref(), mons.is_empty());

        // Every card requests its preview at once, like the picker.
        let mut set = tokio::task::JoinSet::new();
        for m in &mons {
            let (e, id) = (enumerator.clone(), m.id.clone());
            set.spawn(async move {
                let started = Instant::now();
                let r = bounded(&format!("monitor {id}"), e.monitor_thumbnail(&id, 220, 124)).await;
                (true, started, r)
            });
        }
        for w in &wins {
            let (e, id) = (enumerator.clone(), w.id.clone());
            set.spawn(async move {
                let started = Instant::now();
                let r = bounded(&format!("window {id}"), e.window_thumbnail(&id, 220, 124)).await;
                (false, started, r)
            });
        }
        while let Some(joined) = set.join_next().await {
            let (is_monitor, started, r) = joined.expect("thumbnail task panicked");
            let empty = matches!(&r, Some(Ok(b)) if b.is_empty());
            if is_monitor {
                monitor_thumbs.record(started, r.as_ref(), empty);
            } else {
                window_thumbs.record(started, r.as_ref(), empty);
            }
        }
    }
    lists.print("list (monitors)");
    monitor_thumbs.print("monitor thumbnails");
    window_thumbs.print("window thumbnails");
}

async fn direct_capture(enumerator: &Arc<dyn SourceEnumerator>) {
    let monitors = enumerator.list_monitors().await.unwrap_or_default();
    let Some(monitor) = monitors.first() else {
        println!("direct capture: no monitor to capture");
        return;
    };
    let config = CaptureConfig {
        fps: 30,
        width: None,
        height: None,
        show_cursor: true,
        consumer_modifiers: ferricast_core::DmaBufModifiers::default(),
        consumer_orients: false,
    };
    for _ in 0..3 {
        let mut capture = ferricast_capture::WaylandDirectCapture::new();
        let started = Instant::now();
        let source = CaptureSource::FullScreen {
            monitor: Some(monitor.id.clone()),
        };
        let r = bounded("direct start", capture.start(source, config.clone())).await;
        println!(
            "direct start {}: {:?} in {:?}",
            monitor.id,
            r.as_ref().map(std::result::Result::is_ok),
            started.elapsed()
        );
        if matches!(r, Some(Ok(()))) {
            for _ in 0..5 {
                let started = Instant::now();
                let f = bounded("next_frame", capture.next_frame()).await;
                println!(
                    "  frame: {:?} in {:?}",
                    f.as_ref().map(std::result::Result::is_ok),
                    started.elapsed()
                );
            }
        }
        let started = Instant::now();
        let r = bounded("direct stop", capture.stop()).await;
        println!(
            "direct stop: {:?} in {:?}",
            r.map(|r| r.is_ok()),
            started.elapsed()
        );
    }
    let mut capture = ferricast_capture::WaylandDirectCapture::new();
    let started = Instant::now();
    let r = bounded(
        "direct window start",
        capture.start(CaptureSource::Window { identifier: None }, config),
    )
    .await;
    println!(
        "direct window start: {:?} in {:?}",
        r.map(|r| r.is_ok()),
        started.elapsed()
    );
}
