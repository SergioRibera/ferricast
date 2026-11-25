//! What automatic backend selection costs when an H.264 decoder opens:
//! every candidate probed and tried in order (cold: no cache, probe
//! verdicts forgotten) against the remembered winner tried first with
//! the probe verdicts kept (warm).
//!
//! The gap is the time the candidates ahead of the winner take to
//! refuse, so it shows with hardware backends compiled in that cannot
//! run here: `cargo bench -p ferricast-decoder --bench selection
//! --features vaapi-decode,nvdec-decode`. With openh264 alone both
//! numbers are the same.

use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use ferricast_codec::{MemoryCache, NoCache, SelectionCache, cache};
use ferricast_core::{
    BackendSelection, Codec, DecoderConfig, FrameMemory, PixelFormat, VideoDecoder,
};
use ferricast_decoder::H264Decoder;

fn config() -> DecoderConfig {
    DecoderConfig {
        gpu: Default::default(),
        codec: Codec::H264,
        width: 1920,
        height: 1080,
        pixel_format: PixelFormat::Nv12,
        low_latency: true,
        colour: None,
        hdr: None,
        memory: FrameMemory::Cpu,
        backend: BackendSelection::default(),
    }
}

fn open(cache: Arc<dyn SelectionCache>) {
    let mut decoder = H264Decoder::with_cache(cache);
    decoder
        .configure(&config())
        .expect("some H.264 decoder of this build opens here");
}

fn selection(c: &mut Criterion) {
    c.bench_function("h264 decoder open, cold", |b| {
        b.iter(|| {
            cache::reset_probes();
            open(Arc::new(NoCache));
        });
    });
    let warm: Arc<dyn SelectionCache> = Arc::new(MemoryCache::default());
    open(Arc::clone(&warm));
    c.bench_function("h264 decoder open, warm", |b| {
        b.iter(|| open(Arc::clone(&warm)));
    });
}

criterion_group!(benches, selection);
criterion_main!(benches);
