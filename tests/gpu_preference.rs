//! The GPU preference on real hardware: which backend opens, and on which
//! GPU, for each `GpuPreference`. Written for this repository's test
//! machine (one NVIDIA GPU); every test prints why it skips elsewhere.

#![cfg(all(feature = "nvenc", feature = "nvdec", target_os = "linux"))]

use bytes::Bytes;
use ferricast_core::{
    BackendId, BackendSelection, CapturedFrame, Codec, CodecError, DecoderConfig, EncoderConfig,
    FrameMemory, Gpu, GpuId, GpuPreference, GpuVendor, Gpus, PixelFormat, RawFrame, VideoDecoder,
    VideoEncoder,
};
use ferricast_decoder::H264Decoder;
use ferricast_encoder::h264::H264Encoder;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The machine's NVIDIA GPU, when it has exactly one GPU and it is
/// NVIDIA's (the layout these tests assert on).
fn single_nvidia() -> Option<Gpu> {
    let gpus = Gpus::scan();
    match gpus.as_slice() {
        [gpu] if gpu.vendor == GpuVendor::NVIDIA => Some(gpu.clone()),
        other => {
            eprintln!("skipped: needs exactly one GPU, an NVIDIA one; found {other:?}");
            None
        }
    }
}

fn config(gpu: GpuPreference, backend: BackendSelection) -> EncoderConfig {
    EncoderConfig {
        width: WIDTH,
        height: HEIGHT,
        fps: 30,
        bitrate_kbps: 2000,
        pixel_format: PixelFormat::Bgra,
        gpu,
        backend,
        ..EncoderConfig::default()
    }
}

fn frame() -> CapturedFrame {
    CapturedFrame::Cpu(RawFrame {
        width: WIDTH,
        height: HEIGHT,
        stride: WIDTH * 4,
        format: PixelFormat::Bgra,
        data: Bytes::from(vec![128; (WIDTH * HEIGHT * 4) as usize]),
        timestamp_us: 0,
        colour: None,
        hdr: None,
    })
}

/// Opens H.264 encode under `gpu` with automatic selection, encodes one
/// picture, and returns the backend that did it.
fn encoded_by(gpu: &GpuPreference) -> BackendId {
    let mut encoder = H264Encoder::default();
    encoder
        .configure(&config(gpu.clone(), BackendSelection::default()))
        .unwrap_or_else(|error| panic!("{gpu}: {error}"));
    let packet = encoder.encode(frame()).unwrap();
    assert!(packet.is_keyframe, "{gpu}: the first picture is an IDR");
    encoder.backend().expect("configured").0
}

#[test]
fn every_preference_naming_the_nvidia_gpu_picks_nvenc() {
    let _gpu = one_at_a_time();
    let Some(nvidia) = single_nvidia() else {
        return;
    };
    let pci = nvidia.pci.expect("a PCI GPU");
    let preferences = [
        GpuPreference::Auto,
        GpuPreference::Discrete,
        // No integrated GPU here: a preference, not a pin.
        GpuPreference::Integrated,
        GpuPreference::SameAsCapture,
        GpuPreference::Device(GpuId::Node(nvidia.render_node.clone())),
        GpuPreference::Device(GpuId::Pci(pci)),
    ];
    for preference in preferences {
        assert_eq!(encoded_by(&preference), BackendId::Nvenc, "{preference}");
    }
}

#[test]
fn a_missing_gpu_leaves_the_hardware_backends_out() {
    let _gpu = one_at_a_time();
    if single_nvidia().is_none() {
        return;
    }
    let missing: GpuPreference = "/dev/dri/renderD250".parse().unwrap();
    // Automatic selection drops every GPU-bound backend: software encodes.
    assert_eq!(encoded_by(&missing), BackendId::X264);
    // A fixed backend fails, naming the preference.
    let mut encoder = H264Encoder::default();
    let error = encoder
        .configure(&config(missing, BackendId::Nvenc.into()))
        .unwrap_err();
    let CodecError::BackendUnavailable { backend, reason } = &error else {
        panic!("unexpected error: {error}");
    };
    assert_eq!(*backend, BackendId::Nvenc);
    assert!(reason.contains("/dev/dri/renderD250"), "{reason}");
}

#[test]
fn decoders_follow_the_preference_too() {
    let _gpu = one_at_a_time();
    let Some(nvidia) = single_nvidia() else {
        return;
    };
    let decoder_on = |gpu: GpuPreference| {
        let mut decoder = H264Decoder::default();
        decoder
            .configure(&DecoderConfig {
                codec: Codec::H264,
                width: WIDTH,
                height: HEIGHT,
                pixel_format: PixelFormat::Nv12,
                low_latency: true,
                colour: None,
                hdr: None,
                memory: FrameMemory::Cpu,
                backend: BackendSelection::default(),
                gpu: gpu.clone(),
            })
            .unwrap_or_else(|error| panic!("{gpu}: {error}"));
        decoder.backend().expect("configured").0
    };
    assert_eq!(
        decoder_on(GpuPreference::Device(GpuId::Node(nvidia.render_node))),
        BackendId::Nvdec
    );
    assert_eq!(
        decoder_on("renderD250".parse().unwrap()),
        BackendId::OpenH264
    );
}
