# Codecs: backends, selection, istmo plugin

Decision log for the encoder / decoder stack. Append; do not rewrite
history. "Rejected" entries are not to be re-proposed without new facts.

## Layering

```
ferricast-core      data + contracts: frames, ColourDescription, Codec / AudioCodec,
                    BackendId, CodecTarget, BackendSelection, CodecError,
                    VideoEncoder / VideoDecoder / AudioEncoder / AudioDecoder,
                    the four *Config types (wincode derives behind `wire`)
ferricast-codec     the engine: Registry + Candidate, plan/open, Selecting* bodies
                    (probation, replay, mid-stream hand-over), facade macros,
                    bitstream (Annex B / AVCC / parameter sets), AAC framing,
                    `plugin` (istmo, feature `istmo`)
ferricast-encoder   backends + facades (H264Encoder, H265Encoder, AacEncoder,
ferricast-decoder   AacEldEncoder, OpusEncoder / H264Decoder ... PcmDecoder);
                    platform/{windows,apple,android}.rs
ferricast           umbrella: re-exports, `codec_service()` (feature `istmo`)
```

- The codec traits stay in core (receiver, HLS segmenter and manager are
  generic over them) and return `CodecResult<T> = Result<T, CodecError>`.
  `FerricastError::Codec(CodecError)` carries them into the rest of the
  crate, matchable.
- `CodecError` holds owned data only (no `dyn Error` sources) so it
  crosses the istmo wire intact. Backend internals still produce
  `FerricastError` strings; the trait boundary maps them with
  `CodecError::from_backend(backend, operation)`, so every message names
  the backend and the step that failed.

## Selection

`BackendSelection` lives in each codec config (`EncoderConfig::backend`,
`DecoderConfig::backend`, ...):

| Variant | Behaviour |
|---|---|
| `Auto(AutoSelect)` (default) | probe compiled backends by `AccelerationPolicy` (prefer hardware / prefer software / hardware only / software only), minus `exclude`; first that opens wins; runtime fallback per `RuntimeFallback` |
| `Fixed(id)` | exactly that backend; its own error is returned; no fallback |
| `Ordered { backends, runtime_fallback }` | these, in this order |

- The user override without rebuilding is the job's environment
  variable, `FERRICAST_<CODEC>_<ENCODE|DECODE>_BACKEND` (legacy names
  `FERRICAST_H264_BACKEND`, `FERRICAST_H265_BACKEND`,
  `FERRICAST_AAC_DECODER` still read). It replaces `Auto` only: a
  developer's `Fixed` / `Ordered` wins. Text grammar: `auto`, `hardware`,
  `software`, `prefer-hardware`, `prefer-software`, a backend name, or a
  comma list. Invalid text fails `configure` with
  `CodecError::InvalidSelection` naming the variable.
- Behaviour change vs. the old facades: a forced hardware backend
  (`...=nvdec`) used to fall back to software silently; it is now
  `Fixed` and fails with the backend's error. `software` keeps working.
- Registries know the backends a build leaves out, so naming one fails
  with `BackendNotCompiled { feature }` (rebuild hint),
  `BackendUnavailable` ("only on Windows"...) or `BackendCannotHandle`.
  An empty build fails with `NoBackendCompiled { features }` at runtime
  instead of the old `compile_error!`s, whose messages had drifted.
- `Candidate::opt_in()` keeps a backend out of `Auto` (VA-API HEVC
  decode: slice submission unfinished).
- Decoder runtime fallback is the old NVDEC/VA-API behaviour, generalised
  to "next candidate": packets are kept until the first picture
  (probation, 64 by default) and replayed into the next backend on
  failure; `CodecError::UnsupportedStream` hands over at any time (the
  backend's finished pictures are kept, the next one starts at the
  failing packet with the remembered parameter sets in front). The next
  backend is on probation again. Encoders fall back when a
  reconfiguration is refused (old `fall_back_to_x264_or_pending`).

## Colour

- `ColourDescription` = H.273 primaries, transfer, matrix, range, chroma
  location. Strict `TryFrom<u8>` per field; `from_codes` reads reserved
  codes as unspecified (what H.264/H.265 tell decoders to do).
- `RawFrame::colour: Option<ColourDescription>` is what the picture
  signals. `YuvColour` (matrix + range) is only the renderer's
  conversion, derived with `ColourDescription::yuv(height)` (unspecified
  matrix → BT.709 for HD, BT.601 for SD, as players do).
- `EncoderConfig::colour` (default BT.709 limited) drives both the RGB →
  YUV conversion and the VUI the encoder writes; a backend that cannot
  produce it fails `configure` with `UnsupportedConfig`.
- `DecoderConfig::colour` / `VideoStreamInfo::colour` carry the
  container's signalling (MP4 `colr` / `vpcC`, Matroska `Colour`) for
  bitstreams or backends that give none.
- Backends that report no colour (openh264, NVDEC binding, rust_h265) get
  it from the SPS VUI (`bitstream::vui::ColourTracker`, H.264 and H.265
  SPS walked to the VUI), which wins over the container per field group
  (signal type, chroma location), like HDR metadata. Before this an HDR10
  stream over MPEG-TS / RTP (no container colour) came out with an
  unspecified transfer and was shown as SDR.
- NVDEC now reports the stream's colour itself (see "NVIDIA" below): its
  fields win, each unspecified one is filled from the container; only the
  chroma sample location still comes from `ColourTracker`.

## istmo plugin `ferricast.codec`

- Control plane only: `capabilities`, `resolve`, `open` (owned session
  handle), `close`, `request_keyframe`, `set_bitrate`. Errors are the
  typed `CodecError` (`IstmoCodecError::into_codec_error` on the client).
- Data plane is Rust: `CodecService::video_encoder(handle)` etc. return
  the session (`Arc<Mutex<Selecting*>>`); dropping the owned handle
  closes it through the runtime's release hook.
- Rejected: frames through istmo `#[stream]` / unary calls. istmo
  wincode-encodes every payload into a fresh `Vec<u8>` (≥ 2 copies per
  frame in-process, more through JNI / Swift) and generated hosts spawn
  a thread per call: not viable at 1080p60 on a phone.
- Hosts register with `runtime.register_host(CodecsHost::new(service))` (the
  published `0.1.1-alpha.6` macro has no typed-shortcut `register`).
- istmo is pre-1.0 (API and wire format churn); pinned with `=`.

## Platform backends

- Windows: Media Foundation MFTs (hardware async MFTs first, then the
  Microsoft software MFTs). Apple: VideoToolbox (video), AudioToolbox
  (AAC). Android: NDK `AMediaCodec` (API 28+), vendor codec first, Google
  `c2.android.*` software codecs after.
- Their cargo features are in the defaults: the dependencies are
  target-gated, so they are inert off their own OS. The C-library
  backends (x264, libfdk-aac, VA-API, NVENC) are not cross-built here; a
  mobile / Apple / Windows build turns the defaults off and picks the
  platform features plus the pure-Rust ones (`opus-rs`, `aac`,
  `hevc-decode`, `vp8-decode`, `vp9-decode`, `opus-decode`).

| Job | Media Foundation | VideoToolbox / AudioToolbox | MediaCodec |
|---|---|---|---|
| H.264 / H.265 encode | hw + sw (no sw HEVC in Windows) | hw (+ sw on macOS) | hw + `c2.android.*` |
| H.264 / H.265 decode | hw + sw | hw (+ sw on macOS) | hw + `c2.android.*` |
| VP8 / VP9 / AV1 decode | hw + sw (VP9/AV1/HEVC need Store extensions) | VP9 hw + sw (macOS 11+, not iOS), AV1 hw (M3 / A17+); no VP8 | hw + `c2.android.*` |
| HEVC Main10 / P010 (encode, decode) | hw | hw | hw |
| AAC-LC encode / decode | sw | sw (decode adds HE-AAC v1/v2) | sw |
| Opus encode / decode | — | — | sw (encode needs API 29) |
| AAC-ELD, ALAC | — | ELD encode + decode, ALAC decode | — |

- Design notes:
  - Encoders are 1-in / 1-out like x264: each `encode` waits (bounded:
    250 ms MediaCodec, 500 ms async MFT, VT completes up to the frame) for
    the frame's packet; a backend that holds the frame reports it as
    buffered and hands it out on a later call.
  - Parameter sets are kept from the codec-config output / format
    description and put in front of every keyframe, as x264 does.
    MediaCodec and VideoToolbox prime one black frame in `configure` so
    `get_headers` works before the first real frame.
  - Media Foundation sessions each own a worker thread in the MTA: every
    COM pointer is created and used on it, so nothing asserts `Send` for
    COM. The plumbing is `ferricast_codec::mf`, shared by encoder and
    decoder; `ferricast_codec::{apple, android}` are its Core Video / NDK
    counterparts (the three FFI modules of that crate, each `unsafe`
    with SAFETY comments).
  - MediaCodec: the hardware candidate refuses a device whose
    type-default codec is a Google software one, so `Auto` does not try
    the same software codec twice.
- Frames: see "Native surfaces" below.
- Verification: `nix develop -c cargo clippy --target <triple> -- -D
  warnings` for x86_64/aarch64-pc-windows-msvc,
  x86_64/aarch64-apple-darwin, aarch64-apple-ios(-sim) and
  x86_64/aarch64-linux-android compiles and lints them from Linux (the
  flake ships those std libraries; checking needs no SDK). Nothing in
  this list has been linked or run on a device: first runs on each OS
  are expected to surface media-type / format negotiation issues.

## Native surfaces (zero-copy)

- `CapturedFrame::Native(NativeFrame)` carries a platform media surface
  behind `Arc<dyn NativeSurface>` (kind, format, `readback`, `as_any`).
  Core stays free of platform types; they live in `ferricast_codec::{apple,
  android, mf}` (`PixelBuffer`, `HardwareBuffer`, `D3d11Texture`) and a
  consumer that knows one downcasts with `NativeFrame::surface::<T>()`.
  Everything else calls `into_cpu`. `CapturedFrame::Gpu` stays DMA-BUF.
- Decoders hand out native frames only when asked
  (`DecoderConfig::memory = FrameMemory::Native`); the default stays
  `Cpu`, so existing sinks are untouched.
- Encoders take a native surface directly when its size and bit depth
  match the session, and read it back otherwise:
  - VideoToolbox: `CVPixelBuffer` straight into the compression session.
  - Media Foundation: the first D3D11 texture rebuilds the session on
    that texture's device (packets kept, IDR forced, a refused device
    remembered); textures padded to the coded size (1920x1088) do not
    match a 1080 session and are read back.
  - MediaCodec: an `AHardwareBuffer` is locked and copied into the input
    buffer (one copy, no colour conversion). Rejected for now: input
    surfaces, which need an EGL renderer.
- Media Foundation's hardware decoder now always decodes on D3D11; its
  CPU frames are a staging readback.
- MediaCodec native output goes through an `AImageReader`; each image
  returns to the reader when its frame drops. A consumer holding every
  image back-pressures the decoder instead of failing (a failure would
  make `Auto` switch backends).

## 10-bit and HDR

- `PixelFormat::P010` and `BitDepth`; `EncoderConfig::{bit_depth, hdr}`,
  `DecoderConfig::hdr`, `VideoStreamInfo::hdr`, `RawFrame::hdr` /
  `NativeFrame::hdr` carry `HdrMetadata` (ST 2086 mastering display +
  CTA-861.3 light levels in SEI units; `to_sei` / `from_sei`, `from_av1`,
  `to_cta861` / `from_cta861` for the platform encodings).
- P010 in `DecoderConfig::pixel_format` is a preference, not an
  up-conversion: 8-bit streams still come out NV12, and 10-bit pictures
  are narrowed to NV12 for consumers that did not ask for P010.
- HDR precedence: the bitstream (SEI 137 / 144, AV1 metadata OBUs, the
  platform's attachments) wins over the container (`mdcv` / `clli`,
  `SmDm` / `CoLL`, Matroska `MasteringMetadata`), per half. MP4 `SmDm`
  uses AV1 fixed-point units in R, G, B order; `mdcv` uses the SEI layout
  in G, B, R order.
- Encoders that cannot write 10-bit or HDR refuse in `configure`
  (`UnsupportedConfig`) instead of dropping it: x264 (8-bit build), VA-API,
  NVENC. x264 writes the configured colour and HDR SEI; it used to
  hardcode BT.709. 8-bit encoders narrow P010 input to its high bytes.
- `BitDepth::Ten` means HEVC Main10 on all three platforms (Main10HDR10 on
  MediaCodec when `hdr` is set); H.264 + Ten fails. With
  `max_h265_profile = Main10` an 8-bit config encodes Main: bit depth
  decides the profile.
- Media Foundation keeps the mastering primaries in
  `MF_MT_CUSTOM_VIDEO_PRIMARIES`, which Microsoft documents as the ST 2086
  colour volume (not the content's primaries, which stay in
  `MF_MT_VIDEO_PRIMARIES`).
- The GUI renders P010 through Skia YUVA with 16-bit planes and converts
  BT.2020 / PQ / HLG to sRGB on the GPU (tone-mapped SDR, not HDR output).
- NVDEC decodes HEVC Main10 to P016 (= P010 for 10-bit) through the
  patched `shiguredo_nvcodec` in `vendor/` (`[patch.crates-io]`; changes
  marked `[ferricast]`). H.264 High 10 hands over: NVDEC has no 10-bit
  H.264, and neither has openh264, so Linux has no High 10 decoder. VA-API
  HEVC decode is a stub (no slice submission) and stays opt-in.
- `crates/decoder/tests/ten_bit.rs` checks real streams (x265 Main10 with
  B-frames and HDR10 SEI, SVT-AV1 10-bit with metadata OBUs, libvpx VP9
  profile 2) bit-exact against FFmpeg's P010 output, with the container's
  colour and HDR dropped where the bitstream carries them. Also checked
  once at 1080p (coded 1088): rust_h265 and NVDEC bit-exact, NVDEC about
  490 fps against 263.
- Superseded for NVIDIA (see the next section): NVDEC and NVENC run on
  `ferricast-nvcodec`, `vendor/` and the patch are gone, NVENC encodes
  HEVC Main10 + HDR10, and `ten_bit.rs` also checks NVDEC AV1 10-bit and
  VP9 profile 2 bit-exact, each clip through the facade, the software
  backend and NVDEC.

## NVIDIA (`ferricast-nvcodec`)

NVDEC and NVENC moved from `shiguredo_nvcodec` (and its patched copy in
`vendor/`) to ferricast's own binding, `crates/nvcodec` (design and driver
findings: `design/NVCODEC.md`). The binding is a dependency of
`ferricast-codec` (feature `cuda`) on Linux and Windows only; the
encoder's `nvenc` / `nvenc-hevc` and the decoder's `nvdec-decode` /
`nvdec-hevc-decode` features turn it on and are inert on other targets
(the backends are recorded as "only on Linux and Windows" there).

Shared pieces (`ferricast_codec::cuda`):

- `context()`: the primary CUDA context of device 0, shared by the
  decoders and encoders open at the same time (a decoded picture and the
  encoder taking it are on the same context) and released when the last
  one closes. A context marked lost (sticky CUDA error, Xid, resume) is
  replaced: the primary context is retained afresh, or, while the lost one
  is still held, a context of its own is created. Driver start-up failures
  are retried after a growing delay instead of being kept for the process.
- `CudaSurface`: an NVDEC picture in a pooled pitched CUDA buffer, handed
  out as `CapturedFrame::Native` of the new `SurfaceKind::CudaDevice`
  (NV12, P010 for streams deeper than 8 bits). `readback` downloads it;
  `CudaSurface::readbacks()` counts those, process-wide.
- `codec_error`: no driver / too old / no device are `BackendUnavailable`
  (selection moves on: NVENC → VA-API → x264, and the probe verdict is
  kept); an NVENC session limit, a closed encoder and any CUDA failure
  while probing or opening are the new `BackendBusy` (selection moves on,
  but the verdict is not memoized, so the next selection tries NVIDIA
  again); a lost device is `UnsupportedStream` while decoding and
  `BackendBusy` otherwise (an encoder hands over mid-stream), and the
  shared context is marked lost; what the GPU cannot do is
  `UnsupportedStream` while decoding (the next
  decoder takes the stream over, e.g. H.264 High 10, sizes under NVDEC's
  minimum, 4:2:2 / 4:4:4 output ferricast has no pixel format for) and
  `UnsupportedConfig` otherwise; the rest (corrupt packet, failed copy) is
  a `Backend` error.

Decoder (`ferricast_decoder::nvdec`, one generic `NvdecDecoder<C>`):

- H.264, HEVC, VP8, VP9, AV1, each the first candidate of its facade's
  automatic selection, ahead of VA-API, the platform decoders and the
  software decoders.
- Timestamps go through NVIDIA's parser with each packet; the slice-header
  machinery that timed pictures from the bitstream (`nvdec/h264.rs`,
  `nvdec/timestamps.rs`) is deleted. `h265/timestamps.rs` stays for
  `rust_h265`. `tests/timestamps.rs` checks NVDEC H.264 B-frames, VP8, and
  HEVC (CRA + RASL, joining mid-GOP) where the GPU decodes the clip (its
  fixtures are below NVDEC's HEVC / VP9 / AV1 minimum sizes, which hand
  over); `ten_bit.rs` checks presentation-order timestamps for every
  backend on 256x144 clips.
- Colour: NVDEC's (VUI, VP9 colour space, AV1 colour config) wins per
  field; an unspecified field takes the container's; a stream with no
  video signal type at all takes the container's range. Chroma location
  from `ColourTracker` (H.264 / HEVC), else the container.
- HDR: the SEI messages NVDEC hands over per picture (merged per half for
  the stream), else ferricast's own SEI parsing (H.264 / HEVC; it stops
  once NVDEC delivered any), else the container. AV1 metadata OBUs are
  always read by ferricast. Bitstream metadata replaces the container's as
  a whole, as before.
- `FrameMemory::Cpu`: pictures decode into pooled host buffers lent to the
  `RawFrame`'s `Bytes` without a copy (up to 48 held; past that they are
  copied so a consumer that keeps frames cannot exhaust NVDEC's pool of
  64). P010 for deep streams when `pixel_format` asks for it, NV12 (top
  eight bits) otherwise; BGRA / RGBA converted on the CPU as before.
- `FrameMemory::Native`: `CudaSurface`s; 10-bit streams stay P010 whatever
  `pixel_format` says (no GPU narrowing).

Encoder (`ferricast_encoder::nvenc`, `NvencEncoder<C>`):

- Same session as before: P3, low-latency tuning, no B-frames, CBR with
  max == average, IDR every keyframe interval plus forced / requested
  IDRs with parameter sets, live bitrate retarget, `get_headers` from
  `nvEncGetSequenceParams`, BGRA → ARGB / RGBA → ABGR.
- New inputs: NV12 and P010 (`pixel_format`); P010 into an 8-bit session
  is narrowed on the CPU; I420 is refused.
- 10-bit + HDR: `BitDepth::Ten` with P010 input encodes HEVC Main10 with
  the configured VUI colour and HDR10 (NVENC writes it on API 13.0+, the
  binding builds the SEI before); refused by the GPU's capabilities where
  absent. NVENC is out of `eight_bit_sdr_only` (VA-API keeps it). H.264 +
  Ten and 10-bit from 8-bit input are `UnsupportedConfig`.
- The configured colour is written for every input. RGB input is
  converted by NVENC with the matrix the VUI names: verified for BT.709
  and BT.601 (41 dB both ways); other matrices or full range with RGB
  input are refused.
- `CapturedFrame::Native(CudaSurface)` of the session's size and format
  is encoded in place (`encode_device`, the decoder's pooled `Arc` cached
  by NVENC, its `ready` event waited on the encoder stream); the frame is
  held until the packet is out. Other sizes / formats are read back.
- `CapturedFrame::Gpu` (Linux), packed RGB of the session's size (or one
  pixel larger per axis, cropped: the session rounds odd sizes down): the
  DMA-BUF is imported through EGL (`interop::DmaBufImporter`, keyed by
  `DmaBufId`) and encoded from the CUDA array it aliases (tiled) or its
  pitched memory (LINEAR, through the new `Encoder::encode_pitched`; a
  first version copied into a staging buffer, which cannot be written
  while NVENC's cache holds it). The frame and its `BufferLease` are held
  until NVENC is done. A change of the frames' description means the
  capture pool changed: NVENC's registrations are invalidated first, then
  the importer's; registrations unused for 600 frames, and the least
  recently used past 16, are dropped the same way (without a
  `cuCtxSynchronize`: NVENC has locked every picture). Any failure (no EGL
  device, another GPU, an unsupported modifier, a fence timeout, an import
  error) sends that frame through the CPU readback; the importer's creation
  failing is remembered for the session and a buffer's import failing
  until the pool changes. The first fallback is logged as a warning. The
  frame names its GPU (`GpuFrame::producer`; `FERRICAST_DRM_RENDER_NODE`
  for frames that do not) and whether it is complete (`GpuFrame::sync`):
  `Ready` frames are not waited on, `Implicit` ones poll the DMA-BUF,
  which the NVIDIA driver does not fence (see NVCODEC.md, "Capture
  zero-copy").
- `nvenc-zero-copy` is gone: the zero-copy path needs no build-time
  dependency (EGL is loaded at run time like the drivers), so it is part
  of `nvenc`. A root `nvenc-hevc` feature was added.
- The probe queries NVENC's capabilities once per process (one session)
  instead of opening a 1080p encoder.
- `configure` with a new shape reconfigures the session in place
  (`nvEncReconfigureEncoder`: a size up to the opened one, rate, GOP,
  bitrate, colour) and otherwise closes the old session before opening
  the new one (a GPU at its session limit takes it), reopening the old
  configuration if the new one fails. `VideoEncoder::reset` (new, default
  `configure`) always reopens; the stream pipeline's encoder recovery
  calls it, so a failed NVENC session is replaced rather than kept.
- A mid-stream `BackendBusy` / `BackendUnavailable` from `encode` (a lost
  GPU) hands the selection over to the next backend.
- Packets are lent to `EncodedFrame::data` from a pool (`Bytes::from_owner`)
  and come back to NVENC when dropped: no allocation per packet.

Measured (RTX 3060, driver 595.71.05, release build, 1080p60 testsrc2
clips, through the facades with `Fixed(Nvdec)` / `Fixed(Nvenc)`;
`tests/nvidia_throughput.rs`):

| Job | before (shiguredo) | now |
|---|---|---|
| H.264 decode, CPU frames | 595 fps | 638-642 fps |
| H.264 decode, native frames | 572 fps (CPU copies) | 801-810 fps (CUDA) |
| HEVC decode, CPU frames | 958 fps | 1053-1066 fps |
| HEVC decode, native frames | 960 fps (CPU copies) | 1594-1613 fps (CUDA) |
| VP9 decode, CPU / native | software only: 321 fps | 872 / 1204 fps |
| AV1 decode, CPU / native | software only: 158 fps | 1103 / 1634 fps |
| H.264 encode from BGRA | 270-297 fps | 361-369 fps |
| H.264 encode from NV12 | refused | 410-418 fps |
| HEVC encode from BGRA | 257 fps | 339-345 fps |

End to end (`tests/nvidia_e2e.rs`, 640x360, 30 frames): GBM DMA-BUFs
written by GLES (tiled) / CUDA (LINEAR) → NVENC → software decode,
worst PSNR against the written pattern 41.4 dB (H.264 tiled and LINEAR),
41.3 dB (HEVC), 41.7 dB (BT.601); NVDEC device frames → NVENC → decode,
64.3 dB against the first decode with `CudaSurface::readbacks()`
unchanged; HEVC Main10 HDR10 from P010 decodes (rust_h265 and NVDEC) to
P010, BT.2020 PQ and the configured metadata, 62.2 dB luma.

Known gaps:

- Windows: compiled and linted only (`x86_64-pc-windows-msvc` with every
  NVIDIA feature); never run. No D3D11 interop: Media Foundation textures
  reach NVENC through a readback.
- The zero-copy capture path takes packed RGB DMA-BUFs only (what
  PipeWire / wlr-screencopy produce here); NV12 DMA-BUFs are read back.
- Encoder selection does not yet prefer a backend on the capture's GPU
  (`GpuFrame::producer`, `nvenc::imports_from`): on a hybrid laptop NVENC
  is still picked over VA-API on the iGPU and reads the frames back.
- Rotated or flipped captures reach VA-API as `GpuFrame`s with an
  orientation (when NVENC is usable on the machine) and are read back and
  turned on the CPU (`GpuFrame::into_cpu`): the VPP blit is not told the
  orientation yet.
- NVDEC native output is not narrowed on the GPU: a consumer asking for
  NV12 from a 10-bit stream gets P010 surfaces.
- AV1 encode (Ada and newer) is not wired to a facade: there is no AV1
  encoder facade.
- One CUDA device (0) for everything; multi-GPU selection is not exposed.

## Selection cost

- Every attempt is timed (`BackendAttempt::elapsed_us`, `Opened::elapsed`)
  and logged. Probes (device / driver presence) run once per process per
  (backend, job); a failed open invalidates that backend's memo.
- `SelectionCache` remembers the backend that won per key (job +
  acceleration policy + size class + bit depth + frame memory). `Auto`
  tries the remembered winner first only when its plan already contains
  it; a later fallback continues behind it without retrying candidates
  that lost before. `Fixed` / `Ordered` never read or write the cache. A
  mid-stream failure demotes the winner.
- Default: an in-memory process-wide cache. `FERRICAST_CODEC_CACHE=off`
  disables it, a path selects the file-backed cache (versioned text,
  atomic temp + rename, a corrupt file is ignored with a warning). Code
  passes one with `with_cache` on registries, facades, `Selecting*` and
  `Catalog::with_selection_cache`.
- Measured (H.264 decode at 1080p, VA-API + NVDEC compiled in): 400 ms
  cold vs. 261 ms warm. Warm cannot go below opening the winner itself.

## Session events

- `SessionEvent`: `Opened`, `Switched { from, to, reason }`,
  `KeyframeRequested`, `BitrateChanged`, `Failed`, `Lagged { missed }`,
  `Closed`. Emitting never blocks the media path: a slow subscriber gets
  `Lagged` instead.
- Rust: `subscribe(capacity)` on facades and `Selecting*`,
  `CodecService::events(handle, capacity)`. istmo: `Codecs::events` as a
  `#[stream]`, which replays `Opened` for the current backend first and
  ends after `Closed`.
- `Closed` fires when the last reference to the session goes, which in
  `CodecService` includes the data plane's `Arc`s, not only the istmo
  handle.

## Apple codecs

- VP9 uses `VTRegisterSupplementalVideoDecoderIfAvailable`, which exists
  on macOS only; on iOS naming VideoToolbox for VP9 is
  `BackendCannotHandle`. VP9 4:2:2 / 4:4:4 hand over.
- HE-AAC: an `AudioSpecificConfig` in `AudioDecoderConfig::extra_data` is
  used as given. Plain ADTS at a core rate of 24 kHz or less is assumed
  to carry SBR, so a plain-LC stream at those rates comes out at twice
  its rate.
- ALAC / AAC-ELD decode take their cookie / ASC from `extra_data`, else
  the AirPlay defaults (`ferricast_codec::alac::AlacConfig` is shared by
  the symphonia and AudioToolbox decoders). ELD encode is 44.1 kHz,
  480-sample frames, no LD-SBR.
- The VideoToolbox symbols are strongly linked: macOS 11+ in practice.

## Known gaps

- `ferricast-gui` does not build from a clean target dir with the lock
  as it was: `bindgen 0.73` resolved to `syn 3` with `prettyplease 0.2`.
  Fixed by locking `prettyplease 0.3`.
- Workspace lints (`[workspace.lints.clippy]`: pedantic, `dbg_macro`,
  `todo`, `allow_attributes_without_reason`) are on in every member but
  `ferricast-miracast`, which was being written in parallel; it still has
  to opt in. `ferricast-codec` keeps its own, stricter table. Product names
  allowed in docs live in `clippy.toml` (`doc-valid-idents`).
- `nvenc-zero-copy` does not build: it calls `register_external_dmabuf` /
  `encode_external`, which no `shiguredo_nvcodec` has (the fork its comment
  names never existed). It is left out of every feature set that is
  checked. Closed: the feature is removed and zero-copy DMA-BUF input is
  part of `nvenc` (see "NVIDIA").
- `tests/airplay_loop.rs` ("only N of M frames shown") is timing-bound and
  fails under heavy machine load; it passes run alone.
