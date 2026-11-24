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
  bitstreams or backends that give none (openh264, NVDEC binding,
  rust_h265).

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
- Linux hardware decoders (NVDEC binding, VA-API pool) are NV12-only:
  10-bit streams hand over with `UnsupportedStream`, so `Auto` falls back
  to rav1d / rust_h265.

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
- `nvenc-zero-copy` does not build against the `shiguredo_nvcodec` in the
  lock (no `register_external_dmabuf` / `encode_external`); it is left out
  of every feature set that is checked.
- `tests/airplay_loop.rs` ("only N of M frames shown") is timing-bound and
  fails under heavy machine load; it passes run alone.
