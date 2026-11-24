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
| VP8 / VP9 / AV1 decode | hw + sw (VP9/AV1/HEVC need Store extensions) | — | hw + `c2.android.*` |
| AAC-LC encode / decode | sw | sw (decode: AAC-LC core, no SBR/PS) | sw |
| Opus encode / decode | — | — | sw (encode needs API 29) |
| AAC-ELD, ALAC | — | — (ELD unverified, ALAC needs the AirPlay cookie) | — |

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
    COM. The plumbing is `ferricast_codec::mf` (the only `unsafe` module
    of that crate), shared by encoder and decoder.
  - MediaCodec: the hardware candidate refuses a device whose
    type-default codec is a Google software one, so `Auto` does not try
    the same software codec twice.
- Frames are CPU-resident on these paths (`CapturedFrame::Cpu`, NV12 out
  of the decoders). Zero-copy surfaces (CVPixelBuffer, AHardwareBuffer,
  D3D11 textures) are future work; `CapturedFrame::Gpu` is DMA-BUF and
  exists on Linux / Android only.
- Verification: `nix develop -c cargo clippy --target <triple> -- -D
  warnings` for x86_64/aarch64-pc-windows-msvc,
  x86_64/aarch64-apple-darwin, aarch64-apple-ios(-sim) and
  x86_64/aarch64-linux-android compiles and lints them from Linux (the
  flake ships those std libraries; checking needs no SDK). Nothing in
  this list has been linked or run on a device: first runs on each OS
  are expected to surface media-type / format negotiation issues.

## Known gaps

- `ferricast-gui` does not build from a clean target dir with the lock
  as it was: `bindgen 0.73` resolved to `syn 3` with `prettyplease 0.2`.
  Fixed by locking `prettyplease 0.3`.
- Workspace lints are not enabled crate-wide (the existing crates would
  emit thousands of pedantic warnings); `ferricast-codec` carries its own
  `[lints]` table.
