# ferricast — code vs UI mapping

Grepped `Cargo.toml` + every crate's features + the `advertiser`, `encoder`, `decoder` facades. Here is what the Settings UI can actually offer today without lying.

## Protocols (what the sender/receiver can speak)

| Protocol | Source | Compile flag | Runtime toggle | Sub-options | UI status |
| --- | --- | --- | --- | --- | --- |
| **AirPlay 2** | `crates/airplay/` | `airplay` (root) + `fairplay` (airplay crate) | enable discovery + advertise | FairPlay DRM (fp-setup via precomputed key message) · pairing storage (OS keyring via `oo7`) · interface override (`FERRICAST_AIRPLAY_INTERFACE`) | Full |
| **Chromecast** | `crates/chromecast/` | `chromecast` (root) + `opus` (chromecast crate) | enable discovery + advertise + HLS receiver | Cast Streaming (RTP mirror with Opus) vs HLS fallback · HTTP port for HLS · CA/TLS cert for castv2 | Full |
| **ferricast native** | `crates/airplay` is **not** this — it's the openscreen wire that was rebuilt 2026-11-20; crate path is in `chromecast?` section. | always on when chromecast is on | enable advertise | low-latency RTP · codec negotiation on first frame | Full |
| **Miracast / MS-MICE** | `crates/miracast/` (code present, root feature commented out as "Reserved for upcoming protocol crates") — commit `3767963` is in-flight | planned | planned | WiFi-Direct hardware probe · MS-MICE transport (TCP fallback) | Planned |
| **Progressive HLS / MP4 / WebM / MKV** | `crates/hls/`, `crates/demuxer/` | `demuxer/matroska`, `demuxer/audio-files`, `demuxer/http` | always on | prefetch window · temp-file vs range request | Advanced |

Notes:
- "Advertise" is the receiver-side action. "Discover" is the sender-side action. Both toggles land in the Protocols tab.
- FairPlay is GPL-3.0-or-later and sits in a separate crate (`ferricast-fairplay`) — the toggle exists so GPL-shy builds can drop it. Without it AirPlay senders talking to receivers that demand DRM drop to the all-zero base key, which some media services refuse.

## Codecs (encoders + decoders + runtime priority)

### Video encode (sender)

| Codec | Backend | Compile flag | Runtime priority | Notes |
| --- | --- | --- | --- | --- |
| H.264 | NVENC | `nvenc` | 1 (if GPU) | loads `libcuda` + `libnvidia-encode` at runtime; falls through if missing |
| H.264 | VA-API | `vaapi` | 2 (if iGPU/AMD) | pinned to `cros-libva` fork |
| H.264 | x264 (software) | `openh264` (name kept for compat) | 3 (always) | guaranteed fallback |
| HEVC | NVENC | `nvenc` + HEVC marker | 1 (if GPU) | encoder-level feature gate |
| AAC | libfdk-aac | `fdk-aac` | 1 | AAC-LC for HLS audio path |
| AAC | `oxideav-aac` (pure-Rust) | compiled-in fallback | 2 | kicks in when `fdk-aac` is off |
| Opus | opus-rs | `opus-rs` | always | RTP mirror audio |

### Video decode (receiver)

| Codec | Backend | Compile flag | Runtime priority | Env override | Notes |
| --- | --- | --- | --- | --- | --- |
| H.264 | NVDEC | `nvdec` | 1 (if GPU) | — | CUDA loaded at runtime |
| H.264 | VA-API | `vaapi` | 2 (if iGPU/AMD) | `FERRICAST_H264_DECODE_BACKEND=vaapi` to force | opt-in, slice path only "fully verified" when forced |
| H.264 | openh264 (software) | `openh264` | 3 (always) | — | default |
| HEVC | NVDEC | `nvdec` + `nvdec-hevc-decode` | 1 | — | |
| HEVC | VA-API | `vaapi-hevc-decode` | 2 | — | shares cros-libva |
| HEVC | `rust_h265` (software) | `hevc-decode` | 3 | — | |
| VP8 | `vp8-decoder` (pure-Rust) | `vp8-decode` | sole | — | progressive media only |
| VP9 | `rusty_vp9` | `vp9-decode` | sole | — | progressive media only |
| AV1 | `rav1d` | `av1-decode` | sole | — | progressive media only |
| AAC-LC | symphonia | `aac` | 2 | `FERRICAST_AAC_DECODER=symphonia` to force | fallback |
| AAC / HE-AAC v1/v2 | libfdk-aac | `fdk-aac` | 1 | — | adds SBR + PS |
| ALAC | symphonia | `alac-decode` (pulled by airplay) | sole | — | AirPlay audio codec |
| Opus | opus-rs | `opus-decode` | sole | — | progressive / Cast Streaming receive |

### Progressive demuxers

MP3, FLAC, Ogg, Vorbis, Opus (container), WAV via symphonia; MKV/MP4 via `matroska-demuxer`.

## Capture backends (sender, Linux)

All capture goes through the istmo screen-capture plugin (`ferricast-capture`, no features); it picks the backend at run time.

| Backend | Feature | Notes |
| --- | --- | --- |
| wlr-screencopy / ext-image-copy (DMA-BUF) | — | GPU end-to-end, no dialog; needs `libgbm` |
| X11 (XShm) | — | shared memory |
| PipeWire (xdg-desktop-portal) | — | dialog picks the source; used when nothing else can capture it |
| Toplevel lists (ext / wlr foreign-toplevel) | — | in-process picker on wlr / Niri / Plasma 6 |
| Capture backend override | — | env `FERRICAST_CAPTURE=wlr-screencopy\|ext-image-copy\|x11\|portal` |
| DRM render node | — | env `FERRICAST_DRM_RENDER_NODE=/dev/dri/renderD128` |

## Device name (new in UI)

Today: callers pass a `friendly_name: String` into `AdvertiseInfo` (see `crates/core/src/advertiser.rs:32`). Every receiver crate (`airplay/receiver.rs:369`, `chromecast/receiver/mod.rs:60`, `miracast/receiver.rs:38`) just forwards that string.

Proposal:

1. **Template default:** `"Ferricast - {device}"` where `{device}` comes from the **`istmo`** workspace's `device_info` plugin (`istmo/crates/plugins/src/device_info/mod.rs`) — it already reads `model_name` + `manufacturer` from `/sys` and falls back to hostname.
2. **Resolved value shown in UI** as placeholder so the user sees what peers will see even before overriding.
3. **User override:** plain text input in Settings · Receiver. Persists in `XDG_CONFIG_HOME/ferricast/config.toml` (or `AppData` / `~/Library/Application Support`).
4. **On save:** every live advertiser gets a fresh `AdvertiseInfo` with the new `friendly_name` — mDNS re-publishes, the receiver idle hero updates, device picker rows on peers refresh within ~10 s.
5. **Validation:**
   - 1–63 bytes (mDNS label limit per RFC 6763).
   - Trim surrounding whitespace.
   - Reject non-printable / control characters.
   - Show "This is what peers will see" preview below the field.

The dependency on `istmo` is a new workspace entry (path dep since both repos live under `/home/s4rch/Public/rust`, or a crates.io cut if istmo publishes). If the user wants ferricast to stay standalone, mirror istmo's `/sys` reads in `crates/core/src/host.rs` behind a small trait and skip the dep.

## Env vars the UI should replace

Everything below should become a Setting so power users don't need to keep env vars around:

- `FERRICAST_CAPTURE=wlr-screencopy|ext-image-copy|x11|portal` → Settings · Capture → Preferred backend
- `FERRICAST_AIRPLAY_INTERFACE` → Settings · Protocols · AirPlay → Interface override
- `FERRICAST_CAST_INTERFACE` (referenced in airplay advertiser doc) → same pattern
- `FERRICAST_AAC_DECODER=symphonia` → Settings · Codecs → AAC decode = symphonia
- `FERRICAST_H264_DECODE_BACKEND=vaapi` → Settings · Codecs → H.264 decode = VA-API
- `FERRICAST_DRM_RENDER_NODE` → Settings · Advanced
- `FERRICAST_VK_VALIDATION` → Settings · Advanced (dev-only)

Env vars stay supported, but the UI becomes the canonical source of truth and overrides them when the user has clicked a value.
