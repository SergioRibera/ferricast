# ferricast-nvcodec

ferricast's own binding to NVIDIA's video engines (NVDEC, NVENC) and the
slice of the CUDA driver API they need. It replaces `shiguredo_nvcodec`
(and the patched copy in `vendor/`). Decision log: append here.

## Goals, in priority order

1. Safety: no unsoundness reachable from safe code, no panic or unwind
   across FFI, no use-after-free between CUDA / NVDEC / NVENC objects.
2. Stability: works on every driver the supported SDK range covers,
   fails with a typed, actionable error otherwise (never a crash, never a
   silent wrong picture).
3. Zero-copy: pictures can stay on the GPU from capture (DMA-BUF) or
   decode (NVDEC) to encode (NVENC) and to the renderer.
4. Performance: no per-frame allocation, registration or context
   creation on the hot path.

## Layout

```
crates/nvcodec/
  Cargo.toml            no build script, no bindgen at build time
  scripts/bindings.sh   regenerates src/sys/** from pinned nv-codec-headers tags
  src/lib.rs
  src/error.rs          Error (thiserror): CudaError(CUresult name), NvencError(NVENCSTATUS name), DriverTooOld, LibraryMissing, Unsupported...
  src/loader.rs         runtime loading (libloading) of libcuda / libnvcuvid / libnvidia-encode (Linux .so.1, Windows nvcuda.dll / nvcuvid.dll / nvEncodeAPI64.dll)
  src/sys/cuda.rs       generated: CUDA driver API subset (from dynlink_cuda.h)
  src/sys/cuvid.rs      generated: one NVDEC header set
  src/sys/nvenc/v11_1.rs ... v13_1.rs   generated: one module per NVENC API version
  src/cuda.rs           Device, Context (primary context), ContextGuard (push/pop), Stream, Event, DeviceBuffer (pitched), copies
  src/version.rs        NVENC API version negotiation
  src/decode/**         NVDEC
  src/encode/**         NVENC (single-source session logic over every API version)
  src/interop/**        DMA-BUF / EGL → CUDA (zero-copy capture input), Linux only
```

Headers come from FFmpeg's `nv-codec-headers` (MIT, NVIDIA copyright):
tags n11.1.5.4, n12.0.16.2, n12.1.14.1, n12.2.72.0, n13.0.19.1,
n13.1.15.0. The generated bindings are committed; the script pins each
tag's commit hash. No clang / bindgen is needed to build, so the crate
cross-compiles like the rest of the workspace. The code is written from
the headers and NVIDIA's documentation, not from `shiguredo_nvcodec`.

## NVENC API versions (runtime matching)

| API  | Linux driver | Windows driver |
|------|--------------|----------------|
| 11.1 | 470.57.02    | 471.41         |
| 12.0 | 520.56.06    | 522.25         |
| 12.1 | 530.41.03    | 531.61         |
| 12.2 | 550.54.14    | 551.76         |
| 13.0 | 570.0        | 570.0          |
| 13.1 | 610.0        | 610.0          |

- A driver accepts any API version up to the one it reports through
  `NvEncodeAPIGetMaxSupportedVersion`; struct layouts and struct version
  numbers differ between API versions. So the crate carries bindings for
  every version above and, per process, picks the highest one the driver
  accepts. `FERRICAST_NVENC_API=<major>.<minor>` forces one (an error if
  the driver does not accept it), which is also how tests run every
  version on one machine.
- A driver older than 11.1 → `Error::DriverTooOld { found, minimum }`
  naming the driver release to install.
- Session logic is written once and instantiated per version (a trait
  implemented per generated module, or a macro over the modules); every
  field that differs between versions is handled explicitly and
  compile-checked against that version's bindings. Dispatch over versions
  is an enum, not `dyn`.
- Features a version lacks (e.g. AV1 encode before 12.0, mastering display
  / content light SEI fields) are reported in the capability query and
  refused with `Error::Unsupported` at open, never silently dropped.

## NVDEC

- One header set (the newest). The cuvid structs keep their layout across
  these SDKs (new fields take reserved space); verify this when
  regenerating, and gate any symbol a driver may lack (looked up as
  optional).
- Timestamps travel through the parser (`CUVID_PKT_TIMESTAMP` →
  `CUVIDPARSERDISPINFO::timestamp`): no bitstream re-parsing to time
  pictures.
- Output honours the display area (all four edges, not only the size),
  the stream's bit depth (NV12 for 8-bit, P016 above), the colour in
  `CUVIDEOFORMAT::video_signal_description`, and SEI messages (HDR
  metadata) through the parser's SEI callback when the driver has it.
- Two output modes: host (copied into reused host buffers) and device
  (a pooled pitched CUDA buffer handed out as a frame that returns itself
  to the pool on drop), so NVDEC → NVENC and NVDEC → renderer never touch
  host memory.
- Resolution / bit depth changes reconfigure (`cuvidReconfigureDecoder`
  when the change allows it) or recreate the decoder.

## NVENC

- Inputs: host bytes (copied on a CUDA stream into a pooled, registered
  device buffer), a device frame (NVDEC output, `DeviceBuffer`), a CUDA
  array (EGL-imported DMA-BUF). Every input resource is registered once
  and cached; nothing is allocated or registered per frame in steady
  state.
- Formats: NV12, P010 (HEVC Main10 / AV1 10-bit), ARGB / ABGR (NVENC's
  own colour conversion). Codecs: H.264, HEVC, AV1 (where the GPU and API
  version have it; the capability query says).
- Rate control, presets, tuning, GOP, profile, live bitrate reconfigure,
  forced IDR, parameter sets out of band, VUI colour and HDR SEI.
- `nvEncSetIOCudaStreams` puts input copies and encode on one stream.
  Linux has no async encode mode: the lock on the output bitstream
  blocks; an optional output delay (frames in flight) trades latency for
  throughput, default 0 (lowest latency).

## Zero-copy capture input

DMA-BUF (PipeWire / wlr-screencopy, tiled with a DRM modifier) → EGL
image (`EGL_LINUX_DMA_BUF_EXT`, modifier attributes) on an EGL display of
the same GPU → `cuGraphicsEGLRegisterImage` → CUDA array → NVENC
(`NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY`, or a device copy into a pooled
buffer when direct registration is not possible). Registrations are
cached per DMA-BUF identity (`DmaBufId`) and dropped when the pool
changes. The producer's fence is waited on before NVENC reads. Only when
the DMA-BUF lives on the CUDA device's GPU; otherwise the caller falls
back to the CPU path.

## Safety rules

- `unsafe` lives in `sys` users only: `cuda.rs`, `decode`, `encode`,
  `interop`, `loader.rs`. Every block has a `// SAFETY:` comment
  (`clippy::undocumented_unsafe_blocks` denied, `unsafe_op_in_unsafe_fn`
  denied).
- Every object owns what it needs to be destroyed (an `Arc` of the loaded
  library and of the context), so drop order cannot free a context under
  a live decoder, encoder or buffer.
- NVDEC callbacks never unwind (`catch_unwind`) and never lock a mutex
  the caller of `cuvidParseVideoData` holds.
- `Send` / `Sync` are implemented only with a written argument; the CUDA
  context is made current around every call.
