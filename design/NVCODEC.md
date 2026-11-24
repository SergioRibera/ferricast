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

## Decisions (foundation: bindings, loader, CUDA layer, version plumbing)

### Bindings (`crates/nvcodec/scripts/bindings.sh`)

- The script clones nv-codec-headers, checks that every tag still resolves
  to its pinned commit, extracts each tag with `git archive` and runs
  bindgen 0.72 (`nix shell nixpkgs#rust-bindgen` when bindgen is not on
  PATH; run it from the devshell so rustfmt matches `cargo fmt`):
  `nix develop -c crates/nvcodec/scripts/bindings.sh`.
- Enums are bindgen *newtypes* (`struct CUresult(pub u32)` with associated
  constants), never Rust enums: a driver returning a value the header does
  not list must not be undefined behaviour.
- The dynlink typedefs (`tcuInit`, `PNVENC*`) are function types; bindgen
  turns them into `Option<unsafe extern "C" fn ..>`. The loader stores the
  bare pointer for required symbols (`loader::Required<T>` projects it out
  of the `Option`) and `Option<fn>` for optional ones. `CUDAAPI` / `NVENCAPI`
  are `__stdcall` on Windows, the C convention on 64-bit targets; 32-bit
  Windows is a `compile_error!`.
- `--clang-macro-fallback` evaluates the function-like `*_VER` macros, so
  every NVENC struct version constant comes from its own header
  (`NV_ENC_CONFIG_VER` etc. are plain `u32` constants per module).
- NVENC GUIDs are `static const` objects in the header, not exported
  symbols; bindgen would emit them as `extern` statics that do not link.
  The script blocklists them and emits `pub const NAME: GUID` from the
  header text instead (and fails if it parsed fewer than the header has).
  `GUID` derives `PartialEq, Eq, Hash`.
- `NvEncodeAPICreateInstance` / `NvEncodeAPIGetMaxSupportedVersion` are
  plain declarations; the script declares function types for them
  (`tNvEncodeAPI*`, like FFmpeg's `dynlink_loader.h`).
- `scripts/cuda_extra.h` adds what `dynlink_cuda.h` lacks, from NVIDIA's
  `cuda.h` / `cudaEGL.h`: `cuGraphicsEGLRegisterImage`,
  `cuGraphicsResourceGetMappedEglFrame`, `cuGraphicsResourceSetMapFlags_v2`,
  `cuCtxSetCurrent`, `cuMemGetInfo_v2`, `cuDeviceTotalMem_v2`, synchronous
  `cuMemsetD8_v2` / `cuMemsetD2D{8,16,32}_v2`, `cuEventElapsedTime`.
  `CUresult` and `CUdevice_attribute` stay the header's subsets: newtypes
  accept any value, and `error::CudaStatus` names every code of `cuda.h`.
- Layout tests: kept for `cuda.rs` (no `long` in those structs, so they also
  hold on Windows and the Windows clippy run compiles them). Off for
  `cuvid.rs`: `tcu_ulong` is `unsigned long`, generated as `c_ulong`, which
  is 8 bytes on Linux and 4 on Windows; the Rust type follows the target
  like the C one does, but generated layout tests would pin Linux sizes.
  Off for NVENC: ~4 MB of tests for six versions, and these structs are
  `uint32_t` bitfields, pointers and fixed-width integers only.
- The video-source part of nvcuvid (`cuvidCreateVideoSource*`, file
  demuxing, `wchar_t` paths) is not generated.

### NVDEC layout across SDKs (checked by the script on every run)

One cuvid binding (n13.1.15.0) serves every driver. The script generates
layouts for all six tags and fails if any non-reserved field of an older
header sits at a different offset in the newest one. Findings for
11.1 → 13.1:

- No field moved. New fields only took reserved space
  (`CUVIDDECODECAPS::reserved3`, `CUVIDDECODECREATEINFO::Reserved2`,
  `CUVIDH264PICPARAMS::Reserved`, `CUVIDHEVCPICPARAMS::reserved1`,
  `CUVIDPICPARAMS::Reserved`, `CUVIDPROCPARAMS::Reserved2` all start later
  or shrink).
- Every top-level struct keeps its size (`CUVIDDECODECREATEINFO` 176,
  `CUVIDPICPARAMS` 4280, `CUVIDPROCPARAMS` 264, `CUVIDPARSERPARAMS` 136,
  `CUVIDEOFORMAT` 64, `CUVIDEOFORMATEX` 1088, `CUVIDPARSERDISPINFO` 24,
  `CUVIDSOURCEDATAPACKET` 32 on Linux x86_64).
- `CUVIDJPEGPICPARAMS` grew from a 4-byte stub to 2736 bytes (13.1); it
  lives in `CUVIDPICPARAMS`'s codec union next to `CodecReserved[1024]`
  (4096 bytes), so the outer struct does not change.
- `CUVIDAV1SEQHDR` gained the struct tag `_CUVIDAV1SEQHDR` in 13.1: a
  rename, same fields and offsets.
- `CUVIDSEIMESSAGEINFO` and the SEI callback (`pfnGetSEIMsg`) appear in
  12.0; on an 11.1-era driver the parser ignores the field (it sits in what
  was reserved space), so the SEI callback must be treated as optional.
- Enum values are identical across tags; only `NVDECAPI_VERSION` differs.
- Newer entry points (`cuvidDecodePictureAsync`,
  `cuvidRegisterDecodeSurfaces`, 13.1) are optional in the loader, as are
  `cuvidGetDecodeStatus`, `cuvidReconfigureDecoder`, `cuvidCtxLock/Unlock`.

### Loader

- `CudaLibrary`, `CuvidLibrary`, `NvencLibrary`: one table of function
  pointers per library, built by a macro from (field, typedef, symbol
  names); `_v2` names are tried first, then the legacy name
  (`cuDevicePrimaryCtxRelease_v2` → `cuDevicePrimaryCtxRelease`,
  `cuDeviceGetUuid_v2` → `cuDeviceGetUuid`).
- Each table lives in a `OnceLock<Result<_>>` static and is never dropped,
  so the library stays mapped and the copied function pointers stay valid.
  The outcome is cached including the error (hence `Error: Clone`): a
  machine without the driver pays one failed `dlopen`.
- `CudaLibrary::get` calls `cuInit(0)` once, inside the `OnceLock`.
- Windows loads with `LOAD_LIBRARY_SEARCH_SYSTEM32` so a DLL planted next
  to the executable or in the working directory is never picked up.
- Required CUDA symbols are the ones every R470+ driver exports; interop
  (external memory / semaphores, graphics map/unmap, EGL, D3D11, LUID,
  `cuArrayGetPlane`) is optional because WSL and platform builds lack parts
  of it.
- `NvencLibrary` uses the newest header's typedefs for its two exports:
  `GetMaxSupportedVersion` has one signature in every version, and
  `CreateInstance` takes a pointer whose pointee the driver interprets by
  its `version` field, so each version passes its own list through a cast.

### Errors

- `Error` names the failing call and the status by enumerator name:
  `CudaStatus` (table of `cuda.h`'s `CUresult` names, since
  `dynlink_cuda.h` declares five), `NvencStatus` (names matched against the
  generated `NVENCSTATUS` constants, identical in every version), plus the
  session's `nvEncGetLastErrorString` when there is a session.
  `Error::Cuvid` exists for NVDEC calls (they return `CUresult`).
- `DriverTooOld { found, minimum, driver }` and
  `ApiNotSupportedByDriver { requested, origin, driver_max, driver }` name
  the driver release to install for the OS the build targets.

### CUDA layer (`cuda.rs`)

- Only the device's primary context is used (`Context::primary`, retained
  in an `Arc`, released on drop), shared with every other CUDA user in the
  process. `Context` stores the handle as `NonNull`.
- `ContextGuard` (`!Send`) pushes the context around every driver call and
  pops it on drop, warning if the popped context is not the one pushed
  (unbalanced push/pop elsewhere on the thread). Objects therefore carry no
  thread affinity; `Send`/`Sync` impls each carry their argument.
- `Stream`s are created `CU_STREAM_NON_BLOCKING` (no implicit sync with the
  legacy default stream other libraries use); `Event`s without timing.
- `DeviceBuffer::pitched` asks for 16-byte elements (widest vector access;
  gives the ≥256-byte row alignment NVENC / NVDEC surfaces want). Copies
  take a `Region` (origin `Point` + bytes per row + rows), validated against
  both sides with checked arithmetic before the driver sees them.
- Asynchronous copies that touch host memory are `unsafe fn`: the driver
  stages pageable memory before returning but reads / writes page-locked
  memory later, and any CUDA user in the process may have pinned the slice
  passed in. A borrow-holding guard that synchronises on drop cannot make
  this safe (it can be `mem::forget`-ed). Device↔device async copies and
  fills are safe: `cuMemFree` is not stream-ordered and waits for work
  using the memory, so a buffer cannot be freed under a queued copy.

### NVENC version plumbing (`version.rs`, `encode.rs`)

- `NvencVersion` is any version a driver may report (`(major << 4) | minor`
  decoded); `ApiVersion` is the closed set this build supports, with
  `minimum_driver()` per OS. `ApiVersion::select(driver_max, requested)`
  is pure and unit-tested; `negotiate()` adds the driver query and
  `FERRICAST_NVENC_API` (empty = unset) and is not cached (one cheap call).
- `NvencApi` (sealed) is implemented by uninhabited marker types
  `version::api::V11_1 … V13_1`, one per generated module, by a macro: the
  `*_VER` constants and `NVENCAPI_VERSION` from that module, its function
  list type, and the calls whose argument structs differ between versions
  (`create_function_list`, `open_encode_session`, `destroy_encoder`,
  `last_error`). New per-version calls are added to the trait the same way,
  so every field is compile-checked against each version's bindings.
- Runtime dispatch is `ApiVersion::dispatch(visitor)` (an `ApiVisitor` with
  a generic `visit::<A>()`) or the `AnyApi` enum of `Api<A>` (a loaded
  function list); no `dyn`.
- `Api::<A>::load` checks the driver's maximum before
  `NvEncodeAPICreateInstance`, so a too-new version fails with
  `ApiNotSupportedByDriver` rather than a bare `NV_ENC_ERR_INVALID_VERSION`.
- `encode::Session<A>` is the bare RAII session (open on a CUDA context,
  destroy on drop with the context pushed, destroy a half-open session on
  failure as NVENC requires). `Send`, not `Sync`.
- Verified on an RTX 3060 with Linux driver 595.71.05 (CUDA 13.2, NVENC API
  13.0): negotiation picks 13.0; forcing 11.1, 12.0, 12.1, 12.2 and 13.0
  each fills the function list and opens + destroys a session; forcing
  13.1 fails with "NVENC API 13.1 was requested by FERRICAST_NVENC_API, but
  the driver supports 13.0 at most: install driver 610.0 or newer".

## DMA-BUF interop decisions

Verified on an RTX 3060, Linux driver 595.71.05, glvnd libEGL 1.7, mesa
libgbm 26 with the NVIDIA GBM backend, `/dev/dri/renderD128`.

### Chosen: EGL image → `cuGraphicsEGLRegisterImage` (no GL context)

- `eglQueryDevicesEXT` lists three devices for one GPU: the NVIDIA one
  (`EGL_NV_device_cuda`, `EGL_EXT_device_drm_render_node`), a mesa view of
  the same DRM node, and mesa's software device. The importer picks the
  device whose `EGL_CUDA_DEVICE_NV` is the CUDA device, opens it with
  `EGL_PLATFORM_DEVICE_EXT` and checks the producer's DRM node (`dev_t`
  from `linux-dmabuf` feedback / PipeWire) against the device's card and
  render nodes, or by PCI address from sysfs. Another GPU →
  `Error::DeviceMismatch` (the caller falls back to the CPU path).
- `eglCreateImageKHR(EGL_LINUX_DMA_BUF_EXT)` with
  `EGL_DMA_BUF_PLANE0_MODIFIER_{LO,HI}_EXT`, then
  `cuGraphicsEGLRegisterImage(READ_ONLY)` works without any EGL/GL context
  current. The registration is an alias of the DMA-BUF: content written
  after registration is read through the same registration (tested), so
  there is no copy at all. Tiled (block-linear, `0x03000000_00606010`…)
  buffers come back as a CUDA array (`CU_EGL_FRAME_TYPE_ARRAY`, 4 × u8),
  LINEAR as pitched device memory (`CU_EGL_FRAME_TYPE_PITCH`).
- EGL resources are always mapped: `cuGraphicsMapResources` on them fails
  with `CUDA_ERROR_UNKNOWN`. `map` is therefore a fence wait plus
  `cuGraphicsResourceGetMappedEglFrame` (a lookup, ~0.2 µs).
- CUDA refuses `XRGB8888` / `XBGR8888` images (`CUDA_ERROR_INVALID_VALUE`)
  but takes `ARGB8888` / `ABGR8888`; the importer tells EGL the alpha
  variant of the same layout. All four formats read back bit-exact, LINEAR
  and tiled.
- `eglQueryDmaBufModifiersEXT` offers 12 block-linear modifiers + LINEAR
  for all four formats; LINEAR is `external_only` (GL can only sample it as
  `GL_TEXTURE_EXTERNAL_OES`), which does not concern CUDA's EGL interop.
- Displays are initialised once and never terminated (process-wide
  singletons; `EGL_KHR_display_reference` is not offered).
- Each registration owns a dup of the DMA-BUF fd, the EGL image and the
  CUDA resource; it waits for the context before unregistering (pool
  changes only), so work queued on the array cannot outlive it. The
  producer may free its buffer while it is registered.
- Costs at 1920×1080: import (EGL image + register) 0.1–0.2 ms, once per
  pool buffer; `map` ≈ 0.5 µs; map + array→pitched device copy + stream
  sync ≈ 57–60 µs.

### Synchronisation

- Producer → CUDA: CUDA ignores implicit fences and cannot import a
  `sync_file` (no SYNC_FD semaphore type), so `map` waits on the CPU with
  `poll(POLLIN)` on the DMA-BUF (waits for its write fences; same as
  `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` + poll without the extra fd), bounded by
  a timeout. Explicit-sync callers pass `FenceWait::AlreadySignalled`.
- CUDA → producer: CUDA attaches no fence to the DMA-BUF either. The
  consumer must keep the capture buffer's lease until the GPU work reading
  it (NVENC encode, the device copy's stream) has completed.

### Rejected

- **Vulkan import + copy into OPAQUE_FD memory + `cuImportExternalMemory`**:
  costs a GPU copy per frame plus a Vulkan device, command buffers and
  semaphores, to obtain what the EGL path gives for free. Its one advantage
  (GPU-side wait on the producer's sync file through a Vulkan semaphore)
  would still need a Vulkan→CUDA semaphore. Not prototyped; the EGL path
  was measured zero-copy, so there was nothing to win.
- **CUDA importing the DMA-BUF directly**: `cuImportExternalMemory` with
  `OPAQUE_FD` on a DMA-BUF fd returns `CUDA_ERROR_UNKNOWN`; CUDA has no
  DMA-BUF import handle type (DMA-BUF is export-only, through
  `cuMemGetHandleForAddressRange`).
- **EGL image → GL texture → `cuGraphicsGLRegisterImage`**: works for tiled
  buffers (bit-exact), fails for LINEAR (`GL_TEXTURE_EXTERNAL_OES` →
  `CUDA_ERROR_INVALID_VALUE`, and GL cannot render to it, so LINEAR would
  need a shader copy), and needs a GL context per thread. Strictly worse.

### Observed hazard (not on this path)

- CPU `mmap` of a 1080p LINEAR NVIDIA GBM buffer does not show the GPU's
  view past the first 64 KiB–4 MiB (zeros or unrelated pages), even with
  `DMA_BUF_IOCTL_SYNC`; GPU views (CUDA, GL) agree with each other. CPU
  readback of LINEAR buffers on NVIDIA should be treated as suspect.
## NVDEC decisions

- `decode::Decoder<M>` with `M = Host | Device` (sealed): the frame type is
  known statically (`HostFrame` / `DeviceFrame`), no runtime enum for
  callers. One `Packet` = one access unit (Annex B, parameter sets in band)
  / VP8-VP9 frame (superframes whole) / AV1 temporal unit (Section 5 OBUs,
  not Annex B). `whole_pictures` (default on) sets `CUVID_PKT_ENDOFPICTURE`
  so a picture decodes without waiting for the next packet; off for
  arbitrary chunks.
- State reachable by the callbacks lives behind a raw `NonNull` (not a
  `Box`, whose moves assert uniqueness against the parser's copy of the
  pointer). `decode` holds no reference into it while
  `cuvidParseVideoData` runs, so callbacks own it without any lock. Every
  callback body runs under `catch_unwind`; errors are stored and returned by
  the `decode` call. Field order of `Decoder` is drop order: state (NVDEC
  decoder, stream, queued frames) → parser → ctx lock → context.
- Decoder creation: `ulNumDecodeSurfaces = min_num_decode_surfaces +
  max_display_delay` (20 when the driver reports 0), returned from the
  sequence callback to size the parser's DPB. `ulTargetWidth/Height` = coded
  size with a whole-picture `display_area`: VPP is an identity copy and the
  crop happens in our 2D copy, which keeps output bit-exact. A new sequence
  reuses the decoder when codec / chroma / depth / format match, the size is
  within `ulMaxWidth/Height` and the surfaces suffice (resized with
  `cuvidReconfigureDecoder`); otherwise it is destroyed first, then
  recreated. `max_coded_size` reserves headroom.
- Output format: NV12 ≤ 8 bits, P016 above (10-bit P016 = P010); NV16/P216
  and YUV444/YUV444_16Bit where the caps mask allows (HEVC 4:4:4 verified on
  Ampere). Everything is checked against `cuvidGetDecoderCaps` with an
  `Error::Unsupported` naming the exceeded limit.
- Every picture is mapped with `output_stream` = the decoder's stream,
  copied out plane by plane (`cuMemcpy2DAsync`), unmapped right away. Host
  mode syncs that stream (pageable destination); device mode records the
  slot's `ready` event and unmaps without waiting, which is safe because
  the next map of an output surface post-processes on the same stream.
  Device consumers that read asynchronously call
  `DeviceFrame::release_after(stream)`; the decoder's stream waits for that
  event before reusing the buffer.
- Pools (host `Vec`s, device `DeviceBuffer` + 2 events, SEI buffers) are
  bounded by `max_frames` (default 32, `Error::FramesInFlight` beyond),
  start a new generation on a geometry change, and free stale items when
  they come back. Steady state allocates nothing per picture.
- Driver findings (595.71.05, RTX 3060):
  - `CUVIDEOFORMAT::display_area` always has `left = top = 0` (H.264
    `frame_crop_left/top_offset`, HEVC `conf_win_left/top_offset` only
    shrink right/bottom). The decoder reads the cropping window of each SPS
    it is fed (NAL units before the first slice only; `decode/crop.rs`) and
    applies left/top only when the SPS's right/bottom agree with the
    parser's.
  - SEI payloads 137 / 144 arrive decoded into NVIDIA's structs
    (little-endian, `display_primaries_x[3]` then `_y[3]`), not as SEI
    bytes; other types arrive raw. SEI comes per decode surface and is moved
    into the frame at display, so it never leaks to a later picture.
  - `CUVID_PKT_DISCONTINUITY` on a packet with data drops that data (an
    H.264 IDR and its parameter sets, losing the GOP): the flag is sent on an
    empty packet of its own, which outputs what the parser held, then the
    data. `CUVID_PKT_ENDOFSTREAM` (flush) leaves the parser reusable.
  - A truncated AV1 temporal unit (last OBU size past the packet) makes
    `cuvidDecodePicture` fault (SIGBUS) inside the driver. Packets are
    checked for OBU framing (headers and sizes only) and refused with
    `Error::InvalidPacket`. Bit flips / random data inside well-framed OBUs
    did not crash in the tests, but the driver is the trust boundary there.
- `cuvidGetDecodeStatus` (optional) gives `DecodeStatus` per frame;
  `Unknown` without the symbol.
- Tests: `tests/decode.rs` on clips from `scripts/decode_fixtures.sh`, each
  picture compared by FNV-1a 64 with FFmpeg's `rawvideo` decode
  (`-flags unaligned` so FFmpeg honours left crops).
## NVENC decisions

### Per-version structure (`encode/raw.rs`)

- `EncodeApi` (crate-private, on top of `NvencApi`) holds every call whose
  argument structs differ between API versions: caps / GUID / input-format
  queries, initialize, reconfigure, sequence params, bitstream buffers,
  `nvEncSetIOCudaStreams`, register / unregister, map / unmap, encode, EOS,
  lock. One macro body (`encode_api!`) implements it for every marker
  `V11_1 … V13_1`, so each instantiation is compiled against that
  version's bindings. Differences are `yes` / `no` tokens per version,
  expanded by `pick!`:
  - `av1` (12.0): AV1 GUIDs, `av1Config`, `av1PicParams`.
  - `typed_vui` (12.0): VUI colour fields are enums; 11.1 has `u32`.
  - `bit_depth_enum` (12.2): `inputBitDepth` / `outputBitDepth` replace
    HEVC `pixelBitDepthMinus8` and AV1 `(input)PixelBitDepthMinus8`.
  - `native_hdr` (13.0): HEVC / AV1 `outputMasteringDisplay` /
    `outputMaxCll` and `pMasteringDisplay` / `pMaxCll`.
  - `chroma_offset_in` (13.0): `NV_ENC_REGISTER_RESOURCE::chromaOffsetIn`.
  - `uhq` (13.0): `NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY`.
  Other layout changes (`NV_ENC_RC_PARAMS`, `NV_ENC_INITIALIZE_PARAMS`,
  `NV_ENC_PIC_PARAMS`, `NV_ENC_LOCK_BITSTREAM`, ...) need no code: the
  fields used exist in every version, the per-module types and `*_VER`
  constants carry the layout.
- `ApiFeature` (public) names those features with `added_in()`;
  `Capabilities::features` lists what the negotiated version has, and a
  test checks the macro tokens agree with `added_in()`. Use on an older
  version is `Error::Unsupported` naming the version and driver.
- `Encoder` is an enum over `Engine<A>` per version (no `dyn`);
  `Engine<A>` (`encode/engine.rs`) is version-independent.
- 11.1's generated `GUID` derives no `PartialEq`: GUIDs compare by field.

### HDR10

- 13.0+: HEVC (and AV1) metadata written by NVENC (config bits plus
  pointers on every picture). H.264 on every version and HEVC before 13.0:
  this crate builds SEI 137 / 144 payloads (`encode/sei.rs`) and passes them
  through `seiPayloadArray` on pictures it expects to be IDRs. AV1 HDR10
  before 13.0 is refused (no tested OBU path).
- IDR prediction: first picture, forced IDR, and every `idr_period` since
  the last IDR. Verified on driver 595: a forced IDR restarts NVENC's IDR
  period (period 10, forced 4, requested 13 → IDRs 0, 4, 13, 23).

### Inputs and pipeline

- Pipeline depth = 1 + `frames_in_flight` + B frames + lookahead: one
  bitstream buffer and one (lazily created, registered once) host slot per
  position. Inputs are mapped per picture and unmapped after the lock.
- Host input: `cuMemcpy2DAsync` on the encoder stream, then an event wait
  before the call returns (the caller's slice is free afterwards, even if
  pinned by someone).
- Device input: cached by `Arc<DeviceBuffer>` address with a `Weak`; the
  `Weak` keeps the `Arc` allocation, so the address cannot be reused while
  the entry exists, and `strong_count() == 0` retires it (swept on each
  device encode). A pending picture holds a strong `Arc`, so a buffer is
  never freed while NVENC reads it. Deviation from NVIDIA's "unregister
  before free": a dropped buffer's registration is unregistered at the next
  sweep, after `cuMemFree`; it is never mapped in between. A buffer still
  pending is refused rather than mapped twice. The decode module should
  hand out frames as `Arc<DeviceBuffer>` (or something lending one).
- A chroma plane not right after the luma rows (`DeviceFrame::chroma_row`)
  uses `chromaOffsetIn` (13.0); verified bit-identical to a contiguous
  frame on 13.0, refused on 11.1–12.2.
- CUDA arrays (packed RGB only): cached by a caller key, compared with the
  array handle and size; `encode_array` is `unsafe` because a destroyed
  array whose handle is reused under the same key without
  `invalidate_array` would hit a stale registration.
- Evicted registrations are retired and unregistered only once every
  picture submitted before the eviction has been locked.
- Readiness: `NV_ENC_ERR_NEED_MORE_INPUT` holds pictures (B frames). With
  lookahead NVENC answers success while the last `depth` pictures are not
  lockable yet (locking them fails with `INVALID_PARAM`), so after a
  success all but `lookahead` pending pictures are ready; EOS releases all.

### Errors

- Session limit: driver 595 on an RTX 3060 answers
  `NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY` from `nvEncOpenEncodeSessionEx` after
  12 sessions; older drivers `NV_ENC_ERR_OUT_OF_MEMORY`. Both map to
  `Error::NvencSessionLimit`.
- Every NVENC failure carries `nvEncGetLastErrorString` (often empty).

### Verified (RTX 3060, driver 595.71.05, API 11.1–13.0 forced)

- H.264 / HEVC from NV12, ARGB, ABGR and HEVC Main10 from P010 with
  HDR10: identical bitstreams across versions, ffprobe codec / profile /
  pix_fmt / colour / keyframes / HDR side data as configured, PSNR 58.7
  (NV12), 43.5 (RGB, BT.709 conversion), 61.9 dB (P010).
- Host, device and CUDA-array input give identical bitstreams.
- Ultra-high-quality tuning on 13.0 is refused by this GPU
  (`nvEncInitializeEncoder` `INVALID_PARAM`); AV1 is `MissingFromGpu`.
- 1080p NV12 H.264 P1 ultra-low-latency (debug build): 426 fps host, 473
  device; with 3 frames in flight 477 / 484 fps.
- Open: with many NVENC sessions opening and closing at once on the GPU
  (the parallel test run, one test holding the 12-session limit), an
  `nvEncOpenEncodeSessionEx` once failed with `NV_ENC_ERR_INVALID_VERSION`
  instead of the limit status; not reproduced with two processes looping
  over different API versions. The encode tests are serialised and the
  limit test runs in its own process.
