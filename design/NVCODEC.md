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
changes. NVENC reads only complete frames: the capture waits for the
producer (explicit acquire point, or a protocol that reports finished
frames) and says so (`FrameSync::Ready`); the implicit-fence poll is a
best-effort fallback that the NVIDIA driver does not honour (see
"Synchronisation" and "Capture zero-copy"). Only when the DMA-BUF lives
on the CUDA device's GPU; otherwise the caller falls back to the CPU
path.

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
  CUDA resource; it waits before unregistering, so work queued on the
  array cannot outlive it. `forget` / `clear` wait for the whole context
  (`cuCtxSynchronize`, which stalls every session on the process-wide
  primary context); `forget_all` / `clear_with` take a `Quiesce`: an event
  recorded behind the last read, or (unsafe) nothing when the caller knows
  the work completed, which NVENC's zero-copy path does (its encode calls
  return with the packet locked). The
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
- **Measured: the poll does not wait for NVIDIA's own rendering.** With
  3000 GLES clears of a 3840×2160 tiled buffer queued and flushed, the
  poll returned after 14 µs with the work still pending; the work's
  `EGL_ANDROID_native_fence_sync` fence, imported into a DRM timeline
  syncobj, kept the wait for 310 ms (`tests/interop.rs`,
  `implicit_wait_ignores_pending_nvidia_work_explicit_wait_does_not`).
  The driver attaches no fence to the DMA-BUF's reservation object, so
  `FenceWait::Implicit` is only meaningful for Mesa producers. On NVIDIA
  correctness comes from the capture (see "Capture zero-copy").
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
  arbitrary chunks; `Packet::whole_picture` overrides it per packet (see
  [NVDEC hardening](#nvdec-hardening) for why ferricast decides it per
  packet).
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
  mode copies into pooled page-locked buffers and records the slot's
  `ready` event (`Decoder<Host>::next_frame` waits for it; without
  `cuMemHostAlloc` it falls back to pageable memory and a stream sync);
  device mode records the
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
    `Error::InvalidPacket`. The driver is not a trust boundary: its parsers
    fault on several other malformed inputs, see
    [NVDEC hardening](#nvdec-hardening).
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

## Integration decisions (ferricast's decoders and encoders)

`shiguredo_nvcodec`, its patched copy in `vendor/`, the `[patch.crates-io]`
entry and `exclude = ["vendor"]` are gone. The consumers' side is recorded
in `design/CODECS.md` ("NVIDIA"); what changed here for them:

- The crate is a dependency of `ferricast-codec` only (feature `cuda`,
  target-gated to Linux and Windows); the encoder and decoder reach it
  through `ferricast_codec::cuda::nvcodec`, so Apple / Android builds never
  compile it even with the NVIDIA features on.
- `decode::DeviceFrame::shared_buffer()` lends the pooled buffer as
  `Arc<DeviceBuffer>` (the slot holds an `Arc` now), which is what
  `encode::DeviceFrame` caches by: an NVDEC → NVENC pipeline registers each
  pool buffer once. A slot whose buffer is still held elsewhere when it is
  reused (an encoder that kept the `Arc` past the frame) gets a new buffer
  instead of being overwritten (`DeviceSlot::make_exclusive`); ferricast's
  encoder holds the frame until the packet is out, so this does not happen
  in steady state.
- `Encoder::encode_pitched` (unsafe, same contract as `encode_array`, same
  key space): pitched memory owned elsewhere, i.e. a LINEAR DMA-BUF the
  importer maps as `PitchedView`, is registered as
  `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR` and encoded in place. The
  alternative, a device copy into an `Arc<DeviceBuffer>` staging buffer,
  does not work with the cache: `Arc::get_mut` fails while the encoder's
  `Weak` exists, and forgetting the registration every frame would
  register per frame. Verified identical to host input
  (`tests/encode.rs::pitched_input`) and on GBM LINEAR buffers through
  ferricast.
- NVENC accepts the CUDA arrays EGL hands out for tiled (block-linear)
  DMA-BUFs: verified end to end (GBM buffer written by GLES → importer →
  `encode_array` → decode, 41 dB on a gradient pattern).
- Invalidation order on a capture pool change: NVENC's registrations first
  (`clear_input_caches` / `invalidate_array`), then the importer's
  (`clear` / `forget`), so an array is never unregistered from CUDA while
  NVENC still has it registered. ferricast's encoder also declares the
  `Encoder` before the importer so drop order is the same.
- NVENC converts ARGB / ABGR input with the matrix named in the VUI
  (BT.709 and BT.601 both verified round trip), at limited range.
- The GBM / EGL / GLES test writer moved from `tests/interop.rs` to
  `tests/support/dmabuf.rs`; ferricast's root `tests/nvidia_e2e.rs`
  includes the same file with `#[path]`.

## NVDEC hardening

The premise that the driver is the trust boundary (malformed input fails
inside it, cleanly) is wrong: NVIDIA's parsers and `cuvidDecodePicture`
fault on several malformed inputs (driver 595.71.05, RTX 3060), killing the
process. A fuzz campaign over the eleven test clips (bit flips, byte
overwrites, truncation, splices, inserted / deleted blocks, NAL- and
OBU-aware mutations, dropped packets, discontinuities, host and device
mode, random display delays and framing) found these, each cut down to the
input that triggers it; each now has a guard in front of the driver, which
refuses the packet or picture with a typed error and leaves the decoder
usable.

| Input | Fault | Guard |
| --- | --- | --- |
| HEVC VPS cut short (26-byte packet, VPS stopped inside `profile_tier_level()`) | SIGSEGV in `cuvidParseVideoData` | `decode/hevc.rs`: every VPS parsed in full (§7.3.2.1 with `hrd_parameters()`, ranges checked) to its `rbsp_trailing_bits()`, else `Error::InvalidPacket` |
| AV1 inter frame after packet loss whose references were never decoded | SIGSEGV in `cuvidDecodePicture` | `decode/parser.rs` `References`: the surfaces NVDEC decoded since the last shown key frame; an inter / switch frame referencing any other (or `0xff`) is refused (`Error::MissingReference`) and never displayed |
| AV1 picture with `nBitstreamDataLen` wrapped negative (~4 GiB) or tile offsets past it | `memmove` fault in `cuvidDecodePicture` | `check_bitstream`: length within what was fed (and < 2 GiB), every tile's start ≤ end ≤ length |
| AV1 sequence header with damaged fields | SIGSEGV in `cuvidParseVideoData` | `decode/obu.rs`: every sequence header parsed in full (§5.5) to its trailing bits |
| AV1 temporal unit whose last OBU size runs past the packet | SIGBUS in `cuvidDecodePicture` | OBU framing check (already there) |
| VP8 inter frame before any key frame that refreshes the golden frame and copies it to the alternate reference | SIGSEGV in `cuvidParseVideoData` | `decode/vp8.rs`: inter frames refused until a key frame was accepted (`Error::MissingReference`); frame tag, start code, sizes and partition layout checked too |
| VP9 packet ending in what reads as a superframe index with frame sizes past the packet (a random 91-byte packet) | SIGSEGV in `cuvidParseVideoData` | `decode/vp9.rs`: superframe sizes must fit before the index |

Truncating every NAL unit in front of the first slice of the HEVC clips
(343 packets) crashed 8 before the VPS check and none after; every
truncation of an SPS or PPS was refused cleanly by the driver even before,
so they are not parsed. AV1 with 25% packet loss crashed 20 of 40 runs
before the reference check and 0 of 100 after (H.264, HEVC, VP8, VP9: 0 of
30 each, before and after). The campaign itself (two five-minute runs per
clip): 11 of 22 runs crashed before (HEVC 5 of 6, VP8 2 of 2, AV1 4 of 4),
only the AV1 runs after (see the residual risk below).

Framing, which silently lost or corrupted pictures:

- `CUVID_PKT_ENDOFPICTURE` on a packet holding several access units decodes
  only the first; on part of one, decodes the part as a whole picture.
  MPEG-TS (HLS, Miracast) gives one packet per PES, which promises neither.
  ferricast's integration (`crates/decoder/src/nvdec.rs`,
  `PictureFraming`) walks each packet's NAL units once
  (`bitstream::nal_units`, `PacketShape`) and sets the flag per packet
  (`Packet::whole_picture`) only after the stream showed two packets in a
  row that each held one access unit and were followed by one starting a
  new one; never on a keyframe or on a packet of 60 KiB or more (a PES cut
  at its 16-bit length limit); the first packet that continues an access
  unit turns it off for the stream. Otherwise the parser finishes each
  picture when the next starts, one packet later. A split that first shows
  on a non-key packet under 60 KiB after the flag was trusted still decodes
  that one picture from its first part (and the pictures referencing it
  until the next keyframe): the residual risk of saving a packet of
  latency.
- A whole packet after one sent without the flag, in a stream without
  access unit delimiters (MP4 / Matroska framing, which ferricast's
  demuxers give), makes the parser read the open picture and the new packet
  as one picture: one picture lost and its dependants corrupted. The
  decoder sends a lone access unit delimiter between them
  (`Decoder::close_picture`); HEVC needed it too.
- Non-whole HEVC packets may cut a VPS: the VPS at the end of such a packet
  (or a start code cut short) is held back and checked with the next
  packet (`nal::HeldBack`, at most 64 KiB, dropped at a discontinuity or
  flush).

Other fixes: a failed `cuvidReconfigureDecoder` recreates the decoder
instead of failing every later sequence the same way; any error after a
copy is queued synchronises the stream before the destination can go back
to its pool or be freed; ferricast maps `FramesInFlight` and
`CUDA_ERROR_OUT_OF_MEMORY` during decode to `UnsupportedStream`, so
selection hands the stream to the next backend; its host lending limit no
longer counts the decoder's own `Arc`. Device mode still hands out 10-bit
pictures as P010 when NV12 is asked for (no GPU narrowing yet).

Host mode copies each picture into a pooled page-locked buffer
(`cuMemHostAlloc`, freed with its context held) with an event recorded
after the copy; `next_frame` waits for it, so the parser no longer
synchronises the stream inside its display callback, and the DMA runs at
pinned speed. Without `cuMemHostAlloc` (or when it fails) the slot falls
back to pageable memory and a stream sync, as before. One host stream at display
delay 0 (fps, 3 runs, idle GPU): 1080p H.264 622 → 686 (device 760),
1080p HEVC 945 → 1112 (device ~1390), 2160p HEVC ~245 → 289 (device 350).
The rest of the gap is the copy itself, serialised with the next decode
at delay 0 (`next_frame` must wait before the next packet is parsed); at
delay 1 the next picture decodes while the copy runs and host equals
device (2160p HEVC 365 / 364).

Packet scanning: each H.264 / HEVC packet was split into an allocated
`Vec` of NAL units three times in ferricast (parameter sets for the
hand-over, VUI colour, HDR SEI, the last on every packet even for SDR
streams) and scanned byte by byte in ferricast-nvcodec for the SPS crop.
Now `bitstream::nal_units` iterates without allocating, with a start code
search that skips three bytes at a time (the same in `decode/nal.rs`); the
integration walks a packet once for colour, HDR and framing, only parameter
set and SEI units are read; `ParameterSets` copies a set only when it
changed; ferricast-nvcodec walks once for the crop and the VPS check and,
for H.264, stops at the first slice. Per packet (scan work only, ferricast plus
ferricast-nvcodec): 1080p H.264 59.4 → 9.8 µs and 3.07 → 0.03
allocations, 1080p HEVC 59.8 → 14.7 µs, 2160p HEVC 295 → 72 µs.

### Residual risk

- AV1: a frame / tile group OBU with damaged fields inside well-formed
  framing still faults inside `cuvidParseVideoData` (in its tile group
  parser, reading `tile_size_minus_1` far outside the temporal unit: the
  tile sizes it follows run off the data). The fuzz campaign hits it within
  minutes on both AV1 clips. A guard needs the full `uncompressed_header()`
  (§5.9) to find where the tile group starts, which in turn needs the
  state of every reference slot (frame sizes for `frame_size_with_refs`,
  global motion and film grain parameters loaded from the primary
  reference, order hints for skip mode). Realistic sources rarely produce
  it (packet loss and truncation are caught; it takes bit errors inside an
  OBU that keep its framing valid), but untrusted AV1 should be decoded in
  a helper process (or by dav1d) until that parser exists.
- Anything the campaign did not reach: the driver remains unaudited; a
  helper process is the only complete answer.

## Capture zero-copy

What it took for real screen capture (not GBM test buffers) to reach
NVENC without a CPU copy, measured on an RTX 3060, driver 595.71.05,
niri 26.04, 1920×1080 outputs (one normal, two rotated 90° / 270°).

### wlr-screencopy (wayland-direct)

- **Allocation.** The pool used to allocate LINEAR only (an `mmap`
  readback needed row-major bytes). NVIDIA's GBM refuses LINEAR with
  `GBM_BO_USE_RENDERING`, so every allocation failed, the worker switched
  to `wl_shm` for good and every frame was a CPU frame (479/479). Now the
  modifier list is the compositor's `linux-dmabuf` feedback for the fourcc
  ∩ the Vulkan importer's readable modifiers, tried first narrowed to what
  the consumer imports without a copy (`CaptureConfig::consumer_modifiers`,
  filled from `ferricast_encoder::dmabuf_modifiers()`, i.e. NVENC's EGL
  modifiers), then the readable list; GBM picks the layout. Without the
  Vulkan importer the list is LINEAR only and the `mmap` readback stays.
  `INVALID` (implicit) is never chosen: both importers need the explicit
  layout. On this machine: `0x0300000000606014` (block-linear), in NVENC's
  list.
- **CPU fallbacks read back through Vulkan**, never `mmap`: rotated /
  y-inverted outputs for a consumer that does not turn GPU frames (see
  "Rotated outputs") and layouts a CPU encoder needs.
  An `mmap` of a tiled buffer reads scrambled tiles; one of an NVIDIA
  LINEAR buffer stops showing the GPU's view after the first pages (see
  "Observed hazard"). The Vulkan staging buffer is now `HOST_CACHED`
  when the device offers it: uncached (write-combined) reads of a 1080p
  frame took most of a frame period (rotated output: 31 → 58 fps).
- **Producer.** Frames carry `GpuFrame::producer`, the render node the
  bos were allocated on (the feedback's main device); NVENC's importer
  checks it instead of `FERRICAST_DRM_RENDER_NODE` (still read for frames
  without one).
- **Damage.** Frames are requested with `copy_with_damage`; the
  compositor answers when the output changed (niri renders the full
  output into the buffer every time, `render_output` with buffer age 0,
  so every pooled buffer is complete). The stream's `FramePacer` repeats
  the last frame (same `GpuFrame`, same lease) while nothing changes, so
  the cadence and keyframe interval the receivers need are unchanged.
  `FERRICAST_WLR_FULL_COPY=1` requests every frame with `copy`.
- **Synchronisation.** Frames leave with `FrameSync::Ready`. wlr-screencopy
  `ready` means the copy is complete; niri sends it only after the render's
  `SyncPoint` signalled (`submit_after_sync`, a calloop source on the
  exported fence). Correctness on NVIDIA rests on that; the implicit poll
  would not catch an early `ready`. niri 26.04 advertises neither
  `ext_image_copy_capture_manager_v1` nor `wp_linux_drm_syncobj_manager_v1`
  to this client (logged at start); ext-image-copy-capture has no explicit
  sync either, and linux-drm-syncobj-v1 covers client surfaces, not capture.

Live (`mycap`: wayland-direct → `H264Encoder`, NVENC, readbacks counted
by wrapping the frame's importer; 10 s, 60 fps requested):

| Output | Before | Now |
|---|---|---|
| DP-3, normal, every frame (`FULL_COPY`) | 479/479 CPU frames (`wl_shm`), 8 % of a core | 601/601 GPU frames, 0 readbacks, 0.4–1.0 % of a core, 60 fps |
| DP-3, damage + pacing, idle screen | — | 1 capture, 512 repeats, 0.2 % of a core |
| HDMI-A-1, 90°, every frame | 597 CPU frames (`wl_shm`), 61 % | 576 CPU frames (Vulkan readback + CPU rotate + scale), 50 %, 57.6 fps |
| DP-2, 270°, every frame | — | 577 CPU frames, 45 %, 57.7 fps |
| HDMI-A-1, damage + pacing | — | 7 captures, 394 repeats, 20 % |

Picture checks: the Vulkan readback of a GPU frame equals `grim`'s
screenshot of DP-3 bit for bit (PSNR ∞); NVENC's stream decoded by
ffmpeg is 51–52 dB from the readback and from the screenshot; rotated
outputs read back in the screen's orientation (53 dB / 35 dB against a
later screenshot of moving content).

### Rotated outputs

A capture of a rotated or flipped output holds the scanout buffer, not
what the user sees. The correction travels with the frame instead of
being undone by capture for every consumer:

- `ferricast_core::Orientation` (a mirror, then a clockwise quarter-turn
  count: the eight `wl_output` transforms, with screencopy's `Y_INVERT`
  composed in by `Orientation::then`) is `GpuFrame::orientation`.
  `GpuFrame::width` / `height` stay the buffer's; `oriented_size()`,
  `CapturedFrame::width()` / `height()` and `into_cpu()` are upright.
- wayland-direct sets it from the output's live `wl_output` transform and
  `Y_INVERT`; PipeWire from `SPA_META_VideoTransform` (requested as a
  meta, values read like `wl_output`'s), turning its CPU frames itself.
- Capture hands a rotated output over as a `GpuFrame` only when the
  consumer turns GPU frames (`CaptureConfig::consumer_orients`, filled by
  `ferricast_encoder::orients_gpu_frames()`: NVENC's DMA-BUF import works
  on this machine). Otherwise it reads the buffer back and turns it on
  the capture thread, as before. `RawFrame::oriented` is the one CPU
  implementation, used by capture and by `into_cpu`.
- Every consumer stays upright whichever encoder the stream picks: NVENC
  encodes from the DMA-BUF directly only when the orientation is the
  identity and otherwise fits it with the kernels
  (`Geometry::with_orientation`, letterboxed by the oriented size);
  VA-API's VPP path takes only upright frames and sends the rest through
  `into_cpu`; CPU encoders call `into_cpu`. A consumer that reads the
  DMA-BUF itself must check the orientation, as the field documents.

Live (`mycap`, `FERRICAST_WLR_FULL_COPY=1`, 10 s, 60 fps, NVENC H.264 at
the output's upright size; readbacks counted on the frame's importer):

| Output | CPU rotation (`consumer_orients = false`) | GPU rotation |
|---|---|---|
| DP-3, normal | — | 601 GPU frames, 0 readbacks, 0.6 % of a core, 60.0 fps |
| HDMI-A-1, 90° | 601 CPU frames (one Vulkan readback + CPU turn each), 18 %, 60.1 fps | 600 GPU frames, 0 readbacks, 0.6 %, 60.0 fps |
| DP-2, 270° | 601 CPU frames, 19 %, 60.0 fps | 600 GPU frames, 0 readbacks, 0.5 %, 60.0 fps |
| HDMI-A-1 / DP-2 scaled to 606×1080 | — | 600 GPU frames, 0 readbacks, 0.5–0.7 % |

Picture check: the last decoded picture (ffmpeg) against a `grim`
screenshot taken right after: DP-3 43.3 dB, HDMI-A-1 33.8 dB, DP-2
40.9 dB, the same as the CPU-rotation path's stream (33.8 / 40.9 dB);
against the screenshot turned 180° both paths give 13 dB, so the pictures
are upright. `tests/nvidia_e2e.rs` checks a 90° turn and a flipped 270°
with `Y_INVERT` against `RawFrame::oriented` (41 / 46 dB).

### PipeWire

- **Explicit sync.** The worker offers a `ParamBuffers` variant that
  requires `SPA_META_SyncTimeline` (and the meta itself) before the plain
  one; a producer that supports it (Mutter, KWin) adds the acquire and
  release syncobjs to every buffer. The worker waits for the acquire point
  (`DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` with `WAIT_FOR_SUBMIT`, 100 ms, on the
  importer's render node; `ferricast_core::Syncobj`) before anything reads
  the buffer and emits `FrameSync::Ready`; the release point is signalled
  when the buffer is re-queued: for leased GPU frames when the lease drops,
  i.e. after NVENC locked the picture. Without the meta frames are
  `FrameSync::Implicit`. niri's own PipeWire producer has no SyncTimeline
  (it waits for its render before queuing), so this path is unit-tested
  (syncobj wait / signal / sync-file import, pod order) but has not met a
  real explicit-sync producer.
- **LINEAR DMA-BUFs** leave as `GpuFrame`s whenever the Vulkan importer
  reads them (NVIDIA lists LINEAR among its 7 readable modifiers): NVENC
  encodes them in place (`encode_pitched`), CPU encoders read back through
  Vulkan. Only buffers the importer cannot read (implicit modifier, no
  importer) are still `mmap`ped.
- **Negotiation.** Per format, a pod with the modifiers the importer
  reads *and* the consumer imports comes first, then one with every
  readable modifier, then SHM.
- **Producer.** The importer's render node (`VK_EXT_physical_device_drm`).

### NVENC's zero-copy path

- `FrameSync::Ready` maps with `FenceWait::AlreadySignalled`,
  `Implicit` with the (best-effort) poll.
- The importer is created against the first frame's producer; a frame of
  another GPU is read back without an import attempt.
- A buffer whose import failed is remembered (per `DmaBufId`) and read
  back without retrying until the capture pool changes (a description
  change clears every cache).
- Registrations unused for 600 frames (the producer discarded the
  buffer) are dropped, and at 16 the least recently used one, instead of
  keeping freed buffers' VRAM until 16 accumulate. Evictions and pool
  changes unregister with `Quiesce::idle()` (NVENC has locked every
  picture), no `cuCtxSynchronize`.
- A frame one pixel larger per axis than the session (the session rounds
  odd sizes down to even) is encoded from its top-left corner: NVENC takes
  larger inputs (`tests/nvidia_e2e.rs`,
  `odd_sized_frames_go_zero_copy_and_refused_buffers_stay_read_back`).
- **GPU fit.** Other sizes, byte orders and rotated outputs are imported
  through the same bookkeeping (`ZeroCopy::import`: producer check,
  refusal cache, pool change, stale / LRU eviction) and lent to the image
  kernels (`State::with_view`, `fit.rs`), which scale, letterbox and turn
  them by `GpuFrame::orientation` into a pooled device buffer for
  `encode_device`. A failed fit synchronises the fit stream before
  returning, so an eviction's `Quiesce::idle()` stays true.
- **Hybrid laptops.** `GpuFrame::producer` and
  `ferricast_encoder::nvenc::imports_from(producer)` give encoder
  selection what it needs to prefer a backend on the producer's GPU
  (VA-API on the iGPU zero-copy over NVENC + readback). The selection
  itself (registry order in the facades) does not use it yet.

### Open

- PipeWire explicit sync and LINEAR-as-GPU are unverified against a live
  producer here (niri's portal needs an interactive pick).
- The acquire wait blocks the PipeWire loop thread (bounded by 100 ms); a
  GPU-side wait would need a CUDA external semaphore, which CUDA cannot
  import from a sync file.

## GPU image kernels

`kernels` (`src/kernels.rs`, `src/kernels/**`): pictures whose size, orientation
or format differ from the NVENC session are fitted on the GPU into a pooled
device buffer that `Encoder::encode_device` takes. Before, every such frame
(1080p screen into a 720p receiver, a portrait monitor, NVDEC P010 into an 8-bit
session) went readback → CPU scale / narrow → upload: 28.5 ms per frame, 35 fps,
95% of a core at 1080p → 720p.

### Toolchain: committed PTX, JIT-compiled by the driver

- The device code is `src/kernels/kernels.cu`; `scripts/kernels.sh` compiles it
  to `src/kernels/kernels.ptx`, which is committed and loaded with
  `cuModuleLoadDataEx` (error log captured into `Error::KernelLoad`). The crate
  keeps no build script and needs no CUDA toolkit; it cross-compiles as before.
- Compiler: nixpkgs' CUDA 12.6 `nvcc` (from the nixpkgs the flake locks,
  `--inputs-from`; unfree, so `NIXPKGS_ALLOW_UNFREE=1 --impure`) with GCC 13 as
  host compiler. It is the oldest CUDA nixpkgs still builds (11.x and 12.0–12.4
  are "removed: unmaintained"); CUDA 13 can no longer target anything below
  sm_75.
- Target: `-arch=compute_52` (Maxwell 2, the oldest GPU of any driver the crate
  supports), no fast math (the kernels must match the CPU references).
- PTX ISA: nvcc 12.6 writes `.version 8.5`, which only R560+ drivers JIT. The
  script lowers the directive to `.version 6.0` (CUDA 9; the ISA that
  introduced the current memory model, so the code means the same) and has
  `ptxas` verify the claim for sm_52, sm_75 and sm_86: ptxas rejects any
  instruction or directive newer than the declared ISA (checked:
  `shfl.sync` under `.version 5.0` → "requires PTX ISA .version 6.0 or later").
  R470 (NVENC API 11.1, the oldest driver the crate supports; CUDA 11.4)
  accepts PTX ISA up to 7.4, so every supported driver can JIT it. Verified on
  driver 595 only: an R470 machine was not available.
- `#pragma unroll 1` on the data-dependent tap loops: 102 KB of PTX instead of
  431 KB.
- JIT cost: ~120 ms the first time a PTX is seen, ~0.3–0.5 ms afterwards (the
  driver's compute cache). `Kernels::get` loads once per context and shares the
  module (`Weak` cache: unloaded with its last user).
- Rejected: NVRTC at run time (not part of the driver install, would be one more
  library to find and version-match); hand-written PTX (unmaintainable for ten
  kernels); a `build.rs` with nvcc (breaks cross-compiling and builds without
  CUDA).

### Kernels and their parameters

- Ten entry points, one parameter each, a `#[repr(C)]` struct mirrored from the
  C struct (`static_assert` sizes on the C side, `const` asserts on the Rust
  side, and a test that parses the PTX's `.param .align 8 .b8 name_param_0[N]`
  for every entry). One typed launch per kernel family
  (`rgb_to_rgb`, `rgb_to_nv12`, `plane`); no `void**` in the safe API.
  `Module::launch` (unsafe, in `cuda.rs`) is the only raw launch.
- One pass per output plane does everything: crop, orientation (the 8 `wl_output`
  transforms: optional horizontal mirror, then a clockwise rotation; `Y_INVERT`
  is `Orientation::FLIP_VERTICAL`), resampling into the placement rectangle,
  letterbox fill, and the format conversion. The geometry is lowered on the
  host to an integer affine map (oriented → source pixel, crop folded in) and
  two scale ratios.
- Filter: `fast_image_resize`'s convolution with adaptive support (what
  `FrameScaler` runs): triangle (`Filter::Bilinear`) or box (`Filter::Box`)
  stretched by the downscale ratio, so a 4× downscale averages every covered
  pixel (one-pixel text strokes become an even 25% grey instead of aliasing).
  Taps and weights are computed per output pixel in f32 with no rounding
  between the axes (the CPU scaler rounds to u8 between passes). Same-size
  geometries (copy, crop, rotate, convert) read one sample directly.
- RGB → NV12: one thread per 2×2 block; luma per pixel and chroma from the
  rounded average of the block's RGB, with the `yuv` crate's prebuilt 13-bit
  BT.709 / BT.601 limited / full tables and rounding (its scalar path).
  P016 → NV12 rounds (`(v + 128) >> 8`); P016 → P010 keeps 10 bits.
- Inputs: `Input::Array` (tiled DMA-BUF, read through a point-sampled texture
  object; the array's descriptor must be 4 × u8), `Input::Packed` (LINEAR
  DMA-BUF or any pitched RGB, read with `ld.global.nc`), `Input::Nv12` /
  `Input::P016` (NVDEC output). Outputs: RGB in either byte order, NV12, P010.
  NV12/P010 → RGB and NV12 → P010 are refused.
- Host checks before every launch: geometry inside both pictures, source and
  destination sizes and chroma rows, 4:2:0 crop / placement origins even,
  alignment of every plane (4 bytes for RGB, 2 for 16-bit), every object on the
  converter's context. Texture reads are clamped by the hardware anyway.
- Texture objects: created per frame (~1.4 µs host cost per call including the
  descriptor query), destroyed once an event recorded after the kernel has
  completed (checked without waiting at the next call; the converter waits only
  past 16 in flight and on drop). A texture cache keyed by array handle was
  rejected: arrays are owned by the importer, whose invalidation the converter
  cannot see.
- `FramePool`: buffers of one format and size, handed out again once only the
  pool holds the `Arc` (NVENC's pending picture holds a strong `Arc` until its
  bitstream is locked). Nothing is allocated per frame in steady state.
- Ordering is the caller's, with events; nothing waits on the host. Additions
  to `cuda.rs` (own impl blocks): `Module`, `Function`, `Launch`,
  `TextureObject`, `ArrayView::descriptor`, `DeviceBuffer::as_view`. Loader:
  the module / launch / texture / `cuArrayGetDescriptor_v2` entry points are
  optional (every driver has them; only the kernels fail without them).
  `cuda_extra.h` declares `cuModuleLoadDataEx` and `cuArrayGetDescriptor_v2`.

### ferricast integration (`crates/encoder/src/nvenc/fit.rs`)

- `NvencEncoder::submit` tries the GPU fit before any readback: NVDEC pictures
  (`CudaSurface`) of another size or P010 into an 8-bit session; DMA-BUF
  captures of another size or byte order (imported and mapped through the
  zero-copy path's importer, `zero_copy::State::with_view`); CPU pictures of
  another size or P010 into NV12 (uploaded on the fit stream into a staging
  buffer, the frame kept until its packet is out). The geometry is
  `FrameSize::letterbox_in`, the CPU scaler's. Any failure is logged (first,
  then every 3600th) and the frame takes the CPU path.
- RGB sessions keep NVENC's own ARGB → YUV conversion: at 1280×720 scale +
  encode ran at 1043 fps from ARGB and 1049 fps from kernel-made NV12, no
  measurable difference, and switching would change the session format for
  CPU frames too. The NV12 kernel serves NV12 sessions.
- CPU pictures: upload + GPU fit beat `FrameScaler` (release): 1080p → 720p
  0.77 ms vs 2.2 ms, 4K → 1080p 3.2 vs 6.7 ms, portrait 1080×1920 → 1080p
  0.73 vs 6.9 ms, before the CPU path's own upload.
- The CPU path narrows P010 by truncating (`FrameView::narrowed`); the GPU
  rounds, so the two differ by at most one level (51 dB).

### Measured (RTX 3060, driver 595.71.05, release, idle GPU)

GPU time per frame (frames queued back to back), host cost per call ~1.2 µs
(pitched) / 2.6 µs (array, texture included):

| Operation | µs/frame |
|---|---|
| 1080p → 720p BGRA (pitched / array) | 164 / 127 |
| 1080p → 720p NV12, scale + CSC (pitched / array) | 140 / 146 |
| 1080p BGRA → NV12, CSC only | 45 |
| 4K → 1080p BGRA / NV12 | 407 / 416 |
| 1080×1920 → 1920×1080 rotated (BGRA) | 95 |
| 1080×1920 → 1920×1080 letterboxed (BGRA) | 175 |
| 4K P010 → NV12 | 184 |
| 4K P010 → 1080p NV12 | 394 |

`cargo run --release -p ferricast-nvcodec --example kernels_bench`.
Resampling is ALU-bound (per-pixel tap weights); a shared-memory weight table
per block is the next step if it ever matters.

Quality: crop / rotate / flip / letterbox fill / P016 → NV12 / same-size
RGB → NV12 bit-exact against independent CPU references (odd sizes, 1×N, N×1,
1×1); scaling within one level of an f64 `fast_image_resize` reference
(PSNR 71.9 dB to bit-exact); against ferricast's own CPU path (`FrameScaler`, `yuv`
crate): 54.6–68.7 dB, max difference 1 (RGB → NV12: max 1, from the `yuv`
crate's AVX2 path, which rounds differently from its scalar path).

End to end (`tests/nvidia_e2e.rs`, frames carry no CPU importer so a readback
would fail the encode; `CudaSurface::readbacks()` unchanged for NVDEC):
GBM tiled 1920×1080 → 1280×720 H.264 42.0 dB against the CPU path's picture
after a lossy round trip; GBM LINEAR 1080×1920 → 1920×1080 letterboxed
46.2 dB; NVDEC P010 → 8-bit H.264 at the same size 57.1 dB, at half size
57.7 dB, 0 readbacks. Throughput 1080p DMA-BUF → 720p H.264 (release):
1093 fps, 0.92 ms/frame, 2% of a core, against 393 fps and 69% of a core for
the CPU path with an instant readback stand-in (the real Vulkan readback is on
top of that).

### Hazards found on the way

- `DeviceBuffer::upload` / `fill` run on the legacy default stream: a pageable
  `cuMemcpy2D` may return before the DMA lands and `cuMemsetD2D8` is
  asynchronous, and neither is ordered with the crate's non-blocking streams.
  Work on a non-blocking stream right after them can see stale memory (the
  kernel tests first failed this way). Tests synchronise the context after
  uploading; use `upload_async` / `fill_async` on the consuming stream.
