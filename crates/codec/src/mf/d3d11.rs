//! Direct3D 11 for the Media Foundation codecs: the device a transform
//! works on ([`Device`]) and pictures that stay in D3D11 textures
//! ([`D3d11Texture`], a [`NativeSurface`]).
//!
//! # Threading
//!
//! Direct3D 11 devices and resources are free-threaded; only the
//! immediate context is not. Every device this module creates or adopts
//! is switched to multithread protection, and every use of an immediate
//! context here is bracketed by its `ID3D11Multithread` lock, the same
//! lock Media Foundation takes around its own work on the device. A
//! texture can therefore be read back on any thread while its decoder
//! keeps decoding on its worker.
//!
//! A decoded texture belongs to its decoder's sample pool: the
//! [`D3d11Texture`] keeps the output sample alive (the slot returns to
//! the pool when the last clone of the frame drops), so a consumer that
//! hoards frames starves the decoder. Decoders ask for spare samples
//! ([`Transform::reserve_output_samples`]) to leave room for a few.

use std::any::Any;
use std::fmt;
use std::sync::Once;

use bytes::Bytes;
use ferricast_core::{
    CodecError, CodecResult, FrameError, NativeSurface, Operation, PixelFormat, SurfaceKind,
    SurfacePixels,
};
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_10_0, D3D_FEATURE_LEVEL_10_1,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_MAP_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Multithread, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_FORMAT_P010,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Media::MediaFoundation::{
    IMFDXGIBuffer, IMFDXGIDeviceManager, IMFMediaBuffer, IMFSample, MF_SA_D3D11_AWARE,
    MF_SA_MINIMUM_OUTPUT_SAMPLE_COUNT, MFCreateDXGIDeviceManager, MFCreateDXGISurfaceBuffer,
    MFT_MESSAGE_SET_D3D_MANAGER,
};
use windows::Win32::System::Com::CoIncrementMTAUsage;
use windows::core::Interface;

use super::{BACKEND, MfResult, Transform};

/// Keeps the process's multithreaded apartment alive once a D3D11
/// object may outlive the worker thread that made it: without it, the
/// last worker leaving the MTA could unload the transform's DLL while a
/// consumer still holds one of its textures.
fn pin_mta() {
    static PINNED: Once = Once::new();
    PINNED.call_once(|| {
        // SAFETY: plain call; the cookie is deliberately never released.
        if let Err(error) = unsafe { CoIncrementMTAUsage() } {
            tracing::warn!(%error, "CoIncrementMTAUsage failed; D3D11 frames may outlive COM");
        }
    });
}

/// The `ID3D11Multithread` lock of a device's immediate context, held
/// until drop.
struct ContextLock {
    context: ID3D11DeviceContext,
    lock: ID3D11Multithread,
}

impl ContextLock {
    fn enter(device: &ID3D11Device) -> windows::core::Result<Self> {
        // SAFETY: plain COM calls on a live device; the immediate context
        // is only used while the lock taken here is held.
        unsafe {
            let context = device.GetImmediateContext()?;
            let lock = context.cast::<ID3D11Multithread>()?;
            lock.Enter();
            Ok(Self { context, lock })
        }
    }
}

impl Drop for ContextLock {
    fn drop(&mut self) {
        // SAFETY: balances the `Enter` of `ContextLock::enter`.
        unsafe { self.lock.Leave() };
    }
}

/// Turns on multithread protection of `device`'s immediate context,
/// which Media Foundation requires of a device it shares.
fn protect(device: &ID3D11Device, operation: Operation) -> CodecResult<()> {
    // SAFETY: plain COM calls on a live device.
    unsafe {
        let context = device
            .GetImmediateContext()
            .mf(operation, "ID3D11Device::GetImmediateContext")?;
        let lock = context
            .cast::<ID3D11Multithread>()
            .mf(operation, "QueryInterface(ID3D11Multithread)")?;
        // Returns the previous setting, not an error.
        let _previous = lock.SetMultithreadProtected(true);
    }
    Ok(())
}

/// A Direct3D 11 device and the Media Foundation device manager that
/// hands it to transforms. Lives on the session's worker thread.
#[allow(
    missing_debug_implementations,
    reason = "holds COM interfaces, which have no useful Debug"
)]
pub struct Device {
    device: ID3D11Device,
    manager: IMFDXGIDeviceManager,
}

impl Device {
    /// A new hardware device with video support, for a decoder.
    ///
    /// # Errors
    /// [`CodecError::BackendUnavailable`] when no hardware device can be
    /// created (no GPU, a remote session without one), or the device
    /// manager failure.
    pub fn hardware() -> CodecResult<Self> {
        pin_mta();
        let levels: [D3D_FEATURE_LEVEL; 4] = [
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
        ];
        let mut device = None;
        // SAFETY: the out-pointer is valid for the call; no adapter, no
        // software module.
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                Some(&levels),
                D3D11_SDK_VERSION,
                Some(&raw mut device),
                None,
                None,
            )
        }
        .map_err(|error| CodecError::BackendUnavailable {
            backend: BACKEND,
            reason: format!(
                "no Direct3D 11 hardware device with video support (HRESULT {:#010X}: {})",
                error.code().0,
                error.message()
            ),
        })?;
        let device = device.ok_or_else(|| CodecError::BackendUnavailable {
            backend: BACKEND,
            reason: "D3D11CreateDevice returned no device".into(),
        })?;
        Self::adopt(device, Operation::Open)
    }

    /// A device manager around the device `texture` lives on, for an
    /// encoder fed that device's textures.
    ///
    /// # Errors
    /// The device manager failure.
    pub fn of_texture(texture: &D3d11Texture) -> CodecResult<Self> {
        Self::adopt(texture.device(Operation::Encode)?, Operation::Encode)
    }

    fn adopt(device: ID3D11Device, operation: Operation) -> CodecResult<Self> {
        protect(&device, operation)?;
        let mut token = 0u32;
        let mut manager = None;
        // SAFETY: valid out-pointers; the device is live.
        unsafe {
            MFCreateDXGIDeviceManager(&raw mut token, &raw mut manager)
                .mf(operation, "MFCreateDXGIDeviceManager")?;
        }
        let manager = manager.ok_or_else(|| {
            CodecError::backend(
                BACKEND,
                operation,
                "MFCreateDXGIDeviceManager returned no manager",
            )
        })?;
        // SAFETY: plain COM call; `token` is the manager's reset token.
        unsafe { manager.ResetDevice(&device, token) }
            .mf(operation, "IMFDXGIDeviceManager::ResetDevice")?;
        Ok(Self { device, manager })
    }

    /// Whether `texture` lives on this device.
    #[must_use]
    pub fn holds(&self, texture: &D3d11Texture) -> bool {
        texture
            .device(Operation::Encode)
            .is_ok_and(|device| device == self.device)
    }
}

impl Transform {
    /// Hands `device` to the transform (`MFT_MESSAGE_SET_D3D_MANAGER`),
    /// before its media types are set. `false` when the transform is not
    /// Direct3D 11 aware (it then keeps working in system memory).
    ///
    /// # Errors
    /// An aware transform refusing the device.
    pub fn attach_d3d11(&mut self, device: &Device) -> CodecResult<bool> {
        // SAFETY: plain COM calls on the live transform.
        let aware = unsafe {
            self.mft
                .GetAttributes()
                .and_then(|attributes| attributes.GetUINT32(&MF_SA_D3D11_AWARE))
                .unwrap_or(0)
                != 0
        };
        if !aware {
            return Ok(false);
        }
        // SAFETY: the message takes the manager's interface pointer; the
        // transform keeps its own reference.
        unsafe {
            self.mft.ProcessMessage(
                MFT_MESSAGE_SET_D3D_MANAGER,
                device.manager.as_raw() as usize,
            )
        }
        .mf(Operation::Open, "MFT_MESSAGE_SET_D3D_MANAGER")?;
        Ok(true)
    }

    /// Asks a Direct3D transform for `extra` more output samples than it
    /// needs, so frames a consumer still holds do not stall it. Ignored
    /// by transforms without the attribute.
    pub fn reserve_output_samples(&self, extra: u32) {
        // SAFETY: plain attribute calls on the live transform.
        let set = unsafe {
            self.mft
                .GetOutputStreamAttributes(0)
                .and_then(|attributes| {
                    attributes.SetUINT32(&MF_SA_MINIMUM_OUTPUT_SAMPLE_COUNT, extra)
                })
        };
        if let Err(error) = set {
            tracing::debug!(transform = %self.name, %error, "MF_SA_MINIMUM_OUTPUT_SAMPLE_COUNT not accepted");
        }
    }
}

/// The visible part of a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge, in pixels.
    pub x: u32,
    /// Top edge, in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// The decoder output sample a texture belongs to, kept for its
/// reference count only.
struct SampleLease(#[expect(dead_code, reason = "held for its reference count")] IMFSample);

// SAFETY: Media Foundation samples are free-threaded COM objects created
// in the multithreaded apartment (every MF worker is an MTA thread, and
// `pin_mta` keeps the MTA alive); the lease never calls a method on the
// sample, it only releases it on drop, which COM allows from any thread.
unsafe impl Send for SampleLease {}
// SAFETY: as above: shared references give no access to the sample.
unsafe impl Sync for SampleLease {}

/// A picture in one array slice of an `ID3D11Texture2D` (NV12, P010,
/// BGRA or RGBA), with its visible rectangle.
#[derive(Clone)]
pub struct D3d11Texture {
    texture: ID3D11Texture2D,
    subresource: u32,
    format: PixelFormat,
    visible: Rect,
    lease: Option<std::sync::Arc<SampleLease>>,
}

impl fmt::Debug for D3d11Texture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D3d11Texture")
            .field("subresource", &self.subresource)
            .field("format", &self.format)
            .field("visible", &self.visible)
            .field("pooled", &self.lease.is_some())
            .finish_non_exhaustive()
    }
}

fn pixel_format(format: DXGI_FORMAT) -> Option<PixelFormat> {
    match format {
        DXGI_FORMAT_NV12 => Some(PixelFormat::Nv12),
        DXGI_FORMAT_P010 => Some(PixelFormat::P010),
        DXGI_FORMAT_B8G8R8A8_UNORM => Some(PixelFormat::Bgra),
        DXGI_FORMAT_R8G8B8A8_UNORM => Some(PixelFormat::Rgba),
        _ => None,
    }
}

impl D3d11Texture {
    /// Wraps array slice `subresource` of `texture`, of which `visible`
    /// is the picture. Turns on multithread protection of the texture's
    /// device (see the module docs).
    ///
    /// # Errors
    /// [`CodecError::Backend`] for a texture format other than NV12, P010,
    /// BGRA or RGBA, a slice past the array, or a visible rectangle
    /// outside the texture.
    pub fn new(texture: ID3D11Texture2D, subresource: u32, visible: Rect) -> CodecResult<Self> {
        Self::build(texture, subresource, visible, None, Operation::Encode)
    }

    /// The texture behind a Direct3D decoder output sample; `None` when
    /// the sample holds system memory.
    ///
    /// # Errors
    /// The sample's buffer cannot be read, or its texture is unusable
    /// (see [`Self::new`]).
    pub fn from_sample(sample: &IMFSample, visible: Rect) -> CodecResult<Option<Self>> {
        let op = Operation::Decode;
        // SAFETY: plain COM calls on a live sample.
        let buffer = unsafe { sample.GetBufferByIndex(0) }.mf(op, "IMFSample::GetBufferByIndex")?;
        let Ok(dxgi) = buffer.cast::<IMFDXGIBuffer>() else {
            return Ok(None);
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: on success `raw` is an owned (AddRef'd) ID3D11Texture2D,
        // whose reference `from_raw` takes over.
        let texture = unsafe {
            dxgi.GetResource(&ID3D11Texture2D::IID, &raw mut raw)
                .mf(op, "IMFDXGIBuffer::GetResource")?;
            ID3D11Texture2D::from_raw(raw)
        };
        // SAFETY: plain COM call.
        let subresource =
            unsafe { dxgi.GetSubresourceIndex() }.mf(op, "IMFDXGIBuffer::GetSubresourceIndex")?;
        let lease = Some(std::sync::Arc::new(SampleLease(sample.clone())));
        Self::build(texture, subresource, visible, lease, op).map(Some)
    }

    fn build(
        texture: ID3D11Texture2D,
        subresource: u32,
        visible: Rect,
        lease: Option<std::sync::Arc<SampleLease>>,
        operation: Operation,
    ) -> CodecResult<Self> {
        pin_mta();
        let desc = describe(&texture);
        let fail = |what: String| CodecError::backend(BACKEND, operation, what);
        let format = pixel_format(desc.Format).ok_or_else(|| {
            fail(format!(
                "D3D11 texture format {} is not NV12, P010, BGRA or RGBA",
                desc.Format.0
            ))
        })?;
        if subresource >= desc.ArraySize.max(1) * desc.MipLevels.max(1) {
            return Err(fail(format!(
                "subresource {subresource} past a texture of {} slices",
                desc.ArraySize
            )));
        }
        let fits = visible.width > 0
            && visible.height > 0
            && visible
                .x
                .checked_add(visible.width)
                .is_some_and(|r| r <= desc.Width)
            && visible
                .y
                .checked_add(visible.height)
                .is_some_and(|b| b <= desc.Height);
        if !fits {
            return Err(fail(format!(
                "visible area {visible:?} outside a {}x{} texture",
                desc.Width, desc.Height
            )));
        }
        let surface = Self {
            texture,
            subresource,
            format,
            visible,
            lease,
        };
        protect(&surface.device(operation)?, operation)?;
        Ok(surface)
    }

    /// The texture.
    #[must_use]
    pub fn texture(&self) -> &ID3D11Texture2D {
        &self.texture
    }

    /// Its array slice.
    #[must_use]
    pub fn subresource(&self) -> u32 {
        self.subresource
    }

    /// The picture inside the texture.
    #[must_use]
    pub fn visible(&self) -> Rect {
        self.visible
    }

    /// Width and height of the whole texture.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        let desc = describe(&self.texture);
        (desc.Width, desc.Height)
    }

    fn device(&self, operation: Operation) -> CodecResult<ID3D11Device> {
        // SAFETY: plain COM call on a live texture.
        unsafe { self.texture.GetDevice() }.mf(operation, "ID3D11Texture2D::GetDevice")
    }

    /// A Media Foundation buffer around the texture, for an encoder's
    /// input sample.
    ///
    /// # Errors
    /// The `MFCreateDXGISurfaceBuffer` failure.
    pub fn media_buffer(&self) -> CodecResult<IMFMediaBuffer> {
        // SAFETY: the texture is live and `subresource` is in range
        // (checked in `build`).
        unsafe {
            MFCreateDXGISurfaceBuffer(
                &ID3D11Texture2D::IID,
                &self.texture,
                self.subresource,
                false,
            )
        }
        .mf(Operation::Encode, "MFCreateDXGISurfaceBuffer")
    }

    /// Bytes per sample of the luma (or only) plane, and whether a
    /// half-height interleaved chroma plane follows it.
    fn sample_layout(&self) -> (usize, bool) {
        match self.format {
            // No DXGI format maps to I420; it would be planar 8-bit.
            PixelFormat::Nv12 | PixelFormat::I420 => (1, true),
            PixelFormat::P010 => (2, true),
            PixelFormat::Bgra | PixelFormat::Rgba => (4, false),
        }
    }

    fn copy_out(&self) -> Result<SurfacePixels, String> {
        let desc = describe(&self.texture);
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Width: desc.Width,
            Height: desc.Height,
            MipLevels: 1,
            ArraySize: 1,
            Format: desc.Format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: u32::try_from(D3D11_CPU_ACCESS_READ.0).unwrap_or(0),
            MiscFlags: 0,
        };
        let call = |what: &str, error: windows::core::Error| {
            format!(
                "{what} failed with HRESULT {:#010X}: {}",
                error.code().0,
                error.message()
            )
        };
        // SAFETY: plain COM call on a live texture.
        let device = unsafe { self.texture.GetDevice() }
            .map_err(|e| call("ID3D11Texture2D::GetDevice", e))?;
        let mut staging = None;
        // SAFETY: valid description and out-pointer.
        unsafe { device.CreateTexture2D(&raw const staging_desc, None, Some(&raw mut staging)) }
            .map_err(|e| call("ID3D11Device::CreateTexture2D(staging)", e))?;
        let staging = staging.ok_or("CreateTexture2D returned no staging texture")?;
        let locked =
            ContextLock::enter(&device).map_err(|e| call("ID3D11Multithread::Enter", e))?;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: both resources are live and the same size and format;
        // the immediate context is used under its multithread lock.
        unsafe {
            locked.context.CopySubresourceRegion(
                &staging,
                0,
                0,
                0,
                0,
                &self.texture,
                self.subresource,
                None,
            );
            locked
                .context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&raw mut mapped))
                .map_err(|e| call("ID3D11DeviceContext::Map", e))?;
        }
        let pixels = self.pack(&mapped, &desc);
        // SAFETY: balances the successful `Map` above, under the lock.
        unsafe { locked.context.Unmap(&staging, 0) };
        drop(locked);
        pixels
    }

    /// The visible picture out of a mapped staging copy, rows packed,
    /// chroma right after the luma rows.
    fn pack(
        &self,
        mapped: &D3D11_MAPPED_SUBRESOURCE,
        desc: &D3D11_TEXTURE2D_DESC,
    ) -> Result<SurfacePixels, String> {
        let (bytes_per_sample, planar) = self.sample_layout();
        let pitch = mapped.RowPitch as usize;
        let texture_rows = desc.Height as usize;
        let Rect {
            x,
            y,
            width,
            height,
        } = self.visible;
        let (x, y, width, height) = (x as usize, y as usize, width as usize, height as usize);
        // Chroma pairs start on even columns and rows.
        let (x, y) = if planar {
            (x / 2 * 2, y / 2 * 2)
        } else {
            (x, y)
        };
        let row_bytes = if planar {
            width.div_ceil(2) * 2 * bytes_per_sample
        } else {
            width * bytes_per_sample
        };
        let mapped_rows = if planar {
            texture_rows + texture_rows.div_ceil(2)
        } else {
            texture_rows
        };
        if mapped.pData.is_null() || pitch < (x * bytes_per_sample + row_bytes) {
            return Err(format!(
                "mapped texture pitch {pitch} too short for a {width}-pixel row"
            ));
        }
        // SAFETY: a mapped staging texture is readable for `RowPitch`
        // bytes per row over all its rows, and a planar (NV12 / P010) one
        // lays its chroma plane out right after the luma rows with the
        // same pitch; the mapping stays valid until `Unmap`, after `pack`.
        let source =
            unsafe { std::slice::from_raw_parts(mapped.pData.cast::<u8>(), pitch * mapped_rows) };
        let chroma_rows = if planar { height.div_ceil(2) } else { 0 };
        let mut out = Vec::with_capacity(row_bytes * (height + chroma_rows));
        let mut copy_row = |row: usize| -> Result<(), String> {
            let start = row * pitch + x * bytes_per_sample;
            let line = source
                .get(start..start + row_bytes)
                .ok_or_else(|| format!("row {row} outside the mapped texture"))?;
            out.extend_from_slice(line);
            Ok(())
        };
        for row in y..y + height {
            copy_row(row)?;
        }
        for row in 0..chroma_rows {
            copy_row(texture_rows + y / 2 + row)?;
        }
        Ok(SurfacePixels {
            stride: u32::try_from(row_bytes).map_err(|_| "row exceeds u32".to_owned())?,
            data: Bytes::from(out),
        })
    }
}

fn describe(texture: &ID3D11Texture2D) -> D3D11_TEXTURE2D_DESC {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: valid out-pointer on a live texture.
    unsafe { texture.GetDesc(&raw mut desc) };
    desc
}

impl NativeSurface for D3d11Texture {
    fn kind(&self) -> SurfaceKind {
        SurfaceKind::D3d11Texture
    }

    fn format(&self) -> PixelFormat {
        self.format
    }

    fn readback(&self) -> Result<SurfacePixels, FrameError> {
        self.copy_out()
            .map_err(|reason| FrameError::Readback { reason })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
