//! SPA ↔ ferricast format helpers.
//!
//! Owns:
//! * The conversion table between `pipewire::spa::param::video::VideoFormat`
//!   and our public [`PixelFormat`].
//! * Builders for every SPA pod we exchange with `PipeWire`:
//!     - The list of `EnumFormat` pods (one per dmabuf format with
//!       modifier-choice + a SHM fallback) sent at `connect()`.
//!     - The fixated `EnumFormat` list re-sent from `param_changed`
//!       when the compositor returns a non-fixated modifier.
//!     - The `ParamBuffers` and `ParamMeta(Header)` pods sent from
//!       `param_changed` once the format is fully fixated.
//! * The negotiated-format struct that mirrors `VideoInfoRaw` once the
//!   compositor accepts our offer.

use std::io::Cursor;

use ferricast_core::{FerricastError, PixelFormat, Result};

use pipewire as pw;
use pw::spa::buffer::DataType as SpaDataType;
use pw::spa::param::ParamType;
use pw::spa::param::format::{FormatProperties, MediaSubtype, MediaType};
use pw::spa::param::video::VideoFormat;
use pw::spa::pod::{
    ChoiceValue, Object, Pod, Property, PropertyFlags, Value, serialize::PodSerializer,
};
use pw::spa::sys::{
    SPA_META_Header, SPA_PARAM_BUFFERS_buffers, SPA_PARAM_BUFFERS_dataType, SPA_PARAM_META_size,
    SPA_PARAM_META_type,
};
use pw::spa::utils::{Choice, ChoiceEnum, ChoiceFlags, Fraction, Id, Rectangle, SpaTypes};

use super::sync::{MetaSyncTimeline, SPA_META_SYNC_TIMELINE};

/// `SPA_PARAM_BUFFERS_metaType` (`spa/param/buffers.h`): the metas a
/// buffer must carry, as a flags choice of `1 << SPA_META_*`.
const SPA_PARAM_BUFFERS_METATYPE: u32 = 7;

/// `DRM_FORMAT_MOD_LINEAR` — buffer pixels are laid out in row-major order
/// and can be `mmap`'d for CPU read.
pub(super) const DRM_FORMAT_MOD_LINEAR: u64 = 0;
/// `DRM_FORMAT_MOD_INVALID` — the producer didn't pick a specific
/// modifier; in practice these buffers are also linearly-laid-out and
/// safe to `mmap` for read.
pub(super) const DRM_FORMAT_MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;

/// Pixel formats we know how to feed downstream encoders.
///
/// The order matters: the first that the compositor accepts becomes the
/// negotiated format. BGRx/BGRA are first because Wayland compositors
/// typically render that natively, saving a CPU swizzle.
pub(super) const SUPPORTED_FORMATS: &[VideoFormat] = &[
    VideoFormat::BGRx,
    VideoFormat::BGRA,
    VideoFormat::RGBx,
    VideoFormat::RGBA,
];

/// Modifiers we can `mmap` for CPU read without GPU help. Anything tiled
/// or compressed needs the Vulkan import path.
#[allow(
    dead_code,
    reason = "referenced from `modifier_is_cpu_readable`, which only some feature sets compile"
)]
const CPU_READABLE_MODIFIERS: &[u64] = &[DRM_FORMAT_MOD_INVALID, DRM_FORMAT_MOD_LINEAR];

/// Map a SPA [`VideoFormat`] to the public [`PixelFormat`].
pub(super) fn pixel_format(spa: VideoFormat) -> Option<PixelFormat> {
    Some(match spa {
        VideoFormat::BGRA | VideoFormat::BGRx => PixelFormat::Bgra,
        VideoFormat::RGBA | VideoFormat::RGBx => PixelFormat::Rgba,
        VideoFormat::NV12 => PixelFormat::Nv12,
        VideoFormat::I420 => PixelFormat::I420,
        _ => return None,
    })
}

/// What the compositor finally agreed on after the `EnumFormat` exchange.
#[derive(Debug, Clone, Copy)]
pub(super) struct NegotiatedFormat {
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub spa_format: VideoFormat,
    /// Negotiated framerate. Returned by `get_framerate()` so the
    /// encoder + segmenter pace to whatever the compositor actually
    /// delivers (e.g. 24 / 30 / 60 / 144) rather than the
    /// configured-fps hint, which is only a preference.
    pub framerate: Fraction,
    /// DRM modifier when the negotiated buffer is a `DmaBuf`, `None` for shm.
    pub modifier: Option<u64>,
}

impl NegotiatedFormat {
    pub(super) fn from_video_info(info: &pw::spa::param::video::VideoInfoRaw) -> Result<Self> {
        let spa_format = info.format();
        let pixel_format = pixel_format(spa_format).ok_or_else(|| {
            FerricastError::Capture(format!(
                "PipeWire negotiated unsupported video format: {spa_format:?}"
            ))
        })?;
        let size = info.size();
        let m = info.modifier();
        Ok(Self {
            width: size.width,
            height: size.height,
            pixel_format,
            spa_format,
            framerate: info.framerate(),
            modifier: if m == 0 { None } else { Some(m) },
        })
    }

    /// True when this modifier names a layout we can read with `mmap`
    /// (no GPU detiling needed).
    pub(super) fn modifier_is_cpu_readable(&self) -> bool {
        match self.modifier {
            None => true,
            Some(m) => CPU_READABLE_MODIFIERS.contains(&m),
        }
    }
}

/// Default size / fps for the `EnumFormat` ranges. Width/height come from
/// the portal hint when available so the compositor doesn't have to
/// renegotiate for the actual monitor resolution.
pub(super) struct EnumFormatParams {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

// --------------------------------------------------------------------------
// Low-level pod helpers
// --------------------------------------------------------------------------

fn serialize(obj: Object) -> Vec<u8> {
    PodSerializer::serialize(Cursor::new(Vec::new()), &Value::Object(obj))
        .expect("SPA pod serialization is infallible for our inputs")
        .0
        .into_inner()
}

/// Wrap a serialized pod buffer into the `&Pod` view `PipeWire` wants on
/// `Stream::connect` / `update_params`. Caller must keep the backing
/// `Vec<u8>` alive.
pub(super) fn pod_view(bytes: &[u8]) -> &Pod {
    Pod::from_bytes(bytes).expect("our serializer always emits valid pods")
}

fn property_id(key: u32, id: u32) -> Property {
    Property {
        key,
        flags: PropertyFlags::empty(),
        value: Value::Id(Id(id)),
    }
}

fn property_choice_enum_id(key: u32, default: u32, alternatives: Vec<u32>) -> Property {
    Property {
        key,
        flags: PropertyFlags::empty(),
        value: Value::Choice(ChoiceValue::Id(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Enum {
                default: Id(default),
                alternatives: alternatives.into_iter().map(Id).collect(),
            },
        ))),
    }
}

fn property_choice_range_rect(
    key: u32,
    default: Rectangle,
    min: Rectangle,
    max: Rectangle,
) -> Property {
    Property {
        key,
        flags: PropertyFlags::empty(),
        value: Value::Choice(ChoiceValue::Rectangle(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Range { default, min, max },
        ))),
    }
}

fn property_choice_range_fraction(
    key: u32,
    default: Fraction,
    min: Fraction,
    max: Fraction,
) -> Property {
    Property {
        key,
        flags: PropertyFlags::empty(),
        value: Value::Choice(ChoiceValue::Fraction(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Range { default, min, max },
        ))),
    }
}

/// Build the `VideoModifier` property as a `Choice/Enum` of `Long`
/// over the supplied modifier list, with `MANDATORY | DONT_FIXATE`.
/// The first entry is also used as the default. Used in the
/// modifier-bearing pod when GPU import is available.
fn property_video_modifier(modifiers: &[u64]) -> Property {
    debug_assert!(!modifiers.is_empty());
    Property {
        key: FormatProperties::VideoModifier.as_raw(),
        flags: PropertyFlags::MANDATORY | PropertyFlags::DONT_FIXATE,
        value: Value::Choice(ChoiceValue::Long(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Enum {
                default: modifiers[0].cast_signed(),
                alternatives: modifiers.iter().map(|m| (*m).cast_signed()).collect(),
            },
        ))),
    }
}

// --------------------------------------------------------------------------
// EnumFormat pods
// --------------------------------------------------------------------------

fn enum_format_skeleton(params: &EnumFormatParams) -> Object {
    Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties: vec![
            property_id(
                FormatProperties::MediaType.as_raw(),
                MediaType::Video.as_raw(),
            ),
            property_id(
                FormatProperties::MediaSubtype.as_raw(),
                MediaSubtype::Raw.as_raw(),
            ),
            property_choice_range_rect(
                FormatProperties::VideoSize.as_raw(),
                Rectangle {
                    width: params.width.max(1),
                    height: params.height.max(1),
                },
                Rectangle {
                    width: 1,
                    height: 1,
                },
                Rectangle {
                    width: 8192,
                    height: 8192,
                },
            ),
            property_choice_range_fraction(
                FormatProperties::VideoFramerate.as_raw(),
                Fraction {
                    num: params.fps.max(1),
                    denom: 1,
                },
                Fraction { num: 0, denom: 1 },
                Fraction {
                    num: 1000,
                    denom: 1,
                },
            ),
        ],
    }
}

/// One `EnumFormat` pod that advertises a single `VideoFormat` plus
/// a list of `VideoModifier` choices. Used at connect time, once per
/// (format, modifier-list) pair the GPU exposes.
fn enum_format_with_modifier(
    format: VideoFormat,
    modifiers: &[u64],
    params: &EnumFormatParams,
) -> Vec<u8> {
    let mut obj = enum_format_skeleton(params);
    obj.properties.push(property_choice_enum_id(
        FormatProperties::VideoFormat.as_raw(),
        format.as_raw(),
        vec![format.as_raw()],
    ));
    obj.properties.push(property_video_modifier(modifiers));
    serialize(obj)
}

/// SHM fallback pod: no modifier, multiple linear formats. Compositors
/// that can't satisfy any of the modifier-bearing pods fall back to this.
fn enum_format_shm(params: &EnumFormatParams) -> Vec<u8> {
    let mut obj = enum_format_skeleton(params);
    obj.properties.push(property_choice_enum_id(
        FormatProperties::VideoFormat.as_raw(),
        SUPPORTED_FORMATS[0].as_raw(),
        SUPPORTED_FORMATS.iter().map(VideoFormat::as_raw).collect(),
    ));
    serialize(obj)
}

/// `(VideoFormat, modifiers)` pair that the GPU has agreed to import.
/// Built up by [`super::stream`] from [`crate::vulkan::VulkanImporter`]
/// at connect time and passed in here.
pub(super) struct GpuFormat {
    pub format: VideoFormat,
    /// Every modifier the importer reads back.
    pub modifiers: Vec<u64>,
    /// Those of `modifiers` the GPU consumer (NVENC, VA-API) also
    /// imports without a copy (`CaptureConfig::consumer_modifiers`).
    pub preferred: Vec<u64>,
}

/// Build the `EnumFormat` list to send at `connect()`.
///
/// * For each `GpuFormat` we emit a pod with that format pinned and
///   `VideoModifier = Choice/Enum` over the GPU-supported modifiers,
///   `MANDATORY | DONT_FIXATE`.
/// * After those, the SHM fallback pod (no modifier, multi-format
///   choice). The compositor walks the list and picks the first
///   entry it can satisfy — when `DmaBuf` works, it picks one of the
///   modifier pods; when it can't, it falls back to SHM.
///
/// `gpu_formats` empty (Vulkan unavailable / no compatible GPU
/// formats) collapses to SHM-only — equivalent to the previous
/// behaviour and a clean fallback path.
pub(super) fn initial_enum_format_list(
    params: &EnumFormatParams,
    gpu_formats: &[GpuFormat],
) -> Vec<Vec<u8>> {
    // Pods the consumer takes directly come first: the producer picks
    // the first pod it can satisfy, so it only falls back to a layout
    // the consumer would have to read back when it has no better one.
    let preferred = gpu_formats
        .iter()
        .filter(|gf| !gf.preferred.is_empty() && gf.preferred != gf.modifiers)
        .map(|gf| enum_format_with_modifier(gf.format, &gf.preferred, params));
    let readable = gpu_formats
        .iter()
        .filter(|gf| !gf.modifiers.is_empty())
        .map(|gf| enum_format_with_modifier(gf.format, &gf.modifiers, params));
    let mut out: Vec<Vec<u8>> = preferred.chain(readable).collect();
    out.push(enum_format_shm(params));
    out
}

// --------------------------------------------------------------------------
// Buffers + Meta pods (sent from `param_changed` once format is fixated)
// --------------------------------------------------------------------------

/// Preferred pool depth: the compositor's own one or two in-flight
/// buffers, plus up to three held downstream (frame channel, encoder,
/// segmenter's repeat frame), plus a spare.
const PREFERRED_BUFFERS: i32 = 6;
/// `pw_stream`'s own per-stream buffer cap.
const MAX_BUFFERS: i32 = 64;

/// `ParamBuffers` variant that requires explicit sync: DMA-BUFs plus
/// the `SPA_META_SyncTimeline` meta (and, from the producer, its two
/// syncobj data blocks). Offered before [`param_buffers_bytes`]; a
/// producer without explicit sync fails to intersect it and takes the
/// plain variant.
pub(super) fn param_buffers_explicit_sync_bytes() -> Vec<u8> {
    let sync_timeline = 1_i32 << SPA_META_SYNC_TIMELINE;
    let mut obj = buffers_object(1_i32 << SpaDataType::DmaBuf.as_raw());
    obj.properties.push(Property {
        key: SPA_PARAM_BUFFERS_METATYPE,
        flags: PropertyFlags::MANDATORY,
        value: Value::Choice(ChoiceValue::Int(Choice(
            ChoiceFlags::empty(),
            ChoiceEnum::Flags {
                default: sync_timeline,
                flags: vec![sync_timeline],
            },
        ))),
    });
    serialize(obj)
}

/// `ParamBuffers` pod: tells `PipeWire` what shapes of buffers we accept.
///
/// The bitmask covers shm (`MemFd` / `MemPtr`) and `DmaBuf` so the source can
/// pick whichever it can produce cheaply. We deliberately don't lock
/// down `BUFFERS_size` / `_stride` — letting the source pick those is
/// what wlx-capture does and is what Mutter / `KWin` expect;
/// over-specifying them sometimes leaves the source with no satisfying
/// answer and it never starts producing frames.
///
/// `BUFFERS_buffers` only states a preference: GPU frames keep their
/// buffer dequeued while downstream reads them (see `lease`), so a
/// deeper pool means fewer hold-limit readbacks. The range is wide
/// enough to intersect anything a compositor offers.
pub(super) fn param_buffers_bytes() -> Vec<u8> {
    let data_types = (1_i32 << SpaDataType::MemFd.as_raw())
        | (1_i32 << SpaDataType::MemPtr.as_raw())
        | (1_i32 << SpaDataType::DmaBuf.as_raw());
    serialize(buffers_object(data_types))
}

/// The `ParamBuffers` object for buffers of `data_types` (a bitmask of
/// `SPA_DATA_*`).
fn buffers_object(data_types: i32) -> Object {
    Object {
        type_: SpaTypes::ObjectParamBuffers.as_raw(),
        id: ParamType::Buffers.as_raw(),
        properties: vec![
            Property {
                key: SPA_PARAM_BUFFERS_buffers,
                flags: PropertyFlags::empty(),
                value: Value::Choice(ChoiceValue::Int(Choice(
                    ChoiceFlags::empty(),
                    ChoiceEnum::Range {
                        default: PREFERRED_BUFFERS,
                        min: 1,
                        max: MAX_BUFFERS,
                    },
                ))),
            },
            Property {
                key: SPA_PARAM_BUFFERS_dataType,
                flags: PropertyFlags::empty(),
                value: Value::Int(data_types),
            },
        ],
    }
}

/// `ParamMeta(SyncTimeline)` pod: asks an explicit-sync producer for
/// acquire / release timeline points on every buffer (see
/// [`super::sync`]).
pub(super) fn param_meta_sync_timeline_bytes() -> Vec<u8> {
    let obj = Object {
        type_: SpaTypes::ObjectParamMeta.as_raw(),
        id: ParamType::Meta.as_raw(),
        properties: vec![
            Property {
                key: SPA_PARAM_META_type,
                flags: PropertyFlags::empty(),
                value: Value::Id(Id(SPA_META_SYNC_TIMELINE)),
            },
            Property {
                key: SPA_PARAM_META_size,
                flags: PropertyFlags::empty(),
                value: Value::Int(
                    i32::try_from(size_of::<MetaSyncTimeline>())
                        .expect("invariant: spa_meta_sync_timeline is 24 bytes"),
                ),
            },
        ],
    };
    serialize(obj)
}

/// `ParamMeta(Header)` pod: asks `PipeWire` to attach a `spa_meta_header`
/// (PTS / sequence / corruption flag) to every buffer.
pub(super) fn param_meta_header_bytes() -> Vec<u8> {
    let obj = Object {
        type_: SpaTypes::ObjectParamMeta.as_raw(),
        id: ParamType::Meta.as_raw(),
        properties: vec![
            Property {
                key: SPA_PARAM_META_type,
                flags: PropertyFlags::empty(),
                value: Value::Id(Id(SPA_META_Header)),
            },
            Property {
                key: SPA_PARAM_META_size,
                flags: PropertyFlags::empty(),
                value: Value::Int(
                    i32::try_from(std::mem::size_of::<pw::spa::sys::spa_meta_header>())
                        .expect("invariant: spa_meta_header is a few bytes"),
                ),
            },
        ],
    };
    serialize(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> EnumFormatParams {
        EnumFormatParams {
            width: 1920,
            height: 1080,
            fps: 60,
        }
    }

    #[test]
    fn consumer_modifiers_are_offered_first() {
        let formats = [
            GpuFormat {
                format: VideoFormat::BGRx,
                modifiers: vec![1, 2, 0],
                preferred: vec![2],
            },
            GpuFormat {
                format: VideoFormat::RGBx,
                modifiers: vec![1, 0],
                preferred: vec![1, 0],
            },
        ];
        let pods = initial_enum_format_list(&params(), &formats);
        let expected = [
            enum_format_with_modifier(VideoFormat::BGRx, &[2], &params()),
            enum_format_with_modifier(VideoFormat::BGRx, &[1, 2, 0], &params()),
            enum_format_with_modifier(VideoFormat::RGBx, &[1, 0], &params()),
            enum_format_shm(&params()),
        ];
        assert_eq!(pods, expected);
    }

    #[test]
    fn explicit_sync_buffers_param_differs_from_the_plain_one() {
        let explicit = param_buffers_explicit_sync_bytes();
        let plain = param_buffers_bytes();
        assert!(Pod::from_bytes(&explicit).is_some());
        assert_ne!(explicit, plain);
        assert!(Pod::from_bytes(&param_meta_sync_timeline_bytes()).is_some());
    }
}
