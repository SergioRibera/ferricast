//! Apple platform decoders: VideoToolbox (`VTDecompressionSession`) for
//! H.264 / H.265 / VP9 / AV1 and AudioToolbox (`AudioConverter`) for
//! AAC (LC, HE-AAC v1 / v2), AAC-ELD and ALAC.

#[cfg(feature = "audiotoolbox")]
mod audio;
#[cfg(feature = "videotoolbox")]
mod video;

#[cfg(feature = "audiotoolbox")]
pub(crate) use audio::audio;
#[cfg(feature = "videotoolbox")]
pub(crate) use video::video;

/// An `OSStatus` as Apple tools print it: the number, plus its
/// four-character code when it spells one (`'!dat'`, `'fmt?'`).
pub(super) fn os_status(code: i32) -> String {
    let bytes = code.to_be_bytes();
    if bytes.iter().all(u8::is_ascii_graphic) {
        let fourcc: String = bytes.iter().map(|&byte| char::from(byte)).collect();
        format!("OSStatus {code} ('{fourcc}')")
    } else {
        format!("OSStatus {code}")
    }
}
