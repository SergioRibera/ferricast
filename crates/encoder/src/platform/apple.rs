//! Apple platform encoders: `VideoToolbox` (`VTCompressionSession`) for
//! H.264 / H.265 and `AudioToolbox` (`AudioConverter`) for AAC-LC and
//! AAC-ELD.

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

#[cfg(test)]
mod tests {
    use super::os_status;

    #[test]
    fn statuses_show_their_four_character_code() {
        assert_eq!(os_status(-12_908), "OSStatus -12908");
        assert_eq!(
            os_status(i32::from_be_bytes(*b"fmt?")),
            "OSStatus 1718449215 ('fmt?')"
        );
    }
}
