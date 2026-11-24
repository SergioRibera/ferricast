//! The Apple Lossless `ALACSpecificConfig` ("magic cookie") shared by
//! the ALAC decoders.
//!
//! `AirPlay` never sends the cookie in-band: senders use the fixed
//! configuration of the RAOP `fmtp` line (352 samples per frame, 16-bit,
//! `pb = 40`, `mb = 10`, `kb = 14`), so [`AlacConfig::airplay`] builds it
//! from the sample rate and channel count. A container that carries the
//! cookie (MP4 `alac` box) passes it as the decoder's `extra_data`,
//! read with [`AlacConfig::parse`].

/// Samples per channel in one `AirPlay` ALAC frame.
pub const AIRPLAY_FRAME_SAMPLES: u32 = 352;

/// The fields of an `ALACSpecificConfig`, in cookie order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlacConfig {
    /// Samples per channel in a frame.
    pub frame_length: u32,
    /// Bits per sample.
    pub bit_depth: u8,
    /// Rice history multiplier.
    pub pb: u8,
    /// Rice initial history.
    pub mb: u8,
    /// Rice parameter limit.
    pub kb: u8,
    /// Channels per frame.
    pub channels: u8,
    /// Longest run of zeros the coder signals.
    pub max_run: u16,
    /// Largest frame in bytes, 0 when unknown.
    pub max_frame_bytes: u32,
    /// Average bit rate in bits per second, 0 when unknown.
    pub avg_bit_rate: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AlacConfig {
    /// Length of the cookie.
    pub const LEN: usize = 24;

    /// The configuration `AirPlay` senders assume for `sample_rate` and
    /// `channels`.
    #[must_use]
    pub fn airplay(sample_rate: u32, channels: u8) -> Self {
        Self {
            frame_length: AIRPLAY_FRAME_SAMPLES,
            bit_depth: 16,
            pb: 40,
            mb: 10,
            kb: 14,
            channels,
            max_run: 255,
            // Unknown, which decoders accept.
            max_frame_bytes: 0,
            avg_bit_rate: 0,
            sample_rate,
        }
    }

    /// Reads a cookie: the 24-byte `ALACSpecificConfig`, optionally
    /// behind the `frma` / `alac` atom headers some files keep in front
    /// of it. `None` when no configuration of that length is found.
    #[must_use]
    pub fn parse(cookie: &[u8]) -> Option<Self> {
        let body = strip_atoms(cookie)?;
        let bytes: &[u8; Self::LEN] = body.get(..Self::LEN)?.try_into().ok()?;
        let u32_at = |at: usize| {
            u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        Some(Self {
            frame_length: u32_at(0),
            // bytes[4]: compatible version, 0.
            bit_depth: bytes[5],
            pb: bytes[6],
            mb: bytes[7],
            kb: bytes[8],
            channels: bytes[9],
            max_run: u16::from_be_bytes([bytes[10], bytes[11]]),
            max_frame_bytes: u32_at(12),
            avg_bit_rate: u32_at(16),
            sample_rate: u32_at(20),
        })
    }

    /// The 24-byte cookie.
    #[must_use]
    pub fn cookie(self) -> [u8; Self::LEN] {
        let mut cookie = [0_u8; Self::LEN];
        cookie[..4].copy_from_slice(&self.frame_length.to_be_bytes());
        cookie[4..10].copy_from_slice(&[
            0,
            self.bit_depth,
            self.pb,
            self.mb,
            self.kb,
            self.channels,
        ]);
        cookie[10..12].copy_from_slice(&self.max_run.to_be_bytes());
        cookie[12..16].copy_from_slice(&self.max_frame_bytes.to_be_bytes());
        cookie[16..20].copy_from_slice(&self.avg_bit_rate.to_be_bytes());
        cookie[20..24].copy_from_slice(&self.sample_rate.to_be_bytes());
        cookie
    }
}

/// The cookie body behind optional `frma` and `alac` atom headers
/// (`size`, four-character type, and for `alac` a 4-byte version/flags).
fn strip_atoms(cookie: &[u8]) -> Option<&[u8]> {
    let mut rest = cookie;
    if rest.get(4..8) == Some(b"frma") {
        rest = rest.get(12..)?;
    }
    if rest.get(4..8) == Some(b"alac") {
        rest = rest.get(12..)?;
    }
    Some(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_airplay_cookie_round_trips() {
        let config = AlacConfig::airplay(44_100, 2);
        let cookie = config.cookie();
        assert_eq!(&cookie[..4], &352_u32.to_be_bytes());
        assert_eq!(&cookie[4..12], &[0, 16, 40, 10, 14, 2, 0, 255]);
        assert_eq!(&cookie[20..], &44_100_u32.to_be_bytes());
        assert_eq!(AlacConfig::parse(&cookie), Some(config));
    }

    #[test]
    fn atom_headers_in_front_of_the_cookie_are_skipped() {
        let config = AlacConfig::airplay(48_000, 1);
        let mut wrapped = Vec::new();
        wrapped.extend_from_slice(&12_u32.to_be_bytes());
        wrapped.extend_from_slice(b"frmaalac");
        wrapped.extend_from_slice(&36_u32.to_be_bytes());
        wrapped.extend_from_slice(b"alac");
        wrapped.extend_from_slice(&[0; 4]);
        wrapped.extend_from_slice(&config.cookie());
        assert_eq!(AlacConfig::parse(&wrapped), Some(config));
        assert_eq!(AlacConfig::parse(&[0; 10]), None);
    }
}
