use crate::device::{AirPlayPairing, AirplayConfig};

/// What the protocol needs from the user before streaming can begin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingChallenge {
    /// Display a numeric PIN input. `digits` is always 4 for AirPlay.
    Pin {
        digits: u8,
    },
    None,
    /// Ask the user for the receiver's access password. `rejected` is set
    /// when the receiver refused the previous answer, so the prompt can
    /// say so.
    Credential {
        rejected: bool,
    },
}

impl PairingChallenge {
    /// Challenge for a first-time AirPlay connection (no stored credentials).
    ///
    /// HomeKit pairing and PIN-protected legacy receivers show a 4-digit PIN;
    /// legacy and transient pairing need no user input.
    #[must_use]
    pub fn new_airplay(config: &AirplayConfig) -> Self {
        match config.pairing_mode() {
            AirPlayPairing::Hap | AirPlayPairing::LegacyPin => Self::Pin { digits: 4 },
            AirPlayPairing::Legacy | AirPlayPairing::HapTransient => Self::None,
        }
    }
}

/// User's answer to a [`PairingChallenge`].
#[derive(Debug, Clone)]
pub enum PairingResponse {
    Pin(String),
    /// Answer to [`PairingChallenge::Credential`].
    Password(String),
    Confirmed,
    Cancelled,
}
