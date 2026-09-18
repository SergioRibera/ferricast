use crate::device::{Features, Flags};

/// What the protocol needs from the user before streaming can begin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingChallenge {
    /// Display a numeric PIN input. `digits` is always 4 for AirPlay.
    Pin {
        digits: u8,
    },
    None,
    Credential,
}

impl PairingChallenge {
    pub fn new_airplay(flags: Flags, features: &Features) -> Self {
        if flags.contains(Flags::PASSWORD_REQUIRED) && features.prefers_legacy_pairing() {
            return Self::Credential;
        }

        if flags.contains(Flags::PIN_ONE_TIME_PAIRING) {
            return Self::Pin { digits: 4 };
        }

        return Self::None;
    }
}

/// User's answer to a [`PairingChallenge`].
#[derive(Debug, Clone)]
pub enum PairingResponse {
    Pin(String),
    Confirmed,
    Cancelled,
}
