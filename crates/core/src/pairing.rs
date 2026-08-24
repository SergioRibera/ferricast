/// What the protocol needs from the user before streaming can begin.
#[derive(Debug, Clone)]
pub enum PairingChallenge {
    /// Display a numeric PIN input. `digits` is always 4 for AirPlay.
    Pin { digits: u8 },
    /// Ask the user to confirm an action on the target device (press OK, etc.).
    Confirmation,
}

/// User's answer to a [`PairingChallenge`].
#[derive(Debug, Clone)]
pub enum PairingResponse {
    Pin(String),
    Confirmed,
    Cancelled,
}
