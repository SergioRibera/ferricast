//! AirPlay pairing procedures.
//!
//! * [`legacy`]: `/pair-setup` (raw Ed25519 key) + `/pair-verify`, spoken by
//!   UxPlay, RPiPlay and other shairplay derivatives.
//! * [`legacy_pin`]: `/pair-pin-start` + SRP over `/pair-setup-pin`, used by
//!   those receivers when started with a PIN.
//! * [`hap`]: HomeKit pair-setup / pair-verify and transient pairing, used by
//!   Apple TVs, HomePods and AirPlay 2 TVs.

pub(crate) mod hap;
pub(crate) mod legacy;
pub(crate) mod legacy_pin;

/// Shared secrets produced by a completed pairing.
#[derive(Clone, Default)]
pub(crate) struct PairedSession {
    /// X25519 secret from legacy pair-verify. Receivers hash the `SETUP`
    /// AES key with it.
    pub ecdh_secret: Option<[u8; 32]>,
    /// HomeKit session secret; the control connection is encrypted with
    /// keys derived from it.
    pub hap_secret: Option<Vec<u8>>,
}

/// `[0; 4] ‖ label` nonce used by HomeKit ChaCha20-Poly1305 messages.
pub(crate) fn hap_nonce(label: [u8; 8]) -> chacha20poly1305::Nonce {
    let mut nonce = [0_u8; 12];
    nonce[4..].copy_from_slice(&label);
    nonce.into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn nonce_is_twelve_bytes_with_zero_prefix() {
        let n = super::hap_nonce(*b"PV-Msg03");
        assert_eq!(&n[..4], &[0, 0, 0, 0]);
        assert_eq!(&n[4..], b"PV-Msg03");
    }
}
