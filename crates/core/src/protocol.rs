use std::borrow::Cow;

use crate::advertiser::{AdvertiseInfo, Advertiser};
use crate::codec::Codec;
use crate::control::ControlSession;
use crate::device::Device;
use crate::discovery::Discovery;
use crate::error::Result;
use crate::puller::MediaPuller;
use crate::session::CastSession;

/// Whether a protocol handler can run on this host right now.
///
/// Handlers that never need platform resources (anything over plain
/// TCP/UDP + mDNS) return [`Available`](Self::Available) unconditionally.
/// Handlers that reach into platform-specific stacks (Wi-Fi Direct,
/// WinRT casting, kernel-level iftypes) use [`Degraded`](Self::Degraded)
/// when they have a working fallback but a feature-limited setup, and
/// [`Unavailable`](Self::Unavailable) when nothing will work.
///
/// Callers (the GUI, the stream manager) surface `reason` to the user
/// verbatim — it is the single source of truth for "why can't I cast
/// to a Miracast TV from this laptop?".
#[derive(Debug, Clone)]
pub enum ProtocolAvailability {
    /// Every discovery + session path is usable.
    Available,
    /// The protocol works but a secondary path is blocked
    /// (e.g. Miracast over LAN works, but the Wi-Fi radio has no P2P
    /// support so no direct-to-TV path is available).
    Degraded {
        /// One-line human-readable explanation.
        reason: Cow<'static, str>,
    },
    /// The protocol cannot run at all here.
    Unavailable { reason: Cow<'static, str> },
}

impl ProtocolAvailability {
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        !matches!(self, Self::Unavailable { .. })
    }
}

/// Whether a protocol can talk to a specific device.
///
/// Returned by [`ProtocolHandler::device_compatibility`] so the GUI can
/// disable "Cast" on a device the local host can't reach even though
/// discovery surfaced it (e.g. a P2P-only Miracast TV from a laptop
/// without a P2P radio, or a Cast Streaming–only receiver when the
/// local build lacks the Opus encoder).
#[derive(Debug, Clone)]
pub enum DeviceCompatibility {
    Compatible,
    Incompatible { reason: Cow<'static, str> },
}

impl DeviceCompatibility {
    #[must_use]
    pub const fn is_compatible(&self) -> bool {
        matches!(self, Self::Compatible)
    }
}

pub trait ProtocolHandler: Send + Sync {
    const PROTOCOL: &'static str;
    const SUPPORTED_CODECS: &'static [Codec];

    type Discovery: Discovery;
    type Session: CastSession;

    fn create_discovery(&self) -> Self::Discovery;
    fn create_session(&self) -> Result<Self::Session>;

    /// Create a session tailored to a specific device. Default ignores
    /// the device and delegates to [`create_session`]; protocol impls
    /// that support multiple session types (e.g. HLS vs. Cast Streaming
    /// on Chromecast) override this to pick based on
    /// `device.capabilities`.
    fn create_session_for_device(&self, _device: &Device) -> Result<Self::Session> {
        self.create_session()
    }

    /// Whether this handler can run on the current host. Called by the
    /// GUI at registration time to grey out or label unusable
    /// protocols. Default returns [`ProtocolAvailability::Available`],
    /// which is the right answer for anything over plain networking
    /// (Chromecast, AirPlay, DIAL, Miracast-over-LAN).
    ///
    /// Protocols that depend on platform-specific infrastructure
    /// (Wi-Fi Direct hardware + wpa_supplicant on Linux, Miracast WDI
    /// driver on Windows) override this to surface a concrete reason
    /// when the local host can't reach the fast path.
    fn availability(&self) -> ProtocolAvailability {
        ProtocolAvailability::Available
    }

    /// Whether this handler can talk to one particular device.
    ///
    /// Default returns [`DeviceCompatibility::Compatible`]; impls
    /// override to reject devices they can't reach — e.g. Miracast
    /// rejects a P2P-only peer when the local radio can't do P2P,
    /// Chromecast rejects a Cast Streaming–only receiver when the
    /// local build lacks the Opus encoder.
    fn device_compatibility(&self, _device: &Device) -> DeviceCompatibility {
        DeviceCompatibility::Compatible
    }
}

/// Receiver-side protocol — advertise the local process as a sink,
/// accept a control connection from a remote sender, and translate
/// its `MediaCommand::Load { url }` into an internal media pull.
///
/// Chromecast is the canonical impl: advertiser publishes
/// `_googlecast._tcp`, control session speaks CASTV2 over TLS, and
/// the puller is the HLS one (Cast LOAD always hands us an HTTP URL).
/// AirPlay would fit the same shape with `_airplay._tcp` + AirPlay
/// HTTP + the same HLS puller.
///
/// Pure pull protocols (a user pastes an HLS URL into the GUI) don't
/// go through this trait — they invoke a [`MediaPuller`] directly
/// without any advertise/control step.
pub trait ReceiverProtocol: Send + Sync {
    const PROTOCOL: &'static str;
    const SUPPORTED_CODECS: &'static [Codec];

    type Advertiser: Advertiser;
    type Control: ControlSession;
    type Puller: MediaPuller;

    fn create_advertiser(&self) -> Self::Advertiser;
    fn create_control(&self) -> Result<Self::Control>;
    fn create_puller(&self) -> Result<Self::Puller>;

    /// What to publish about this receiver. Called once at
    /// registration; impls bake in service-type-specific TXT keys
    /// (`md`, `ca`, `ic` for Cast; `features`, `deviceid` for
    /// AirPlay) on top of the generic fields.
    fn advertise_info(&self) -> AdvertiseInfo;

    /// Mirror of [`ProtocolHandler::availability`] for the receiver
    /// path. A sink that needs e.g. MSIX packaging to accept Miracast
    /// returns [`ProtocolAvailability::Unavailable`] on hosts where
    /// the packaging isn't in place.
    fn availability(&self) -> ProtocolAvailability {
        ProtocolAvailability::Available
    }
}
