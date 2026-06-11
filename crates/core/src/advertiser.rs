//! Receiver-side advertisement — make the local process discoverable
//! on the LAN as a sink for a given protocol.
//!
//! Mirror of [`crate::discovery`]: where `Discovery` browses for
//! receivers a sender can stream to, `Advertiser` publishes the local
//! process so external senders (a phone running YouTube, a laptop
//! running an HLS player) can find it. Concretely a Chromecast
//! receiver impl advertises `_googlecast._tcp` with the right TXT
//! records; an AirPlay sink impl advertises `_airplay._tcp` and
//! `_raop._tcp`; etc.
//!
//! Concrete impls live in the per-protocol crates so the heavy mDNS
//! / Zeroconf deps stay out of `ferricast-core`.

use std::collections::HashMap;

use bytes::Bytes;

use crate::device::DeviceCapabilities;
use crate::error::Result;

/// What to publish about the local receiver. Fields map onto the
/// union of what real protocols carry in their service records:
/// Chromecast TXT (`md`, `fn`, `ic`, `ca`), AirPlay (`deviceid`,
/// `features`, `model`), etc. Protocol-specific extras go in
/// [`AdvertiseInfo::txt`].
#[derive(Debug, Clone)]
pub struct AdvertiseInfo {
    /// Human-visible name that shows up in the sender's picker
    /// ("Living Room", "Sergio's Laptop"). Required by every known
    /// receiver protocol.
    pub friendly_name: String,
    /// TCP port the receiver's control server is listening on.
    /// Advertiser publishes this verbatim; binding the socket is
    /// the receiver impl's job.
    pub port: u16,
    /// Decoder caps to surface so a sender can pre-filter codec /
    /// profile choices before establishing the session.
    pub capabilities: DeviceCapabilities,
    /// Stable per-device identifier embedded in the service record.
    /// Chromecast uses a hex UUID in `id=`; AirPlay uses MAC-style
    /// `deviceid=`. Caller controls the format; advertiser just
    /// passes it through.
    pub device_id: String,
    /// Optional PNG/JPEG icon some protocols expose
    /// (Chromecast `ic=`). Empty = no icon advertised.
    pub icon: Bytes,
    /// Protocol-specific TXT record entries. Merged on top of
    /// whatever the impl computes from the typed fields above.
    pub txt: HashMap<String, String>,
}

/// Publish the local process as a receiver. One `Advertiser` per
/// protocol, owned by the receiver runtime; `start` is idempotent
/// from the caller's perspective but impls MAY return an error if
/// called twice without `stop` in between.
pub trait Advertiser: Send + Sync {
    const PROTOCOL: &'static str;

    fn start(&mut self, info: AdvertiseInfo) -> impl Future<Output = Result<()>> + Send;
    fn stop(&mut self) -> impl Future<Output = Result<()>> + Send;
    fn is_running(&self) -> bool;
}

/// Marker for advertisers that publish over multicast DNS.
/// Mirror of [`crate::discovery::MdnsDiscovery`].
pub trait MdnsAdvertiser: Advertiser {
    const SERVICE_TYPE: &'static str;
}
