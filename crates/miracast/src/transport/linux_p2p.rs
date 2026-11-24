//! Wi-Fi Direct Miracast transport via `wpa_supplicant` on D-Bus.
//!
//! Drives the `fi.w1.wpa_supplicant1` system-bus service: picks a
//! P2P-Device-capable interface, uploads the WFD IE bytes so peers
//! can see us as a Miracast source/sink, runs `P2PDevice.Find` and
//! surfaces each peer whose `IEs` property carries the Wi-Fi Alliance
//! WFD subtype (OUI `50:6F:9A`, OUI type `0x0A`).
//!
//! Once a user picks a peer, [`LinuxP2pConnector::connect`] calls
//! `P2PDevice.Connect`, waits for the `GroupStarted` signal, and
//! returns the virtual group interface name + the GO's IP so the
//! existing WFD RTSP source/sink code can plug in.
//!
//! Compiled only on Linux with the `linux-p2p` feature.

use std::collections::HashMap;
use std::time::Duration;

use futures_util::StreamExt;
use tokio::sync::mpsc;
use zbus::zvariant::{ObjectPath, OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, Proxy};

use super::TransportError;
use crate::ie::{DeviceRole, SubelementId, WFD_OUI_TYPE, WIFI_ALLIANCE_OUI, WfdIe};

const SUPPLICANT_BUS: &str = "fi.w1.wpa_supplicant1";
const SUPPLICANT_ROOT_PATH: &str = "/fi/w1/wpa_supplicant1";
const ROOT_IFACE: &str = "fi.w1.wpa_supplicant1";
const P2P_IFACE: &str = "fi.w1.wpa_supplicant1.Interface.P2PDevice";
const PEER_IFACE: &str = "fi.w1.wpa_supplicant1.Peer";

/// A peer discovered on the Wi-Fi Direct bus.
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    /// supplicant `Peer` object path, unique per session.
    pub path: String,
    /// Human-readable device name (what the user sees in the picker).
    pub name: String,
    /// Peer's MAC — stable across sessions, so used as a durable id.
    pub address: [u8; 6],
    /// Parsed WFD Information Element; `None` for a plain P2P peer
    /// that happens to be in range (not a Miracast device).
    pub wfd: Option<WfdIe>,
}

impl DiscoveredPeer {
    /// Whether the peer advertises itself as a Miracast sink, source,
    /// or dual-role device.
    #[must_use]
    pub fn is_miracast(&self) -> bool {
        self.wfd
            .as_ref()
            .is_some_and(|w| w.device_information.is_some())
    }

    /// Whether we (a source) should offer to cast to it.
    #[must_use]
    pub fn is_sink(&self) -> bool {
        self.wfd
            .as_ref()
            .and_then(|w| w.device_information)
            .map(|info| {
                matches!(
                    info.role,
                    DeviceRole::PrimarySink | DeviceRole::SecondarySink | DeviceRole::DualRole
                )
            })
            .unwrap_or(false)
    }
}

/// High-level client for the Wi-Fi Direct Miracast subsystem.
pub struct LinuxP2pClient {
    conn: Connection,
    interface_path: OwnedObjectPath,
}

impl LinuxP2pClient {
    /// Connect to the system wpa_supplicant and pick the first
    /// P2P-Device-capable interface. The WFD IE bytes you pass are
    /// written to the `WFDIEs` property so peers see the Miracast
    /// subtype in our probe/beacon frames.
    pub async fn connect(wfd_ie: &WfdIe) -> Result<Self, TransportError> {
        let conn = Connection::system()
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("system bus: {e}")))?;

        let interface_path = resolve_or_create_interface(&conn).await?;

        // Push the WFD subelements globally — supplicant wraps them in
        // the vendor-specific IE header on the air itself.
        let bytes = wfd_ie.to_subelements();
        let value = Value::from(bytes.to_vec());
        let root = Proxy::new(&conn, SUPPLICANT_BUS, SUPPLICANT_ROOT_PATH, ROOT_IFACE)
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("root proxy: {e}")))?;
        root.set_property("WFDIEs", value)
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("set WFDIEs: {e}")))?;

        tracing::info!(iface = %interface_path.as_str(), "miracast linux p2p: ready");
        Ok(Self {
            conn,
            interface_path,
        })
    }

    /// Start continuous P2P discovery. Each discovered peer is parsed
    /// for its WFD IE and pushed on the returned channel.
    pub async fn start_discovery(&self) -> Result<mpsc::Receiver<DiscoveredPeer>, TransportError> {
        let (tx, rx) = mpsc::channel(16);

        let p2p = self.p2p_proxy().await?;
        let empty: HashMap<&str, Value<'_>> = HashMap::new();
        p2p.call_method("Find", &(empty,))
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("Find: {e}")))?;

        let listen_result = p2p
            .call_method("ExtendedListen", &(HashMap::<&str, Value>::new(),))
            .await;
        if let Err(e) = listen_result {
            tracing::debug!(error = %e, "ExtendedListen rejected (optional)");
        }

        let conn = self.conn.clone();
        let interface_path = self.interface_path.clone();
        tokio::spawn(async move {
            let p2p_owned =
                match Proxy::new(&conn, SUPPLICANT_BUS, interface_path.clone(), P2P_IFACE).await {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error = %e, "linux p2p: proxy for signal stream");
                        return;
                    }
                };
            let mut stream = match p2p_owned.receive_signal("DeviceFound").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "linux p2p: subscribe DeviceFound");
                    return;
                }
            };
            while let Some(msg) = stream.next().await {
                let path: OwnedObjectPath = match msg.body().deserialize() {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let path_ref = path.as_ref();
                match read_peer(&conn, &path_ref).await {
                    Ok(peer) => {
                        if peer.is_miracast() && tx.send(peer).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => tracing::debug!(error = %e, "linux p2p: peer read"),
                }
            }
        });

        Ok(rx)
    }

    /// Advertise the local host as a Wi-Fi Direct *Miracast sink*:
    /// set the P2P device name + Primary Device Type to Display/TV so
    /// phones' Smart View / Cast UIs classify us correctly, enter
    /// continuous `Listen` mode, and auto-accept inbound
    /// `GONegotiationRequest`s via the configured [`PairingMode`].
    /// Every successful `GroupStarted` is pushed on the returned
    /// channel so the caller can bind RTSP on the new virtual
    /// interface.
    pub async fn listen_as_sink(
        &self,
        device_name: &str,
        pairing: &PairingMode,
    ) -> Result<mpsc::Receiver<GroupStarted>, TransportError> {
        // Primary Device Type: Category 0x0005 Display, OUI 50:6F:9A
        // (Wi-Fi Alliance), Subcategory 0x0001 Television.
        const PRIMARY_DEVICE_TYPE_DISPLAY_TV: [u8; 8] =
            [0x00, 0x05, 0x50, 0x6F, 0x9A, 0x02, 0x00, 0x01];

        let p2p = self.p2p_proxy().await?;
        // Push P2PDeviceConfig (DeviceName + PrimaryDeviceType).
        // supplicant merges with the existing config so this is
        // non-destructive for other keys (`GOIntent`, `SsidPostfix`,
        // …). `ConfigMethod` is NOT a valid P2PDeviceConfig key —
        // its old presence silently rejected the whole dict and left
        // the device unnamed in Android's picker. Advertised WPS
        // methods come from the global `config_methods` supplicant
        // option (default = all); we honour the caller's choice at
        // GON-accept time instead.
        let name_val = Value::from(device_name.to_string());
        let pdt_val = Value::from(PRIMARY_DEVICE_TYPE_DISPLAY_TV.to_vec());
        let mut cfg: HashMap<&str, Value> = HashMap::new();
        cfg.insert("DeviceName", name_val);
        cfg.insert("PrimaryDeviceType", pdt_val);
        if let Err(e) = p2p.set_property("P2PDeviceConfig", Value::from(cfg)).await {
            tracing::error!(error = %e, "miracast sink: P2PDeviceConfig set FAILED — peers will see us as unnamed");
        } else {
            tracing::info!(name = %device_name, "miracast sink: P2PDeviceConfig set");
        }

        // `Listen(timeout)` with 0 = until explicitly stopped.
        p2p.call_method("Listen", &(0i32,))
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("Listen: {e}")))?;

        let (tx, rx) = mpsc::channel(4);
        let conn = self.conn.clone();
        let interface_path = self.interface_path.clone();
        let pairing = pairing.clone();
        tokio::spawn(async move {
            let p2p_owned =
                match Proxy::new(&conn, SUPPLICANT_BUS, interface_path.clone(), P2P_IFACE).await {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error = %e, "sink: p2p proxy");
                        return;
                    }
                };
            let mut gon = match p2p_owned.receive_signal("GONegotiationRequest").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe GONegotiationRequest");
                    return;
                }
            };
            let mut started = match p2p_owned.receive_signal("GroupStarted").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe GroupStarted");
                    return;
                }
            };
            // supplicant fires ProvisionDiscoveryRequestDisplayPin with
            // a PIN *it* generated as soon as the phone picks us from
            // its list and asks "show me a PIN I can type". That PIN
            // is what the phone screen's keypad expects — if we don't
            // display it the user sees the entry field but has nothing
            // to type.
            let mut prov_pin =
                match p2p_owned.receive_signal("ProvisionDiscoveryRequestDisplayPin").await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(error = %e, "sink: subscribe ProvDisc display-pin");
                        return;
                    }
                };
            let mut prov_pbc = match p2p_owned.receive_signal("ProvisionDiscoveryPBCRequest").await
            {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe ProvDisc PBC");
                    return;
                }
            };
            let mut prov_fail = match p2p_owned.receive_signal("ProvisionDiscoveryFailure").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe ProvDisc failure");
                    return;
                }
            };
            let mut group_fail = match p2p_owned.receive_signal("GroupFormationFailure").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe GroupFormationFailure");
                    return;
                }
            };
            let mut wps_fail = match p2p_owned.receive_signal("WpsFailed").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe WpsFailed");
                    return;
                }
            };
            let mut gon_fail = match p2p_owned.receive_signal("GONegotiationFailure").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe GONegotiationFailure");
                    return;
                }
            };
            // Last PIN supplicant handed us for the pending peer — used
            // if the subsequent GONegotiationRequest asks for the same
            // method so we feed back the PIN the phone is typing.
            let mut last_prov_pin: Option<(OwnedObjectPath, String)> = None;
            // De-dup: same (peer, pin) within this many ms is logged
            // once. If supplicant regenerates a different PIN for the
            // same peer we announce it as "UPDATED".
            let mut last_prov_at: Option<std::time::Instant> = None;
            // De-dup Connect calls: supplicant sometimes emits GON
            // signal twice (once per ProvDisc completion, once per
            // peer re-request). Calling Connect twice on the same peer
            // aborts the first attempt. Track peers we've already
            // responded to within a short window.
            let mut last_connect_at: Option<(OwnedObjectPath, std::time::Instant)> = None;
            let mut gon_ok = match p2p_owned.receive_signal("GONegotiationSuccess").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe GONegotiationSuccess");
                    return;
                }
            };
            loop {
                tokio::select! {
                    msg = prov_pin.next() => {
                        let Some(sig) = msg else { break };
                        // `ProvisionDiscoveryRequestDisplayPin(peer: o, pin: s)`.
                        let Ok((peer, pin)): Result<(OwnedObjectPath, String), _>
                            = sig.body().deserialize() else { continue };
                        // supplicant sometimes retries ProvDisc and emits
                        // a fresh PIN each time. Dedup same (peer,pin)
                        // within 2s; a *different* PIN logs as UPDATED.
                        let now = std::time::Instant::now();
                        let same = last_prov_pin
                            .as_ref()
                            .is_some_and(|(p, pp)| p == &peer && pp == &pin);
                        let too_soon = last_prov_at
                            .is_some_and(|t| now.duration_since(t) < std::time::Duration::from_millis(2000));
                        if same && too_soon {
                            continue;
                        }
                        let updated = last_prov_pin
                            .as_ref()
                            .is_some_and(|(p, pp)| p == &peer && pp != &pin);
                        if updated {
                            tracing::warn!(
                                peer = %peer.as_str(),
                                pin = %pin,
                                "miracast sink: PIN UPDATED, type on phone → **{}** ← (ignore any earlier PIN)",
                                pin,
                            );
                        } else {
                            tracing::warn!(
                                peer = %peer.as_str(),
                                pin = %pin,
                                "miracast sink: PIN to type on phone → **{}** ←",
                                pin,
                            );
                        }
                        last_prov_pin = Some((peer, pin));
                        last_prov_at = Some(now);
                    }
                    msg = prov_pbc.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(peer) = sig.body().deserialize::<OwnedObjectPath>() {
                            tracing::info!(peer = %peer.as_str(), "sink: ProvDisc PBC request");
                        }
                    }
                    msg = prov_fail.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok((peer, status)) = sig.body().deserialize::<(OwnedObjectPath, i32)>() {
                            tracing::warn!(peer = %peer.as_str(), status, "sink: ProvDisc failed");
                        }
                    }
                    msg = group_fail.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(reason) = sig.body().deserialize::<String>() {
                            tracing::error!(reason = %reason, "miracast sink: GroupFormationFailure");
                        } else {
                            tracing::error!("miracast sink: GroupFormationFailure (opaque)");
                        }
                    }
                    msg = wps_fail.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(dict) = sig.body().deserialize::<HashMap<String, OwnedValue>>() {
                            let reason = dict.get("msg").and_then(|v| v.downcast_ref::<&str>().ok()).unwrap_or("");
                            let error = dict.get("config_error").and_then(|v| v.downcast_ref::<u32>().ok()).unwrap_or(0);
                            tracing::error!(reason, config_error = error, "miracast sink: WpsFailed (PIN mismatch or timeout)");
                        }
                    }
                    msg = gon_fail.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(dict) = sig.body().deserialize::<HashMap<String, OwnedValue>>() {
                            tracing::error!(?dict, "miracast sink: GONegotiationFailure");
                        }
                    }
                    msg = gon_ok.next() => {
                        let Some(sig) = msg else { break };
                        tracing::info!(?sig, "sink: GONegotiationSuccess");
                    }
                    msg = gon.next() => {
                        let Some(sig) = msg else { break };
                        // supplicant's signal body is
                        // `(peer: o, password_id: q, go_intent: y)`.
                        let Ok((peer, pwd_id, go_intent)): Result<(OwnedObjectPath, u16, u8), _>
                            = sig.body().deserialize() else { continue };
                        // Skip if we Connect-ed to this peer recently;
                        // doubling up aborts the first attempt.
                        let now = std::time::Instant::now();
                        if let Some((p, t)) = &last_connect_at
                            && p == &peer
                            && now.duration_since(*t) < std::time::Duration::from_secs(5)
                        {
                            tracing::debug!(peer = %peer.as_str(), "sink: GON duplicate, skipping second Connect");
                            continue;
                        }

                        // WSC Device Password ID (spec §12):
                        //   0x0000 Default / label PIN
                        //   0x0001 User-specified (peer wants *us* to display a PIN)
                        //   0x0004 Push-Button
                        //   0x0005 Registrar-specified
                        //   0x0007 NFC
                        // Prefer the PIN supplicant generated at
                        // ProvDisc time for this very peer — that's the
                        // PIN the phone is prompting for.
                        let prov_pin_for_peer = last_prov_pin
                            .as_ref()
                            .filter(|(p, _)| p == &peer)
                            .map(|(_, pin)| pin.clone());
                        let (wps_method, maybe_pin): (&str, Option<String>) = match pwd_id {
                            0x0004 => ("pbc", None),
                            _ => match (&pairing, prov_pin_for_peer) {
                                (_, Some(pin)) => ("display", Some(pin)),
                                (PairingMode::PushButton, None) => {
                                    let pin = generate_wps_pin();
                                    tracing::warn!(
                                        pin = %pin,
                                        "miracast sink: GON asked PIN w/o ProvDisc hint — displaying **{}**",
                                        pin,
                                    );
                                    ("display", Some(pin))
                                }
                                (PairingMode::DisplayPin(p), None) => ("display", Some(p.clone())),
                                (PairingMode::EnterPin(p), None) => ("keypad", Some(p.clone())),
                            },
                        };
                        tracing::info!(
                            peer = %peer.as_str(),
                            pwd_id = format_args!("0x{:04x}", pwd_id),
                            go_intent,
                            wps_method,
                            pin = maybe_pin.as_deref().unwrap_or("-"),
                            "sink: auto-accepting GON request"
                        );
                        let mut args: HashMap<&str, Value> = HashMap::new();
                        args.insert("peer", Value::from(peer.as_ref().to_owned()));
                        args.insert("wps_method", Value::from(wps_method));
                        if let Some(pin) = maybe_pin {
                            args.insert("pin", Value::from(pin));
                        }
                        // Bias ourselves toward being the Group Owner —
                        // the sink needs to run a DHCP server for the
                        // phone. supplicant parses `go_intent` as
                        // `DBUS_TYPE_INT32` (`i`), not u32 — a u32
                        // triggers InvalidArgs on the whole call.
                        args.insert("go_intent", Value::from(14i32));
                        // Record BEFORE Connect so dedupe fires even
                        // when Connect itself errored out.
                        last_connect_at = Some((peer.clone(), now));
                        if let Err(e) = p2p_owned.call_method("Connect", &(args,)).await {
                            tracing::warn!(error = %e, "sink: Connect");
                        }
                    }
                    msg = started.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(dict) = sig.body().deserialize::<HashMap<String, OwnedValue>>() {
                            if let Some(group) = parse_group_started(&dict) {
                                tracing::info!(iface = %group.interface_path, role = ?group.role, "sink: group started");
                                if tx.send(group).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Stop P2P discovery / listen. Idempotent.
    pub async fn stop_discovery(&self) -> Result<(), TransportError> {
        let p2p = self.p2p_proxy().await?;
        let _ = p2p.call_method("StopFind", &()).await;
        Ok(())
    }

    /// Initiate GO negotiation with a peer. Returns the virtual group
    /// interface name (e.g. `p2p-wlan0-0`) once `GroupStarted` fires.
    /// The caller then runs RTSP on that iface's IP.
    pub async fn connect_peer(&self, peer_path: &str) -> Result<GroupStarted, TransportError> {
        let p2p = self.p2p_proxy().await?;
        let mut stream = p2p
            .receive_signal("GroupStarted")
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("subscribe GroupStarted: {e}")))?;

        let mut args: HashMap<&str, Value> = HashMap::new();
        let peer_obj =
            ObjectPath::try_from(peer_path).map_err(|e| TransportError::LinuxP2p(e.to_string()))?;
        args.insert("peer", Value::from(peer_obj));
        args.insert("wps_method", Value::from("pbc"));
        p2p.call_method("Connect", &(args,))
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("Connect: {e}")))?;

        // 30 s is the Wi-Fi Alliance recommended GO negotiation ceiling.
        let fut = async {
            while let Some(sig) = stream.next().await {
                if let Ok(dict) = sig.body().deserialize::<HashMap<String, OwnedValue>>()
                    && let Some(iface_val) = dict.get("interface_object")
                    && let Ok(path) = iface_val.try_clone().and_then(ObjectPath::try_from)
                {
                    let role = dict
                        .get("role")
                        .and_then(|v| v.downcast_ref::<&str>().ok())
                        .unwrap_or("client")
                        .to_string();
                    return Ok(GroupStarted {
                        interface_path: path.to_string(),
                        role: if role == "GO" {
                            GroupRole::Owner
                        } else {
                            GroupRole::Client
                        },
                    });
                }
            }
            Err(TransportError::LinuxP2p("GroupStarted stream ended".into()))
        };
        tokio::time::timeout(Duration::from_secs(30), fut)
            .await
            .map_err(|_| TransportError::LinuxP2p("GO negotiation timed out".into()))?
    }
}

/// Generate a valid 8-digit WPS PIN (7 random digits + mod-10 checksum).
/// Wi-Fi Protected Setup §C.1 — supplicant rejects PINs whose checksum
/// doesn't verify.
fn generate_wps_pin() -> String {
    let base: u32 = rand::random::<u32>() % 10_000_000;
    let checksum = wps_pin_checksum(base);
    format!("{base:07}{checksum}")
}

fn wps_pin_checksum(pin_7: u32) -> u8 {
    let p = u64::from(pin_7) * 10;
    let accum = 3 * ((p / 10_000_000) % 10)
        + ((p / 1_000_000) % 10)
        + 3 * ((p / 100_000) % 10)
        + ((p / 10_000) % 10)
        + 3 * ((p / 1_000) % 10)
        + ((p / 100) % 10)
        + 3 * ((p / 10) % 10);
    ((10 - (accum % 10)) % 10) as u8
}

#[cfg(test)]
mod pin_tests {
    use super::*;

    #[test]
    fn checksum_reference_vectors() {
        // WPS spec §C.1 worked examples.
        assert_eq!(wps_pin_checksum(1234567), 0);
        assert_eq!(wps_pin_checksum(0), 0);
        // Self-consistency: every generated PIN must validate.
        for _ in 0..100 {
            let pin = generate_wps_pin();
            let (base, cs) = pin.split_at(7);
            let base: u32 = base.parse().unwrap();
            let cs: u8 = cs.parse().unwrap();
            assert_eq!(wps_pin_checksum(base), cs);
        }
    }
}

fn parse_group_started(dict: &HashMap<String, OwnedValue>) -> Option<GroupStarted> {
    let iface_val = dict.get("interface_object")?;
    let path = iface_val.try_clone().ok().and_then(|v| ObjectPath::try_from(v).ok())?;
    let role = dict
        .get("role")
        .and_then(|v| v.downcast_ref::<&str>().ok())
        .unwrap_or("client")
        .to_string();
    Some(GroupStarted {
        interface_path: path.to_string(),
        role: if role == "GO" {
            GroupRole::Owner
        } else {
            GroupRole::Client
        },
    })
}

impl LinuxP2pClient {

    async fn p2p_proxy(&self) -> Result<Proxy<'_>, TransportError> {
        Proxy::new(
            &self.conn,
            SUPPLICANT_BUS,
            self.interface_path.clone(),
            P2P_IFACE,
        )
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("p2p proxy: {e}")))
    }
}

/// The virtual group interface a GO negotiation produced.
#[derive(Debug, Clone)]
pub struct GroupStarted {
    /// supplicant object path of the new interface (e.g.
    /// `/fi/w1/wpa_supplicant1/Interfaces/5`). Pair with
    /// `fi.w1.wpa_supplicant1.Interface` to read the `Ifname` property
    /// if you need the kernel name (`p2p-wlan0-0`).
    pub interface_path: String,
    pub role: GroupRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupRole {
    Owner,
    Client,
}

/// How the peer authenticates with the sink during WPS provisioning.
///
/// WPS config method bits (Wi-Fi Simple Config §12):
/// - `0x0080` Push-Button
/// - `0x0008` Display
/// - `0x0100` Keypad
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingMode {
    /// Push-Button Config. Peer sees the sink in its picker and
    /// taps "Connect"; sink auto-accepts, no PIN. Default — matches
    /// how most phones/TVs ship out of the box.
    PushButton,
    /// Sink *displays* a fixed 4- or 8-digit PIN; the peer types it.
    /// Classic "TV Miracast" flow.
    DisplayPin(String),
    /// Sink *prompts the user* for a PIN the peer displays. Rare in
    /// consumer flows; useful when the sink is a device without a
    /// screen but has a keypad.
    EnterPin(String),
}

impl PairingMode {
    /// WPS ConfigMethod bitmap we advertise in P2PDeviceConfig.
    #[must_use]
    pub const fn config_methods_bitmap(&self) -> u32 {
        match self {
            Self::PushButton => 0x0080,
            Self::DisplayPin(_) => 0x0008,
            Self::EnterPin(_) => 0x0100,
        }
    }

    /// `wps_method` string supplicant's `Connect` expects.
    #[must_use]
    pub const fn wps_method(&self) -> &'static str {
        match self {
            Self::PushButton => "pbc",
            Self::DisplayPin(_) => "display",
            Self::EnterPin(_) => "keypad",
        }
    }

    /// PIN to pass in the `Connect` dict, when any.
    #[must_use]
    pub fn pin(&self) -> Option<&str> {
        match self {
            Self::PushButton => None,
            Self::DisplayPin(p) | Self::EnterPin(p) => Some(p.as_str()),
        }
    }
}

impl Default for PairingMode {
    fn default() -> Self {
        Self::PushButton
    }
}

async fn resolve_or_create_interface(conn: &Connection) -> Result<OwnedObjectPath, TransportError> {
    let root = Proxy::new(conn, SUPPLICANT_BUS, SUPPLICANT_ROOT_PATH, ROOT_IFACE)
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("root proxy: {e}")))?;

    // Prefer an already-existing interface so we coexist with
    // NetworkManager (which keeps the sole wpa_supplicant around).
    if let Ok(existing) = root
        .get_property::<Vec<OwnedObjectPath>>("Interfaces")
        .await
        && let Some(path) = existing.into_iter().next()
    {
        return Ok(path);
    }

    // Fallback: try wlan0. Users with a different radio can override via
    // a future `with_interface(name)` hook on the builder.
    let mut args: HashMap<&str, Value> = HashMap::new();
    args.insert("Ifname", Value::from("wlan0"));
    let reply = root
        .call_method("CreateInterface", &(args,))
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("CreateInterface: {e}")))?;
    let path: OwnedObjectPath = reply
        .body()
        .deserialize()
        .map_err(|e| TransportError::LinuxP2p(format!("CreateInterface reply: {e}")))?;
    Ok(path)
}

async fn read_peer(
    conn: &Connection,
    path: &ObjectPath<'_>,
) -> Result<DiscoveredPeer, TransportError> {
    let proxy = Proxy::new(conn, SUPPLICANT_BUS, path.to_owned(), PEER_IFACE)
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("peer proxy: {e}")))?;

    let name: String = proxy
        .get_property("DeviceName")
        .await
        .unwrap_or_else(|_| "Unknown".to_string());
    let address: Vec<u8> = proxy
        .get_property("DeviceAddress")
        .await
        .map_err(|e| TransportError::LinuxP2p(format!("DeviceAddress: {e}")))?;
    let address_arr: [u8; 6] = address
        .as_slice()
        .try_into()
        .map_err(|_| TransportError::LinuxP2p("DeviceAddress is not 6 bytes".into()))?;

    // `IEs: ay` (raw per-peer IE blob) + `VendorExtension: aay` (vendor
    // IEs split out). Different supplicant builds expose one or both;
    // scan both.
    let direct: Vec<u8> = proxy.get_property("IEs").await.unwrap_or_default();
    let vendor: Vec<Vec<u8>> = proxy
        .get_property("VendorExtension")
        .await
        .unwrap_or_default();

    let wfd = find_wfd_in_vendor(&direct, &vendor);

    // Keep the compiler happy about `SubelementId` export — we deliberately
    // take it in-scope so reviewers can see the WFD IE parse path at a
    // glance even though parsing is delegated to the `ie` module.
    let _ = SubelementId::DeviceInformation;

    Ok(DiscoveredPeer {
        path: path.to_string(),
        name,
        address: address_arr,
        wfd,
    })
}

fn find_wfd_in_vendor(direct: &[u8], vendor: &[Vec<u8>]) -> Option<WfdIe> {
    if let Some(wfd) = extract_wfd(direct) {
        return Some(wfd);
    }
    for v in vendor {
        if let Some(wfd) = extract_wfd(v) {
            return Some(wfd);
        }
    }
    None
}

fn extract_wfd(bytes: &[u8]) -> Option<WfdIe> {
    // Each IE: `0xDD <len> <OUI:3> <OUI_type:1> <subelements...>`.
    // `IEs: ay` may be a concatenation of several; scan for 0xDD.
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] != 0xDD {
            i += 1;
            continue;
        }
        let len = bytes[i + 1] as usize;
        if i + 2 + len > bytes.len() || len < 4 {
            i += 1;
            continue;
        }
        let oui = &bytes[i + 2..i + 5];
        let oui_type = bytes[i + 5];
        if oui == WIFI_ALLIANCE_OUI && oui_type == WFD_OUI_TYPE {
            let payload = &bytes[i + 6..i + 2 + len];
            if let Ok(wfd) = WfdIe::parse(payload) {
                return Some(wfd);
            }
        }
        i += 2 + len;
    }
    None
}

/// Probe whether the host has a usable wpa_supplicant service — the
/// GUI calls this to tell users "install wpa_supplicant" or "your
/// adapter does not expose P2P" without blowing up on D-Bus errors.
pub async fn probe_supplicant() -> Result<(), TransportError> {
    let _ = LinuxP2pClient::connect(&WfdIe::new(crate::ie::DeviceInformation::linux_dual_role()))
        .await?;
    Ok(())
}

// `Mutex` is only pulled in by callers that hold the client across
// task boundaries. Nothing in this module needs it directly today, so
// we avoid defining an unused helper type.
#[cfg(any())]
type _M = Mutex<()>;
