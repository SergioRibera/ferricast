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
const INTERFACE_IFACE: &str = "fi.w1.wpa_supplicant1.Interface";
const P2P_IFACE: &str = "fi.w1.wpa_supplicant1.Interface.P2PDevice";
const PEER_IFACE: &str = "fi.w1.wpa_supplicant1.Peer";
const BSS_IFACE: &str = "fi.w1.wpa_supplicant1.BSS";

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
        // Detect STA's current frequency early so we can pass it to
        // `Connect` later. mt7925e-class single-radio drivers can only
        // host the P2P group on the same freq the STA is already
        // tuned to; without passing `frequency` to Connect, supplicant
        // lets the peer-GO pick whatever it wants and the driver then
        // silently refuses to tune two channels at once, leaving the
        // group formation to time out.
        let iface_ref_for_freq = self.interface_path.as_ref();
        let sta_freq = detect_sta_frequency(&self.conn, &iface_ref_for_freq).await;
        // Primary Device Type (WPS 2.0 Appendix B):
        //   category(u16 BE) = 0x0005  (Display)
        //   OUI(u8[3])       = 50:6F:9A (Wi-Fi Alliance)
        //   OUI type(u8)     = 0x04    (WPS Device Category)
        //   sub-category(u16 BE) = 0x0001 (Television)
        // Byte 5 was 0x02 (wrong) — supplicant-managed sinks may still
        // accept it but Android Smart View mis-classifies the peer.
        const PRIMARY_DEVICE_TYPE_DISPLAY_TV: [u8; 8] =
            [0x00, 0x05, 0x50, 0x6F, 0x9A, 0x04, 0x00, 0x01];

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
        // Pin P2P to a channel compatible with the current STA
        // association. mt7925e-class drivers can't do STA + P2P on
        // different bands; if STA sits on 5 GHz "Rico 5G" and P2P
        // listens on 2.4 GHz ch 1, group formation silently dies
        // after GO-Nego-Success. We read `CurrentBSS.Frequency` via
        // D-Bus and pick a matching P2P channel + reg class. No STA
        // → fall back to social channel 1 / reg class 81.
        let iface_ref = self.interface_path.as_ref();
        let (ch, reg) = detect_sta_channel(&self.conn, &iface_ref)
            .await
            .unwrap_or((1, 81));
        cfg.insert("ListenRegClass", Value::from(reg));
        cfg.insert("ListenChannel", Value::from(ch));
        cfg.insert("OperRegClass", Value::from(reg));
        cfg.insert("OperChannel", Value::from(ch));
        // `GOIntent` is part of the persisted config too — supplicant
        // reuses it when it auto-responds before we get a chance to
        // override in Connect(). 14 keeps the sink preferred-but-not-
        // forced (phone with hard 15 wins).
        cfg.insert("GOIntent", Value::from(14u32));
        // Reuse STA iface for the P2P group instead of spawning a
        // p2p-wlp7s0-N virtual. mt7925e sometimes can't bring up the
        // virtual iface while STA is active on the same radio; the
        // group formation then silently stalls after GON-Success.
        cfg.insert("NoGroupIface", Value::from(true));
        if let Err(e) = p2p.set_property("P2PDeviceConfig", Value::from(cfg)).await {
            tracing::error!(error = %e, "miracast sink: P2PDeviceConfig set FAILED — peers will see us as unnamed");
        } else {
            tracing::info!(
                name = %device_name,
                p2p_channel = ch,
                reg_class = reg,
                "miracast sink: P2PDeviceConfig set (channel matched to STA band)"
            );
        }

        // `Find({Timeout: 0})` cycles scan + listen states internally,
        // keeping the sink discoverable in the phone's picker. `Listen`
        // alone parks the radio for at most `max_remain_on_chan` (5 s
        // default) and then supplicant goes idle; `Connect` also clears
        // `p2p_long_listen`, so after a single GON attempt the radio
        // would go dark and Android's picker would lose us. MiracleCast
        // (`wifid-supplicant.c:2019-2062`) and gnome-network-displays
        // (`nd-wfd-p2p-provider.c:195-208`) both re-issue `Find` on a
        // timer for this reason.
        arm_find(&p2p).await?;

        let (tx, rx) = mpsc::channel(4);
        let conn = self.conn.clone();
        let interface_path = self.interface_path.clone();
        let pairing = pairing.clone();
        let sta_freq_for_task = sta_freq;
        tokio::spawn(async move {
            let p2p_owned =
                match Proxy::new(&conn, SUPPLICANT_BUS, interface_path.clone(), P2P_IFACE).await {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error = %e, "sink: p2p proxy");
                        return;
                    }
                };
            // 20 s re-arm ticker: supplicant's own P2PState is
            // unreliable (per MiracleCast's `start_scan` comment at
            // `wifid-supplicant.c:2036-2041`). Belt-and-suspenders
            // re-arm keeps the sink visible across driver quirks.
            let mut rearm = tokio::time::interval(std::time::Duration::from_secs(20));
            rearm.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            rearm.tick().await; // consume the immediate first tick
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
            // Always overwritten by newer emissions; Android Smart View
            // retransmits PD-Req and the phone displays the LAST pin
            // supplicant generated, so dedup would stale the state.
            let mut last_prov_pin: Option<(OwnedObjectPath, String)> = None;
            // Set true once we get GroupStarted or synthesise one from
            // InterfaceAdded, so the Connect watchdog can distinguish
            // "peer stalled" from "group already up".
            let group_formed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            // Suppress the periodic Find re-arm during an in-flight
            // GON exchange. Calling `Find` while supplicant is in
            // `P2P_PROVISIONING` or `P2P_GO_NEG` cancels the state
            // and the peer sees a dropped session — Android then
            // closes its connect dialog ("died"). Set true on
            // ProvDisc or GON, cleared on any terminal signal.
            let gon_in_flight = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            // When set, a background branch logs the PIN after an
            // 800 ms quiet period so phone retries collapse to one
            // display event.
            let mut pending_pin_display: Option<std::time::Instant> = None;
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
            // Some supplicant builds skip `GroupStarted` and only emit
            // `InterfaceAdded` on the root path when the new p2p-*
            // virtual iface appears. Subscribe both so group formation
            // is detected either way.
            let root_proxy = match Proxy::new(
                &conn,
                SUPPLICANT_BUS,
                SUPPLICANT_ROOT_PATH,
                ROOT_IFACE,
            )
            .await
            {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: root proxy");
                    return;
                }
            };
            let mut iface_added = match root_proxy.receive_signal("InterfaceAdded").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe InterfaceAdded");
                    return;
                }
            };
            let mut group_finished = match p2p_owned.receive_signal("GroupFinished").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "sink: subscribe GroupFinished");
                    return;
                }
            };
            loop {
                tokio::select! {
                    msg = prov_pin.next() => {
                        let Some(sig) = msg else { break };
                        // `ProvisionDiscoveryRequestDisplayPin(peer: o, pin: s)`.
                        // Android Smart View retransmits PD-Req; each
                        // retry makes supplicant regenerate a fresh PIN.
                        // The phone screen displays whichever one was
                        // ACK'd LAST. We silently cache every emission;
                        // a debounce ticker below surfaces only the
                        // final value so the user sees one PIN.
                        let Ok((peer, pin)): Result<(OwnedObjectPath, String), _>
                            = sig.body().deserialize() else { continue };
                        // Peer started provisioning — suspend Find re-arm.
                        gon_in_flight.store(true, std::sync::atomic::Ordering::Release);
                        last_prov_pin = Some((peer, pin));
                        pending_pin_display = Some(std::time::Instant::now());
                    }
                    _ = async {
                        match pending_pin_display {
                            Some(t) => {
                                let elapsed = t.elapsed();
                                let wait = std::time::Duration::from_millis(800)
                                    .saturating_sub(elapsed);
                                tokio::time::sleep(wait).await;
                            }
                            None => std::future::pending::<()>().await,
                        }
                    } => {
                        if let Some((peer, pin)) = last_prov_pin.as_ref() {
                            tracing::warn!(
                                peer = %peer.as_str(),
                                pin = %pin,
                                "miracast sink: PIN to type on phone → **{}** ←",
                                pin,
                            );
                        }
                        pending_pin_display = None;
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
                        gon_in_flight.store(false, std::sync::atomic::Ordering::Release);
                        rearm_find_from_task(&p2p_owned).await;
                    }
                    msg = group_fail.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(reason) = sig.body().deserialize::<String>() {
                            tracing::error!(reason = %reason, "miracast sink: GroupFormationFailure");
                        } else {
                            tracing::error!("miracast sink: GroupFormationFailure (opaque)");
                        }
                        gon_in_flight.store(false, std::sync::atomic::Ordering::Release);
                        rearm_find_from_task(&p2p_owned).await;
                    }
                    msg = wps_fail.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(dict) = sig.body().deserialize::<HashMap<String, OwnedValue>>() {
                            let reason = dict.get("msg").and_then(|v| v.downcast_ref::<&str>().ok()).unwrap_or("");
                            let error = dict.get("config_error").and_then(|v| v.downcast_ref::<u32>().ok()).unwrap_or(0);
                            tracing::error!(reason, config_error = error, "miracast sink: WpsFailed (PIN mismatch or timeout)");
                        }
                        gon_in_flight.store(false, std::sync::atomic::Ordering::Release);
                        rearm_find_from_task(&p2p_owned).await;
                    }
                    msg = gon_fail.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(dict) = sig.body().deserialize::<HashMap<String, OwnedValue>>() {
                            tracing::error!(?dict, "miracast sink: GONegotiationFailure");
                        }
                        gon_in_flight.store(false, std::sync::atomic::Ordering::Release);
                        rearm_find_from_task(&p2p_owned).await;
                    }
                    msg = gon_ok.next() => {
                        let Some(sig) = msg else { break };
                        tracing::info!(?sig, "sink: GONegotiationSuccess");
                    }
                    _ = rearm.tick() => {
                        gon_in_flight.store(false, std::sync::atomic::Ordering::Release);
                        rearm_find_from_task(&p2p_owned).await;
                    }
                    msg = group_finished.next() => {
                        let Some(_sig) = msg else { break };
                        tracing::info!("sink: GroupFinished — re-arming Find");
                        group_formed.store(false, std::sync::atomic::Ordering::Release);
                        gon_in_flight.store(false, std::sync::atomic::Ordering::Release);
                        rearm_find_from_task(&p2p_owned).await;
                    }
                    msg = iface_added.next() => {
                        let Some(sig) = msg else { break };
                        // `InterfaceAdded(o, a{sv})`. When path contains
                        // "p2p-" it's the virtual group iface the sink
                        // needs — treat as GroupStarted fallback.
                        if let Ok((path, props)) =
                            sig.body().deserialize::<(OwnedObjectPath, HashMap<String, OwnedValue>)>()
                        {
                            let ifname = props
                                .get("Ifname")
                                .and_then(|v| v.downcast_ref::<&str>().ok())
                                .map(str::to_string)
                                .unwrap_or_default();
                            if ifname.starts_with("p2p-") {
                                tracing::info!(
                                    iface = %path.as_str(),
                                    ifname = %ifname,
                                    "sink: InterfaceAdded (group iface) — synthesizing GroupStarted"
                                );
                                let group = GroupStarted {
                                    interface_path: path.to_string(),
                                    role: GroupRole::Client,
                                };
                                group_formed.store(true, std::sync::atomic::Ordering::Release);
                                if tx.send(group).await.is_err() {
                                    return;
                                }
                            } else {
                                tracing::debug!(iface = %path.as_str(), ifname, "sink: InterfaceAdded (non-group)");
                            }
                        }
                    }
                    msg = gon.next() => {
                        let Some(sig) = msg else { break };
                        // supplicant's signal body is
                        // `(peer: o, password_id: q, go_intent: y)`.
                        let Ok((peer, pwd_id, go_intent)): Result<(OwnedObjectPath, u16, u8), _>
                            = sig.body().deserialize() else { continue };
                        gon_in_flight.store(true, std::sync::atomic::Ordering::Release);
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
                        // `peer` must be the explicit `Value::ObjectPath`
                        // variant (`o` signature). `Value::from(OwnedObjectPath)`
                        // round-trips through generic conversion; the
                        // explicit variant is what supplicant's handler
                        // (dbus_new_handlers_p2p.c:619) matches on.
                        args.insert(
                            "peer",
                            Value::ObjectPath(peer.as_ref().to_owned()),
                        );
                        args.insert("wps_method", Value::from(wps_method));
                        if let Some(pin) = maybe_pin {
                            args.insert("pin", Value::from(pin));
                        }
                        // GO intent trade-off:
                        //   15 = spec-recommended for sink (WFD 2.1 §4.5.1)
                        //        but phones like Xiaomi Smart View also
                        //        send 15 → P2P status 9 "both indicated
                        //        Intent 15" → GON aborts.
                        //   14 = still high (beats default 7), no tie
                        //        against phones that hard-code 15.
                        // We lose GO role when phone wins; works anyway
                        // because RTSP listens on 0.0.0.0:7236 and the
                        // P2P client iface still hosts that port.
                        args.insert("go_intent", Value::from(14i32));
                        // If the local STA is associated, force GON to
                        // settle on that exact freq. Single-radio
                        // drivers can't retune, so if the peer GO
                        // picks a different channel the group never
                        // forms. supplicant's Connect() handler takes
                        // `frequency` as DBUS_TYPE_INT32 (MHz).
                        if let Some(freq) = sta_freq_for_task {
                            args.insert("frequency", Value::from(i32::from(freq)));
                            tracing::info!(freq_mhz = freq, "sink: pinning Connect frequency to STA band");
                        }
                        // Record BEFORE Connect so dedupe fires even
                        // when Connect itself errored out.
                        last_connect_at = Some((peer.clone(), now));
                        match p2p_owned.call_method("Connect", &(args,)).await {
                            Ok(reply) => {
                                // Connect returns the PIN supplicant
                                // actually armed for the subsequent
                                // GO-Negotiation-Response. Compare with
                                // what we passed — mismatch is a sure
                                // sign the WPS handshake will fail.
                                let armed: String = reply
                                    .body()
                                    .deserialize()
                                    .unwrap_or_else(|_| "<no-reply>".to_string());
                                tracing::info!(armed_pin = %armed, "sink: Connect accepted, awaiting GON completion");
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "sink: Connect");
                            }
                        }
                        // Watchdog: if neither GroupStarted nor any
                        // failure signal arrives within 15 s, the peer
                        // bailed silently. Skip when the group has
                        // already formed in the meantime (false-positive
                        // after successful connect).
                        let watchdog_peer = peer.clone();
                        let watchdog_conn = conn.clone();
                        let watchdog_iface = interface_path.clone();
                        let group_ready_flag = group_formed.clone();
                        tokio::spawn(async move {
                            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
                            if group_ready_flag.load(std::sync::atomic::Ordering::Acquire) {
                                return;
                            }
                            tracing::warn!(
                                peer = %watchdog_peer.as_str(),
                                "sink: 15s after Connect — no GroupStarted/Failure signal. \
                                 Peer probably timed out. Issuing P2PDevice.Cancel() to reset."
                            );
                            match Proxy::new(
                                &watchdog_conn,
                                SUPPLICANT_BUS,
                                watchdog_iface,
                                P2P_IFACE,
                            )
                            .await
                            {
                                Ok(p) => {
                                    if let Err(e) = p.call_method("Cancel", &()).await {
                                        tracing::debug!(error = %e, "watchdog: Cancel ignored");
                                    }
                                }
                                Err(e) => tracing::debug!(error = %e, "watchdog: proxy"),
                            }
                        });
                    }
                    msg = started.next() => {
                        let Some(sig) = msg else { break };
                        if let Ok(dict) = sig.body().deserialize::<HashMap<String, OwnedValue>>() {
                            if let Some(group) = parse_group_started(&dict) {
                                tracing::info!(iface = %group.interface_path, role = ?group.role, "sink: group started");
                                group_formed.store(true, std::sync::atomic::Ordering::Release);
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

/// Issue `P2PDevice.Find({Timeout: 0})`. Idempotent — supplicant drops
/// a duplicate with `AlreadyExists` which we log at debug level.
async fn arm_find(p2p: &Proxy<'_>) -> Result<(), TransportError> {
    let args: HashMap<&str, Value> = HashMap::new();
    match p2p.call_method("Find", &(args,)).await {
        Ok(_) => Ok(()),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("AlreadyExists") || msg.contains("in progress") {
                tracing::debug!(error = %e, "sink: Find already active, skipping");
                Ok(())
            } else {
                tracing::warn!(error = %e, "sink: Find rearm failed");
                Err(TransportError::LinuxP2p(format!("Find: {e}")))
            }
        }
    }
}

/// Re-arm Find from inside the task loop. Logs on failure but doesn't
/// propagate — losing re-arm once is better than killing the loop.
async fn rearm_find_from_task(p2p: &Proxy<'_>) {
    if let Err(e) = arm_find(p2p).await {
        tracing::debug!(error = %e, "sink: re-arm Find deferred to next ticker");
    }
}

/// Read the current STA association's channel + Wi-Fi Alliance
/// regulatory class via wpa_supplicant D-Bus. Returns `None` when the
/// interface isn't associated to an infrastructure Wi-Fi.
async fn detect_sta_channel(
    conn: &Connection,
    interface_path: &ObjectPath<'_>,
) -> Option<(u32, u32)> {
    let freq = detect_sta_frequency(conn, interface_path).await?;
    let (ch, reg) = freq_to_channel_class(freq);
    tracing::info!(
        freq_mhz = freq,
        channel = ch,
        reg_class = reg,
        "sink: STA current channel detected — pinning P2P to match band"
    );
    Some((ch, reg))
}

/// Raw STA centre frequency (MHz) via D-Bus, used both for channel
/// mapping and for `Connect(frequency=...)` so GON settles on the
/// freq the radio is already tuned to.
async fn detect_sta_frequency(conn: &Connection, interface_path: &ObjectPath<'_>) -> Option<u16> {
    let iface = Proxy::new(
        conn,
        SUPPLICANT_BUS,
        interface_path.to_owned(),
        INTERFACE_IFACE,
    )
    .await
    .ok()?;
    let bss_path: OwnedObjectPath = iface.get_property("CurrentBSS").await.ok()?;
    if bss_path.as_str() == "/" {
        return None;
    }
    let bss = Proxy::new(conn, SUPPLICANT_BUS, bss_path, BSS_IFACE).await.ok()?;
    bss.get_property::<u16>("Frequency").await.ok()
}

/// Map centre frequency in MHz to (channel, Wi-Fi Alliance regulatory
/// class). Picks the P2P-usable class for each band — DFS channels
/// (52-144) are deliberately excluded because P2P can't use them.
fn freq_to_channel_class(freq: u16) -> (u32, u32) {
    match freq {
        // 2.4 GHz: channels 1-13. Reg class 81 (global).
        2412..=2472 => (u32::from((freq - 2407) / 5), 81),
        // 5 GHz UNII-1 (36/40/44/48). Reg class 115.
        5180..=5240 => (u32::from((freq - 5000) / 5), 115),
        // 5 GHz UNII-3 (149/153/157/161/165). Reg class 124 (US) /
        // 125 (global). Pick 125 so the regdomain check doesn't drop
        // the whole group on non-US hosts.
        5745..=5825 => (u32::from((freq - 5000) / 5), 125),
        // Anything else (DFS, 6 GHz): fall back to social ch 1.
        _ => {
            tracing::warn!(
                freq_mhz = freq,
                "sink: STA on a channel P2P can't share (DFS or 6 GHz); forcing 2.4 GHz ch 1"
            );
            (1, 81)
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

    /// Read `fi.w1.wpa_supplicant1.Interface.Ifname` from the given
    /// supplicant interface path. For a Miracast group that reused
    /// `wlp7s0` (NoGroupIface=true) this returns `"wlp7s0"`; for a
    /// virtual group iface it returns `"p2p-wlp7s0-0"` and so on.
    pub async fn resolve_group_ifname(&self, interface_path: &str) -> Result<String, TransportError> {
        let path = ObjectPath::try_from(interface_path.to_string())
            .map_err(|e| TransportError::LinuxP2p(format!("bad iface path: {e}")))?;
        let iface = Proxy::new(&self.conn, SUPPLICANT_BUS, path, INTERFACE_IFACE)
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("iface proxy: {e}")))?;
        iface
            .get_property::<String>("Ifname")
            .await
            .map_err(|e| TransportError::LinuxP2p(format!("Ifname: {e}")))
    }

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
