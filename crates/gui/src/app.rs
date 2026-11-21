use ferricast::prelude::*;
use ferricast::{ManagerEvent, PairingChallenge, PairingResponse};
use freya::{prelude::*, radio::*};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use uuid::Uuid;

use crate::daemon::{PickerRequest, SharedManager};
use crate::pairing::{self, PairingResponder};
use crate::picker;
use crate::receiver_window;

/// Request to spawn a receiver window for an incoming transmission.
/// Pumped from the manager's sink factory to the App's render loop
/// (which holds `Platform`) over a tokio mpsc — same shape the picker
/// uses to bridge daemon → UI.
///
/// Carries the `rx` halves of the per-session video / audio
/// channels: the sink (manager-side) holds the `tx`, the window
/// (Freya-side) holds the `rx`. One window per session.
pub struct ReceiverWindowReq {
    pub remote: ferricast::RemoteSender,
    pub info: ferricast::MediaInfo,
    pub counters: Arc<receiver_window::ReceiverCounters>,
    pub video_rx: tokio::sync::mpsc::Receiver<receiver_window::VideoFrame>,
    pub audio_rx: tokio::sync::mpsc::Receiver<receiver_window::AudioBuf>,
}

/// Work for the App's window listener (it owns `Platform`).
pub enum UiCommand {
    /// Ask the user for a PIN or password.
    PairingPrompt {
        device_name: String,
        secret: pairing::Secret,
        respond: PairingResponder,
    },
    /// A remote sender connected on a receiver.
    ReceiverSession { receiver_id: Uuid, remote_id: Uuid },
    /// That receiver's session ended: close its window.
    ReceiverEnded { receiver_id: Uuid },
}

/// Where a device's stream is. Devices without an entry are idle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamStatus {
    Starting,
    Streaming,
    Reconnecting { attempt: u32 },
    Failed(String),
}

impl StreamStatus {
    /// A stream exists in the manager (clicking the card stops it).
    pub fn is_active(&self) -> bool {
        match self {
            StreamStatus::Starting
            | StreamStatus::Streaming
            | StreamStatus::Reconnecting { .. } => true,
            StreamStatus::Failed(_) => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeLevel {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notice {
    pub id: u64,
    pub level: NoticeLevel,
    pub text: String,
}

/// Notices kept on screen; older ones are dropped.
const MAX_NOTICES: usize = 4;

#[derive(Default)]
pub struct AppState {
    pub devices: HashMap<Uuid, Device>,
    /// Devices discovery lost while they still had a stream; their
    /// card stays (so the stream can be stopped) until it ends.
    pub lost: HashSet<Uuid>,
    pub streams: HashMap<Uuid, StreamStatus>,
    pub notices: Vec<Notice>,
    next_notice: u64,
}

impl AppState {
    pub fn notify(&mut self, level: NoticeLevel, text: impl Into<String>) {
        self.next_notice += 1;
        self.notices.push(Notice {
            id: self.next_notice,
            level,
            text: text.into(),
        });
        let excess = self.notices.len().saturating_sub(MAX_NOTICES);
        self.notices.drain(..excess);
    }

    pub fn dismiss(&mut self, id: u64) {
        self.notices.retain(|n| n.id != id);
    }

    fn device_name(&self, id: Uuid) -> String {
        self.devices
            .get(&id)
            .map_or_else(|| id.to_string(), |d| d.name.clone())
    }

    /// Fold a manager event into the UI state.
    pub fn apply(&mut self, event: &ManagerEvent) {
        match event {
            ManagerEvent::DeviceFound(device) => {
                self.lost.remove(&device.id);
                self.devices.insert(device.id, device.clone());
            }
            ManagerEvent::DeviceLost(id) => {
                if self.streams.get(id).is_some_and(StreamStatus::is_active) {
                    self.lost.insert(*id);
                } else {
                    self.devices.remove(id);
                    self.streams.remove(id);
                }
            }
            ManagerEvent::StreamStarting { device_id } => {
                self.streams.insert(*device_id, StreamStatus::Starting);
            }
            ManagerEvent::StreamStarted { device_id, .. } => {
                self.streams.insert(*device_id, StreamStatus::Streaming);
            }
            ManagerEvent::StreamReconnecting {
                device_id,
                attempt,
                reason,
            } => {
                self.streams
                    .insert(*device_id, StreamStatus::Reconnecting { attempt: *attempt });
                if *attempt == 1 {
                    let name = self.device_name(*device_id);
                    self.notify(
                        NoticeLevel::Warning,
                        format!("Lost {name}, reconnecting: {reason}"),
                    );
                }
            }
            ManagerEvent::StreamError { device_id, message } => {
                self.streams
                    .insert(*device_id, StreamStatus::Failed(message.clone()));
                let name = self.device_name(*device_id);
                self.notify(NoticeLevel::Error, format!("{name}: {message}"));
            }
            ManagerEvent::StreamWarning { device_id, message } => {
                let name = self.device_name(*device_id);
                self.notify(NoticeLevel::Warning, format!("{name}: {message}"));
            }
            ManagerEvent::StreamStopped { device_id, reason } => {
                // A failure stays visible on the card until the next
                // attempt.
                if self
                    .streams
                    .get(device_id)
                    .is_some_and(StreamStatus::is_active)
                {
                    self.streams.remove(device_id);
                }
                if let Some(reason) = reason {
                    let name = self.device_name(*device_id);
                    self.notify(NoticeLevel::Info, format!("{name} stopped: {reason}"));
                }
                if self.lost.remove(device_id) {
                    self.devices.remove(device_id);
                    self.streams.remove(device_id);
                }
            }
            ManagerEvent::DiscoveryError { protocol, message } => {
                self.notify(
                    NoticeLevel::Warning,
                    format!("{protocol} discovery: {message}"),
                );
            }
            ManagerEvent::ReceiverError { message, .. } => {
                self.notify(
                    NoticeLevel::Error,
                    format!("Incoming cast failed: {message}"),
                );
            }
            ManagerEvent::PairingRequired { .. }
            | ManagerEvent::ReceiverIncoming { .. }
            | ManagerEvent::ReceiverStarted { .. }
            | ManagerEvent::ReceiverStateChanged { .. }
            | ManagerEvent::ReceiverStopped { .. } => {}
        }
    }

    /// Route a pairing challenge: a PIN or password goes to the dialog,
    /// the rest is answered here.
    pub fn pairing_prompt(
        &mut self,
        device_id: Uuid,
        challenge: &PairingChallenge,
        respond: &PairingResponder,
    ) -> Option<UiCommand> {
        match challenge {
            PairingChallenge::Pin { digits } => Some(UiCommand::PairingPrompt {
                device_name: self.device_name(device_id),
                secret: pairing::Secret::Pin { digits: *digits },
                respond: respond.clone(),
            }),
            PairingChallenge::None => {
                pairing::answer(respond, PairingResponse::Confirmed);
                None
            }
            PairingChallenge::Credential { rejected } => Some(UiCommand::PairingPrompt {
                device_name: self.device_name(device_id),
                secret: pairing::Secret::Password {
                    rejected: *rejected,
                },
                respond: respond.clone(),
            }),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
pub enum AppChannel {
    Devices,
    Streaming,
    Notices,
}

impl RadioChannel<AppState> for AppChannel {
    // The main window reads every part of the state, and one manager
    // event can touch devices, streams and notices at once.
    fn derive_channel(self, _state: &AppState) -> Vec<Self> {
        vec![
            AppChannel::Devices,
            AppChannel::Streaming,
            AppChannel::Notices,
        ]
    }
}

pub struct FerricastApp {
    pub stream_manager: SharedManager,
    pub radio_station: RadioStation<AppState, AppChannel>,
    /// Receiver for picker requests coming from the daemon's
    /// D-Bus path (e.g. `--background` mode where there's no
    /// in-process share button to trigger the picker). Wrapped in
    /// `Rc<RefCell<Option<_>>>` because `App::render` only sees
    /// `&self` and the receiver isn't `Clone` — the first render
    /// `.take()`s it and starts the listener task.
    pub picker_req_rx: Rc<RefCell<Option<tokio::sync::mpsc::Receiver<PickerRequest>>>>,
    /// Receiver-window requests from the pump. Same one-shot-take
    /// pattern as `picker_req_rx`.
    pub receiver_req_rx: Rc<RefCell<Option<tokio::sync::mpsc::Receiver<ReceiverWindowReq>>>>,
    /// Pairing prompts and receiver lifecycle, same pattern.
    pub ui_cmd_rx: Rc<RefCell<Option<tokio::sync::mpsc::Receiver<UiCommand>>>>,
}

impl App for FerricastApp {
    fn render(&self) -> impl IntoElement {
        let stream_manager = self.stream_manager.clone();
        use_share_radio(move || self.radio_station);

        let devices_radio = use_radio::<AppState, AppChannel>(AppChannel::Devices);
        let streaming_radio = use_radio::<AppState, AppChannel>(AppChannel::Streaming);
        let mut notices_radio = use_radio::<AppState, AppChannel>(AppChannel::Notices);
        let binding = devices_radio.read();

        // `Platform::get()` consumes a root context that's only valid
        // from inside a component body.
        let platform = Platform::get();

        // Picker-request listener: drive the daemon→freya channel
        // exactly once, on first mount, and answer each D-Bus
        // `StartStream` that needs a source with the user's pick.
        let picker_req_rx_init = self.picker_req_rx.clone();
        let picker_platform = platform.clone();
        use_hook(move || {
            if let Some(mut rx) = picker_req_rx_init.borrow_mut().take() {
                let platform = picker_platform;
                spawn(async move {
                    while let Some(req) = rx.recv().await {
                        tracing::debug!(
                            device_id = ?req.device_id,
                            "daemon picker request — opening picker window"
                        );
                        let (dto_tx, dto_rx) = tokio::sync::oneshot::channel();
                        picker::open_picker_for_dto(platform.clone(), dto_tx);
                        let _ = req.reply.send(dto_rx.await.ok().flatten());
                    }
                });
            }
        });

        // Window listener: receiver windows (one per incoming session,
        // closed when that session ends) and pairing prompts.
        let receiver_req_rx_init = self.receiver_req_rx.clone();
        let ui_cmd_rx_init = self.ui_cmd_rx.clone();
        let window_platform = platform.clone();
        use_hook(move || {
            let windows_rx = receiver_req_rx_init.borrow_mut().take();
            let cmd_rx = ui_cmd_rx_init.borrow_mut().take();
            if let (Some(windows_rx), Some(cmd_rx)) = (windows_rx, cmd_rx) {
                spawn(run_window_listener(window_platform, windows_rx, cmd_rx));
            }
        });

        let notices = notices_radio.read().notices.clone();

        let mut cards: Vec<(Uuid, Device)> = binding
            .devices
            .iter()
            .map(|(id, d)| (*id, d.clone()))
            .collect();
        cards.sort_by(|a, b| a.1.name.cmp(&b.1.name));

        rect()
            .expanded()
            .background((18, 18, 24))
            .vertical()
            .child(
                rect()
                    .expanded()
                    .padding(24.)
                    .vertical()
                    .spacing(8.)
                    .maybe_child(
                        binding.devices.is_empty().then_some(
                            label()
                                .text("Buscando dispositivos...")
                                .font_size(13.)
                                .color((140, 140, 160)),
                        ),
                    )
                    .maybe(!binding.devices.is_empty(), |r| {
                        r.child(
                            ScrollView::new()
                                .expanded()
                                .direction(Direction::Vertical)
                                .children(cards.into_iter().map(|(device_id, device)| {
                                    let status =
                                        streaming_radio.read().streams.get(&device_id).cloned();
                                    let lost = binding.lost.contains(&device_id);
                                    let sm = stream_manager.clone();
                                    DeviceCard {
                                        device,
                                        status,
                                        lost,
                                        on_request_picker: Rc::new({
                                            let sm = sm.clone();
                                            let platform = platform.clone();
                                            move || {
                                                picker::open_picker(
                                                    platform.clone(),
                                                    sm.clone(),
                                                    device_id,
                                                    move |e| {
                                                        notices_radio
                                                            .write()
                                                            .notify(NoticeLevel::Error, e);
                                                    },
                                                );
                                            }
                                        }),
                                        on_stop: Rc::new(move || {
                                            let sm = sm.clone();
                                            // Off the UI thread: stopping waits
                                            // for the stream task to wind down.
                                            tokio::spawn(async move {
                                                let sm = sm.read().await;
                                                if let Err(e) = sm.stop_stream(device_id).await {
                                                    tracing::error!(%e, ?device_id, "stop_stream");
                                                }
                                            });
                                        }),
                                        key: DiffKey::default(),
                                    }
                                    .key(device_id)
                                    .into()
                                })),
                        )
                    }),
            )
            .maybe_child((!notices.is_empty()).then(|| {
                rect()
                    .width(Size::fill())
                    .padding(Gaps::new(0., 24., 16., 24.))
                    .vertical()
                    .spacing(6.)
                    .children(notices.into_iter().map(|notice| {
                        let id = notice.id;
                        notice_banner(notice, move || notices_radio.write().dismiss(id)).into()
                    }))
            }))
    }
}

/// Opens receiver windows and pairing prompts; closes a receiver's
/// window when its session ends.
async fn run_window_listener(
    platform: Platform,
    mut windows_rx: tokio::sync::mpsc::Receiver<ReceiverWindowReq>,
    mut cmd_rx: tokio::sync::mpsc::Receiver<UiCommand>,
) {
    let mut windows: HashMap<Uuid, freya::winit::window::WindowId> = HashMap::new();
    let mut sessions: HashMap<Uuid, Uuid> = HashMap::new();
    loop {
        tokio::select! {
            // Window requests first: a session's end must not be
            // handled before its window exists.
            biased;
            req = windows_rx.recv() => {
                let Some(req) = req else { break };
                tracing::info!(
                    sender = %req.remote.addr,
                    audio = req.info.audio.is_some(),
                    video = req.info.video.is_some(),
                    "opening receiver window"
                );
                let remote_id = req.remote.id;
                let wid = receiver_window::open_receiver_window(
                    platform.clone(),
                    req.remote,
                    req.info,
                    req.counters,
                    req.video_rx,
                    req.audio_rx,
                )
                .await;
                if let Some(old) = windows.insert(remote_id, wid) {
                    platform.close_window(old);
                }
            }
            cmd = cmd_rx.recv() => match cmd {
                None => break,
                Some(UiCommand::PairingPrompt { device_name, secret, respond }) => {
                    pairing::open_pairing_dialog(platform.clone(), device_name, secret, respond);
                }
                Some(UiCommand::ReceiverSession { receiver_id, remote_id }) => {
                    sessions.insert(receiver_id, remote_id);
                }
                Some(UiCommand::ReceiverEnded { receiver_id }) => {
                    if let Some(remote_id) = sessions.remove(&receiver_id)
                        && let Some(wid) = windows.remove(&remote_id)
                    {
                        platform.close_window(wid);
                    }
                }
            },
        }
    }
}

fn notice_banner(notice: Notice, mut on_dismiss: impl FnMut() + 'static) -> Rect {
    let (bg, border) = match notice.level {
        NoticeLevel::Info => ((28, 36, 52), (80, 120, 200)),
        NoticeLevel::Warning => ((52, 44, 24), (210, 170, 60)),
        NoticeLevel::Error => ((56, 26, 30), (220, 80, 90)),
    };
    rect()
        .width(Size::fill())
        .background(bg)
        .corner_radius(8.)
        .border(Border::new().fill(border).width(1.))
        .padding(Gaps::new(8., 12., 8., 12.))
        .on_press(move |_| on_dismiss())
        .child(
            label()
                .text(notice.text)
                .font_size(13.)
                .color((235, 235, 240)),
        )
}

// --- DeviceCard component ---

#[derive(Clone)]
struct DeviceCard {
    device: Device,
    status: Option<StreamStatus>,
    /// Discovery no longer sees the device; the card stays while its
    /// stream runs.
    lost: bool,
    /// Open the picker in a separate top-level window.
    on_request_picker: Rc<dyn Fn()>,
    /// Stop (or cancel the start of) the device's stream.
    on_stop: Rc<dyn Fn()>,
    key: DiffKey,
}

impl KeyExt for DeviceCard {
    fn write_key(&mut self) -> &mut DiffKey {
        &mut self.key
    }
}

impl PartialEq for DeviceCard {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.status == other.status && self.lost == other.lost
    }
}

impl DeviceCard {
    fn status_line(&self) -> (String, (u8, u8, u8)) {
        let muted = (170, 170, 190);
        match &self.status {
            None => (self.device.protocol.to_string(), muted),
            Some(StreamStatus::Starting) => ("Starting… (click to cancel)".into(), muted),
            Some(StreamStatus::Streaming) if self.lost => (
                "Streaming, device no longer announced (click to stop)".into(),
                (210, 170, 60),
            ),
            Some(StreamStatus::Streaming) => ("Streaming (click to stop)".into(), (120, 210, 150)),
            Some(StreamStatus::Reconnecting { attempt }) => (
                format!("Reconnecting, attempt {attempt}… (click to stop)"),
                (210, 170, 60),
            ),
            Some(StreamStatus::Failed(message)) => (format!("Failed: {message}"), (230, 110, 110)),
        }
    }
}

impl Component for DeviceCard {
    fn render(&self) -> impl IntoElement {
        let active = self.status.as_ref().is_some_and(StreamStatus::is_active);
        let streaming = matches!(self.status, Some(StreamStatus::Streaming));
        let on_request_picker = self.on_request_picker.clone();
        let on_stop = self.on_stop.clone();
        let (status_text, status_color) = self.status_line();

        let (bg, border_color) = match &self.status {
            Some(StreamStatus::Failed(_)) => ((44, 26, 30), (200, 80, 90)),
            _ if streaming => ((30, 60, 40), (80, 200, 120)),
            _ if active => ((40, 40, 30), (200, 170, 70)),
            _ => ((28, 28, 38), (50, 50, 65)),
        };

        let icon = svg(self.device.protocol_icon.clone())
            .width(Size::px(36.))
            .height(Size::px(36.));
        let icon = if self.device.protocol == "chromecast" {
            icon.fill(Color::WHITE)
        } else {
            icon.stroke(Color::WHITE)
        };

        rect()
            .width(Size::fill())
            .height(Size::px(72.))
            .background(bg)
            .corner_radius(10.)
            .border(Border::new().fill(border_color).width(1.))
            .padding(Gaps::new(0., 16., 0., 16.))
            .cross_align(Alignment::center())
            .horizontal()
            .spacing(12.)
            .on_press(move |_| {
                if active {
                    (on_stop)();
                } else {
                    (on_request_picker)();
                }
            })
            .child(rect().center().child(icon))
            .child(
                rect()
                    .width(Size::fill())
                    .vertical()
                    .spacing(3.)
                    .child(
                        label()
                            .text(self.device.name.clone())
                            .font_size(15.)
                            .color((230, 230, 240)),
                    )
                    .child(
                        label()
                            .text(status_text)
                            .font_size(12.)
                            .max_lines(1)
                            .color(status_color),
                    )
                    .child(
                        label()
                            .text(self.device.addresses.to_string())
                            .font_size(11.)
                            .max_lines(1)
                            .color((130, 130, 150)),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(name: &str) -> Device {
        Device {
            id: Uuid::new_v4(),
            name: name.into(),
            protocol: "chromecast",
            protocol_icon: bytes::Bytes::new(),
            addresses: std::net::SocketAddr::from(([127, 0, 0, 1], 8009)).into(),
            model: None,
            capabilities: DeviceCapabilities::default(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn stream_lifecycle_updates_status() {
        let mut s = AppState::default();
        let d = device("tv");
        let id = d.id;
        s.apply(&ManagerEvent::DeviceFound(d));
        s.apply(&ManagerEvent::StreamStarting { device_id: id });
        assert_eq!(s.streams.get(&id), Some(&StreamStatus::Starting));
        s.apply(&ManagerEvent::StreamStarted {
            device_id: id,
            device_name: "tv".into(),
        });
        s.apply(&ManagerEvent::StreamReconnecting {
            device_id: id,
            attempt: 1,
            reason: "gone".into(),
        });
        assert_eq!(
            s.streams.get(&id),
            Some(&StreamStatus::Reconnecting { attempt: 1 })
        );
        assert_eq!(s.notices.len(), 1);
        s.apply(&ManagerEvent::StreamStopped {
            device_id: id,
            reason: None,
        });
        assert!(!s.streams.contains_key(&id));
    }

    #[test]
    fn failure_stays_visible_after_stop() {
        let mut s = AppState::default();
        let d = device("tv");
        let id = d.id;
        s.apply(&ManagerEvent::DeviceFound(d));
        s.apply(&ManagerEvent::StreamError {
            device_id: id,
            message: "boom".into(),
        });
        s.apply(&ManagerEvent::StreamStopped {
            device_id: id,
            reason: None,
        });
        assert_eq!(
            s.streams.get(&id),
            Some(&StreamStatus::Failed("boom".into()))
        );
        assert_eq!(s.notices[0].level, NoticeLevel::Error);
    }

    #[test]
    fn lost_device_keeps_its_card_while_streaming() {
        let mut s = AppState::default();
        let d = device("tv");
        let id = d.id;
        s.apply(&ManagerEvent::DeviceFound(d));
        s.apply(&ManagerEvent::StreamStarted {
            device_id: id,
            device_name: "tv".into(),
        });
        s.apply(&ManagerEvent::DeviceLost(id));
        assert!(s.devices.contains_key(&id));
        assert_eq!(s.streams.get(&id), Some(&StreamStatus::Streaming));
        s.apply(&ManagerEvent::StreamStopped {
            device_id: id,
            reason: Some("other app".into()),
        });
        assert!(!s.devices.contains_key(&id));
        assert_eq!(s.notices.last().map(|n| n.level), Some(NoticeLevel::Info));
    }

    #[test]
    fn idle_lost_device_is_removed() {
        let mut s = AppState::default();
        let d = device("tv");
        let id = d.id;
        s.apply(&ManagerEvent::DeviceFound(d));
        s.apply(&ManagerEvent::DeviceLost(id));
        assert!(s.devices.is_empty());
    }

    #[test]
    fn notices_are_capped() {
        let mut s = AppState::default();
        for i in 0..10 {
            s.notify(NoticeLevel::Info, format!("n{i}"));
        }
        assert_eq!(s.notices.len(), MAX_NOTICES);
        assert_eq!(s.notices.last().map(|n| n.text.as_str()), Some("n9"));
        let id = s.notices[0].id;
        s.dismiss(id);
        assert_eq!(s.notices.len(), MAX_NOTICES - 1);
    }

    #[test]
    fn pin_and_password_challenges_go_to_the_dialog() {
        let mut s = AppState::default();
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        let respond: PairingResponder = Arc::new(tokio::sync::Mutex::new(Some(tx)));
        let id = Uuid::new_v4();
        assert!(matches!(
            s.pairing_prompt(id, &PairingChallenge::Pin { digits: 4 }, &respond),
            Some(UiCommand::PairingPrompt {
                secret: pairing::Secret::Pin { digits: 4 },
                ..
            })
        ));
        assert!(matches!(
            s.pairing_prompt(
                id,
                &PairingChallenge::Credential { rejected: true },
                &respond
            ),
            Some(UiCommand::PairingPrompt {
                secret: pairing::Secret::Password { rejected: true },
                ..
            })
        ));
        assert!(rx.try_recv().is_err());
        assert!(
            s.pairing_prompt(id, &PairingChallenge::None, &respond)
                .is_none()
        );
        assert!(matches!(rx.try_recv(), Ok(PairingResponse::Confirmed)));
        assert!(s.notices.is_empty());
    }
}
