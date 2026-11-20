//! Receiver-side state machine for one CASTv2 sender connection.
//!
//! [`ReceiverMonitor`] consumes every inbound message (and timer ticks)
//! and answers with an [`Action`] for the I/O loop to carry out. It owns
//! no sockets and takes the current time as an argument, so launch
//! handling, heartbeat liveness and end-of-session detection are tested
//! without a network or a real clock.

use std::time::Duration;

use tokio::time::Instant;
use tracing::{debug, error, info, warn};

use crate::castv2::{
    CastMessage, DEFAULT_RECEIVER_ID, DEFAULT_SENDER_ID, GenericPayload, MediaDetailedErrorCode,
    MediaStatusPayload, ReceiverStatusPayload, namespace, pong_reply,
};

/// Timers driving liveness. Production uses [`Default`]; tests shrink
/// them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Timing {
    /// How often we PING the receiver.
    pub heartbeat: Duration,
    /// Silence after which the receiver counts as gone. Three missed
    /// heartbeat round-trips.
    pub inactivity: Duration,
    /// Minimum gap between two GET_STATUS nudges while waiting for
    /// the launched app to show up.
    pub nudge_interval: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            heartbeat: Duration::from_secs(5),
            inactivity: Duration::from_secs(15),
            nudge_interval: Duration::from_millis(500),
        }
    }
}

/// Why a session stopped being usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SessionEnd {
    /// The user or another sender ended our session on purpose ("Stop
    /// casting", another app took the screen). Reconnecting would steal
    /// the TV back.
    Remote(String),
    /// Anything else: the link died, the receiver errored, a send
    /// failed. Reconnecting is reasonable.
    Failure(String),
}

impl SessionEnd {
    pub(crate) fn to_error(&self) -> ferricast_core::FerricastError {
        match self {
            Self::Remote(why) => ferricast_core::FerricastError::RemoteStopped(why.clone()),
            Self::Failure(why) => ferricast_core::FerricastError::Connection(why.clone()),
        }
    }
}

/// What the I/O loop must do after feeding the monitor.
#[derive(Debug)]
pub(crate) enum Action {
    Nothing,
    /// Send this message (a PONG answering the receiver's PING).
    Reply(CastMessage),
    /// Send a receiver GET_STATUS.
    Nudge,
    /// The app we launched is up; the monitor already switched to the
    /// running phase.
    Launched {
        transport_id: String,
        session_id: String,
    },
    /// The session is over.
    End(SessionEnd),
}

#[derive(Debug)]
enum Phase {
    Launching {
        app_id: String,
        launch_request_id: i64,
    },
    Running {
        transport_id: String,
        session_id: String,
    },
}

#[derive(Debug)]
pub(crate) struct ReceiverMonitor {
    timing: Timing,
    phase: Phase,
    last_rx: Instant,
    last_nudge: Instant,
    nudge_due: Option<Instant>,
    media: MediaTracker,
}

impl ReceiverMonitor {
    /// Start in the launch phase, waiting for `app_id` (asked for with
    /// LAUNCH `launch_request_id`). `now` counts as the moment of the
    /// last nudge because the caller sends GET_STATUS right after
    /// LAUNCH.
    pub(crate) fn launching(
        app_id: &str,
        launch_request_id: i64,
        timing: Timing,
        now: Instant,
    ) -> Self {
        Self {
            timing,
            phase: Phase::Launching {
                app_id: app_id.to_owned(),
                launch_request_id,
            },
            last_rx: now,
            last_nudge: now,
            nudge_due: None,
            media: MediaTracker::default(),
        }
    }

    /// When the I/O loop must call [`Self::on_tick`] if no message
    /// arrives first.
    pub(crate) fn next_wakeup(&self) -> Instant {
        let silent = self.last_rx + self.timing.inactivity;
        self.nudge_due.map_or(silent, |due| due.min(silent))
    }

    /// Timer expiry: declare the receiver dead after
    /// [`Timing::inactivity`] of silence, or send a deferred nudge.
    pub(crate) fn on_tick(&mut self, now: Instant) -> Action {
        if now.duration_since(self.last_rx) >= self.timing.inactivity {
            return Action::End(SessionEnd::Failure(format!(
                "no message from the receiver for {}s (powered off or off the network)",
                self.timing.inactivity.as_secs_f32()
            )));
        }
        match self.nudge_due {
            Some(due) if now >= due => {
                self.nudge_due = None;
                self.last_nudge = now;
                Action::Nudge
            }
            _ => Action::Nothing,
        }
    }

    pub(crate) fn on_message(&mut self, msg: &CastMessage, now: Instant) -> Action {
        self.last_rx = now;
        if msg.destination_id != DEFAULT_SENDER_ID && msg.destination_id != "*" {
            debug!(dst = %msg.destination_id, "ignoring cast message for another sender");
            return Action::Nothing;
        }
        let payload: Option<GenericPayload> = msg
            .payload_utf8
            .as_deref()
            .and_then(|p| serde_json::from_str(p).ok());
        let ty = payload.as_ref().map_or("", |p| p.msg_type.as_str());
        let request_id = payload.as_ref().and_then(|p| p.request_id);

        match msg.namespace.as_str() {
            namespace::HEARTBEAT => match ty {
                "PING" => pong_reply(msg).map_or(Action::Nothing, Action::Reply),
                _ => Action::Nothing,
            },
            namespace::CONNECTION if ty == "CLOSE" => self.on_close(msg),
            namespace::RECEIVER => self.on_receiver(msg, ty, request_id, now),
            namespace::MEDIA => self.on_media(msg, ty, now),
            _ => Action::Nothing,
        }
    }

    fn on_close(&self, msg: &CastMessage) -> Action {
        let src = msg.source_id.as_str();
        match &self.phase {
            Phase::Launching { .. } => Action::End(SessionEnd::Failure(format!(
                "receiver closed the connection (src={src}) before the app launched; \
                 rebooting the chromecast clears stuck session state"
            ))),
            Phase::Running { .. } if src == DEFAULT_RECEIVER_ID => {
                error!("receiver-0 closed the platform connection");
                Action::End(SessionEnd::Failure(
                    "receiver closed the platform connection".into(),
                ))
            }
            Phase::Running { transport_id, .. } if src == transport_id => {
                self.app_gone("the receiver app closed our connection".into())
            }
            Phase::Running { .. } => Action::Nothing,
        }
    }

    fn on_receiver(
        &mut self,
        msg: &CastMessage,
        ty: &str,
        request_id: Option<i64>,
        now: Instant,
    ) -> Action {
        info!(
            payload = msg.payload_utf8.as_deref().unwrap_or("(none)"),
            "receiver namespace message"
        );
        match (&self.phase, ty) {
            (Phase::Launching { app_id, .. }, "RECEIVER_STATUS") => {
                match extract_app(msg, app_id) {
                    ExtractApp::Found {
                        transport_id,
                        session_id,
                    } => {
                        info!(%transport_id, %session_id, "receiver app fully launched");
                        self.phase = Phase::Running {
                            transport_id: transport_id.clone(),
                            session_id: session_id.clone(),
                        };
                        self.nudge_due = None;
                        Action::Launched {
                            transport_id,
                            session_id,
                        }
                    }
                    ExtractApp::Pending(reason) => self.request_nudge(&reason, now),
                    ExtractApp::Unparsable => Action::Nothing,
                }
            }
            (
                Phase::Launching {
                    launch_request_id, ..
                },
                "LAUNCH_ERROR" | "INVALID_REQUEST",
            ) if request_id.is_none_or(|id| id == *launch_request_id) => {
                Action::End(SessionEnd::Failure(format!(
                    "receiver refused to launch the app ({ty}): {}",
                    msg.payload_utf8.as_deref().unwrap_or("(no payload)")
                )))
            }
            (Phase::Running { session_id, .. }, "RECEIVER_STATUS") => match running_apps(msg) {
                Some(apps) if !apps.iter().any(|(_, s)| s == session_id) => {
                    let others: Vec<&str> = apps.iter().map(|(a, _)| a.as_str()).collect();
                    self.app_gone(format!(
                        "our cast session ended; the receiver now runs {others:?}"
                    ))
                }
                _ => Action::Nothing,
            },
            (_, "LAUNCH_ERROR" | "INVALID_REQUEST") => {
                warn!(ty, payload = ?msg.payload_utf8, "receiver rejected a request");
                Action::Nothing
            }
            _ => Action::Nothing,
        }
    }

    /// Send GET_STATUS now, or schedule it if the last one went out
    /// less than [`Timing::nudge_interval`] ago. Some firmwares emit a
    /// single half-populated RECEIVER_STATUS and go quiet, so the nudge
    /// is needed; answering every status immediately would ping-pong at
    /// round-trip speed for the whole launch.
    fn request_nudge(&mut self, reason: &str, now: Instant) -> Action {
        let earliest = self.last_nudge + self.timing.nudge_interval;
        if now >= earliest {
            info!(%reason, "launch ack pending, sending GET_STATUS");
            self.last_nudge = now;
            self.nudge_due = None;
            Action::Nudge
        } else {
            debug!(%reason, "launch ack pending, GET_STATUS deferred");
            self.nudge_due = Some(earliest);
            Action::Nothing
        }
    }

    fn on_media(&mut self, msg: &CastMessage, ty: &str, now: Instant) -> Action {
        let Phase::Running { transport_id, .. } = &self.phase else {
            return Action::Nothing;
        };
        if msg.source_id != *transport_id {
            debug!(src = %msg.source_id, "ignoring media message from another app");
            return Action::Nothing;
        }
        match ty {
            "LOAD_FAILED" | "INVALID_REQUEST" | "INVALID_PLAYER_STATE" | "ERROR" => {
                log_media_error(msg, ty);
                Action::End(SessionEnd::Failure(format!(
                    "receiver rejected our media ({ty})"
                )))
            }
            "LOAD_CANCELLED" => Action::End(SessionEnd::Remote(
                "another sender replaced our media".into(),
            )),
            "MEDIA_STATUS" => {
                let parsed = msg.parse_payload::<MediaStatusPayload>();
                match parsed {
                    Ok(p) => self
                        .media
                        .on_status(&p, now)
                        .map_or(Action::Nothing, Action::End),
                    Err(e) => {
                        warn!(%e, payload = ?msg.payload_utf8, "MEDIA_STATUS parse failed");
                        Action::Nothing
                    }
                }
            }
            _ => {
                debug!(ty, payload = ?msg.payload_utf8, "chromecast media message");
                Action::Nothing
            }
        }
    }

    /// Our app disappeared. Usually the user pressed "Stop casting" or
    /// another app took over, which must not trigger a reconnect. The
    /// Default Media Receiver also exits by itself after sustained
    /// BUFFERING; that one is a failure worth retrying.
    fn app_gone(&self, why: String) -> Action {
        if self.media.last_state.as_deref() == Some("BUFFERING") {
            Action::End(SessionEnd::Failure(format!("{why} while buffering")))
        } else {
            Action::End(SessionEnd::Remote(why))
        }
    }
}

enum ExtractApp {
    Found {
        transport_id: String,
        session_id: String,
    },
    /// A status arrived but our app is missing or half-populated.
    Pending(String),
    Unparsable,
}

fn extract_app(msg: &CastMessage, app_id: &str) -> ExtractApp {
    let payload: ReceiverStatusPayload = match msg.parse_payload() {
        Ok(p) => p,
        Err(e) => {
            warn!(%e, payload = ?msg.payload_utf8, "RECEIVER_STATUS parse failed");
            return ExtractApp::Unparsable;
        }
    };
    let Some(status) = payload.status else {
        return ExtractApp::Pending("status block absent".into());
    };
    let Some(app) = status
        .applications
        .iter()
        .find(|a| a.app_id.eq_ignore_ascii_case(app_id))
    else {
        return ExtractApp::Pending(format!(
            "our app not in applications list ({} entries)",
            status.applications.len()
        ));
    };
    if app.transport_id.is_empty() || app.session_id.is_empty() {
        return ExtractApp::Pending(format!(
            "app entry incomplete (transport_id='{}', session_id='{}')",
            app.transport_id, app.session_id
        ));
    }
    ExtractApp::Found {
        transport_id: app.transport_id.clone(),
        session_id: app.session_id.clone(),
    }
}

/// `(appId, sessionId)` of every running app, or `None` when the
/// message carries no status block. A status without an `applications`
/// list means nothing runs: that is how receivers report the idle
/// screen.
fn running_apps(msg: &CastMessage) -> Option<Vec<(String, String)>> {
    let payload: ReceiverStatusPayload = msg.parse_payload().ok()?;
    let status = payload.status?;
    Some(
        status
            .applications
            .into_iter()
            .map(|a| (a.app_id, a.session_id))
            .collect(),
    )
}

fn log_media_error(msg: &CastMessage, ty: &str) {
    let detailed_raw = msg
        .payload_utf8
        .as_deref()
        .and_then(|p| serde_json::from_str::<serde_json::Value>(p).ok())
        .and_then(|v| {
            v.get("detailedErrorCode")
                .and_then(serde_json::Value::as_i64)
        });
    let detailed = detailed_raw.map(MediaDetailedErrorCode::from_code);
    error!(
        ty,
        detailed_code = ?detailed_raw,
        detailed = ?detailed,
        retryable = detailed.map(MediaDetailedErrorCode::is_retryable),
        payload = ?msg.payload_utf8,
        "chromecast rejected our media (session is terminal)"
    );
    match detailed {
        Some(MediaDetailedErrorCode::SegmentNetwork) => error!(
            "SegmentNetwork (301): receiver gave up fetching a segment. \
             On 1st/2nd-gen Chromecast this is firmware-internal — the link \
             is usually fine. Auto-reconnect supervisor will rebuild the \
             session; if it loops, lower StreamConfig::bitrate_kbps or use \
             newer Cast hardware."
        ),
        Some(MediaDetailedErrorCode::MediaSrcNotSupported) => error!(
            "MediaSrcNotSupported (104): the receiver's decoder rejected the \
             bitstream. Check `max_h264_profile` in capabilities_for_model() \
             — older Chromecasts choke on High profile."
        ),
        Some(MediaDetailedErrorCode::LoadFailed) => error!(
            "LoadFailed (905): receiver couldn't start the load. Check that \
             the HLS URL we advertised is reachable from the device's network \
             (the `HLS self-test OK` line on startup should confirm)."
        ),
        _ => {}
    }
}

/// MEDIA_STATUS bookkeeping: decides when playback ended for good and
/// keeps the diagnostics that make the log a readable timeline.
#[derive(Debug)]
struct MediaTracker {
    /// Set once the receiver reported PLAYING / PAUSED / BUFFERING. The
    /// IDLE/LOADING statuses before that are part of a normal load.
    playback_started: bool,
    last_state: Option<String>,
    state_entered_at: Instant,
    last_current_time: Option<f64>,
    last_status_at: Instant,
}

impl Default for MediaTracker {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            playback_started: false,
            last_state: None,
            state_entered_at: now,
            last_current_time: None,
            last_status_at: now,
        }
    }
}

impl MediaTracker {
    fn on_status(&mut self, payload: &MediaStatusPayload, now: Instant) -> Option<SessionEnd> {
        let s0 = payload.status.first()?;
        let new_state = s0.player_state.clone();
        self.log_status(s0, new_state.as_deref(), now);

        let verdict = match (new_state.as_deref(), s0.idle_reason.as_deref()) {
            (Some("PLAYING" | "PAUSED" | "BUFFERING"), _) => {
                self.playback_started = true;
                None
            }
            (Some("IDLE"), Some("ERROR")) => Some(SessionEnd::Failure(
                "receiver stopped playback with an error".into(),
            )),
            (Some("IDLE"), Some(reason @ ("CANCELLED" | "INTERRUPTED")))
                if self.playback_started =>
            {
                Some(SessionEnd::Remote(format!(
                    "playback ended on the receiver ({reason})"
                )))
            }
            (Some("IDLE"), Some("FINISHED")) if self.playback_started => Some(SessionEnd::Failure(
                "receiver considers the live stream finished".into(),
            )),
            _ => None,
        };
        if new_state != self.last_state {
            self.state_entered_at = now;
            self.last_state = new_state;
        }
        verdict
    }

    /// Log one status. State changes go out at info!/warn!, repeats at
    /// debug!. Derived values:
    /// * `advance_s`: currentTime progress since the previous status;
    ///   collapses to ~0 when the receiver silently buffers.
    /// * `lag_s`: distance from the live edge; growing without bound
    ///   means a 301 is coming.
    /// * `window_s`: playable history the receiver sees.
    fn log_status(
        &mut self,
        s0: &crate::castv2::MediaStatus,
        new_state: Option<&str>,
        now: Instant,
    ) {
        let wall_delta = now.duration_since(self.last_status_at).as_secs_f64();
        self.last_status_at = now;
        let current_time = s0.current_time;
        let advance_s = current_time.zip(self.last_current_time).map(|(c, p)| c - p);
        let seekable = s0.live_seekable_range.as_ref();
        let lag_s = current_time.zip(seekable).map(|(c, r)| r.end - c);
        let window_s = seekable.map(|r| r.end - r.start);
        if current_time.is_some() {
            self.last_current_time = current_time;
        }
        let changed = new_state != self.last_state.as_deref();
        let prev_state_ms = changed.then(|| now.duration_since(self.state_entered_at).as_millis());

        macro_rules! emit {
            ($level:ident, $msg:literal) => {
                tracing::$level!(
                    state = ?new_state,
                    prev_state = ?self.last_state,
                    ?prev_state_ms,
                    ?current_time,
                    ?advance_s,
                    wall_delta_s = format_args!("{wall_delta:.2}"),
                    ?lag_s,
                    ?window_s,
                    idle_reason = ?s0.idle_reason,
                    extended_status = ?s0.extended_status,
                    playback_rate = ?s0.playback_rate,
                    $msg
                )
            };
        }
        match (changed, new_state) {
            (true, Some("BUFFERING")) => emit!(warn, "MEDIA_STATUS: chromecast entered BUFFERING"),
            (true, Some("IDLE")) => emit!(warn, "MEDIA_STATUS: chromecast entered IDLE"),
            (true, _) => emit!(info, "MEDIA_STATUS: state transition"),
            (false, _) => emit!(debug, "MEDIA_STATUS: steady"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::castv2::{DEFAULT_MEDIA_RECEIVER_APP_ID, ping_message};
    use serde_json::json;

    const LAUNCH_ID: i64 = 2;
    const TRANSPORT: &str = "transport-1";
    const SESSION: &str = "session-1";

    fn msg(ns: &str, src: &str, payload: &serde_json::Value) -> CastMessage {
        CastMessage::new_json(src, DEFAULT_SENDER_ID, ns, payload).unwrap()
    }

    fn receiver_status(apps: &serde_json::Value) -> CastMessage {
        msg(
            namespace::RECEIVER,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "RECEIVER_STATUS", "requestId": 0, "status": {"applications": apps}}),
        )
    }

    fn our_app() -> serde_json::Value {
        json!([{"appId": DEFAULT_MEDIA_RECEIVER_APP_ID, "transportId": TRANSPORT, "sessionId": SESSION}])
    }

    fn media_status(state: &str, idle_reason: Option<&str>) -> CastMessage {
        let mut s = json!({"mediaSessionId": 1, "playerState": state});
        if let Some(r) = idle_reason {
            s["idleReason"] = json!(r);
        }
        msg(
            namespace::MEDIA,
            TRANSPORT,
            &json!({"type": "MEDIA_STATUS", "status": [s]}),
        )
    }

    fn launching(now: Instant) -> ReceiverMonitor {
        ReceiverMonitor::launching(
            DEFAULT_MEDIA_RECEIVER_APP_ID,
            LAUNCH_ID,
            Timing::default(),
            now,
        )
    }

    fn running(now: Instant) -> ReceiverMonitor {
        let mut m = launching(now);
        assert!(matches!(
            m.on_message(&receiver_status(&our_app()), now),
            Action::Launched { .. }
        ));
        m
    }

    fn ended(action: Action) -> SessionEnd {
        match action {
            Action::End(end) => end,
            other => panic!("expected End, got {other:?}"),
        }
    }

    #[test]
    fn launch_resolves_on_complete_app_entry() {
        let now = Instant::now();
        let mut m = launching(now);
        match m.on_message(&receiver_status(&our_app()), now) {
            Action::Launched {
                transport_id,
                session_id,
            } => {
                assert_eq!(transport_id, TRANSPORT);
                assert_eq!(session_id, SESSION);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn pending_status_nudges_are_rate_limited() {
        let t0 = Instant::now();
        let mut m = launching(t0);
        let empty = receiver_status(&json!([]));
        // Right after the GET_STATUS sent with LAUNCH: defer.
        assert!(matches!(m.on_message(&empty, t0), Action::Nothing));
        assert_eq!(m.next_wakeup(), t0 + Duration::from_millis(500));
        // A storm of statuses within the window sends nothing.
        for ms in [10, 50, 200, 499] {
            let t = t0 + Duration::from_millis(ms);
            assert!(matches!(m.on_message(&empty, t), Action::Nothing));
        }
        // The deferred nudge fires on the timer.
        assert!(matches!(
            m.on_tick(t0 + Duration::from_millis(500)),
            Action::Nudge
        ));
        assert!(matches!(
            m.on_tick(t0 + Duration::from_millis(600)),
            Action::Nothing
        ));
        // After the interval a status nudges immediately.
        assert!(matches!(
            m.on_message(&empty, t0 + Duration::from_millis(1001)),
            Action::Nudge
        ));
    }

    #[test]
    fn launch_error_fails_immediately() {
        let now = Instant::now();
        let mut m = launching(now);
        let err = msg(
            namespace::RECEIVER,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "LAUNCH_ERROR", "requestId": LAUNCH_ID, "reason": "NOT_FOUND"}),
        );
        let SessionEnd::Failure(why) = ended(m.on_message(&err, now)) else {
            panic!("launch error must be a failure");
        };
        assert!(why.contains("NOT_FOUND"));
    }

    #[test]
    fn invalid_request_for_other_request_is_ignored() {
        let now = Instant::now();
        let mut m = launching(now);
        let other = msg(
            namespace::RECEIVER,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "INVALID_REQUEST", "requestId": 99, "reason": "INVALID_COMMAND"}),
        );
        assert!(matches!(m.on_message(&other, now), Action::Nothing));
        let ours = msg(
            namespace::RECEIVER,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "INVALID_REQUEST", "requestId": LAUNCH_ID}),
        );
        assert!(matches!(m.on_message(&ours, now), Action::End(_)));
    }

    #[test]
    fn ping_is_answered_with_pong_to_sender() {
        let now = Instant::now();
        let mut m = running(now);
        let mut ping = ping_message().unwrap();
        ping.source_id = DEFAULT_RECEIVER_ID.into();
        ping.destination_id = DEFAULT_SENDER_ID.into();
        match m.on_message(&ping, now) {
            Action::Reply(pong) => {
                assert_eq!(pong.message_type().as_deref(), Some("PONG"));
                assert_eq!(pong.destination_id, DEFAULT_RECEIVER_ID);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn silence_past_inactivity_timeout_ends_session() {
        let t0 = Instant::now();
        let mut m = running(t0);
        assert_eq!(m.next_wakeup(), t0 + Duration::from_secs(15));
        assert!(matches!(
            m.on_tick(t0 + Duration::from_secs(14)),
            Action::Nothing
        ));
        // Any message (a PONG here) re-arms the timer.
        let pong = msg(
            namespace::HEARTBEAT,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "PONG"}),
        );
        let t1 = t0 + Duration::from_secs(10);
        assert!(matches!(m.on_message(&pong, t1), Action::Nothing));
        assert!(matches!(
            m.on_tick(t0 + Duration::from_secs(16)),
            Action::Nothing
        ));
        assert!(matches!(
            ended(m.on_tick(t1 + Duration::from_secs(15))),
            SessionEnd::Failure(_)
        ));
    }

    #[test]
    fn another_app_taking_over_is_a_remote_stop() {
        let now = Instant::now();
        let mut m = running(now);
        // Same app, unrelated status (volume change): keep going.
        assert!(matches!(
            m.on_message(&receiver_status(&our_app()), now),
            Action::Nothing
        ));
        let youtube = receiver_status(
            &json!([{"appId": "233637DE", "transportId": "t-yt", "sessionId": "s-yt"}]),
        );
        assert!(matches!(
            ended(m.on_message(&youtube, now)),
            SessionEnd::Remote(_)
        ));
    }

    #[test]
    fn idle_screen_after_stop_casting_is_a_remote_stop() {
        let now = Instant::now();
        let mut m = running(now);
        let idle = msg(
            namespace::RECEIVER,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "RECEIVER_STATUS", "status": {"volume": {"level": 0.5}}}),
        );
        assert!(matches!(
            ended(m.on_message(&idle, now)),
            SessionEnd::Remote(_)
        ));
    }

    #[test]
    fn status_without_status_block_is_ignored() {
        let now = Instant::now();
        let mut m = running(now);
        let bare = msg(
            namespace::RECEIVER,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "RECEIVER_STATUS"}),
        );
        assert!(matches!(m.on_message(&bare, now), Action::Nothing));
    }

    #[test]
    fn app_close_is_remote_unless_buffering() {
        let now = Instant::now();
        let close = msg(namespace::CONNECTION, TRANSPORT, &json!({"type": "CLOSE"}));

        let mut m = running(now);
        assert!(matches!(
            m.on_message(&media_status("PLAYING", None), now),
            Action::Nothing
        ));
        assert!(matches!(
            ended(m.on_message(&close, now)),
            SessionEnd::Remote(_)
        ));

        let mut m = running(now);
        assert!(matches!(
            m.on_message(&media_status("BUFFERING", None), now),
            Action::Nothing
        ));
        assert!(matches!(
            ended(m.on_message(&close, now)),
            SessionEnd::Failure(_)
        ));
    }

    #[test]
    fn platform_close_is_a_failure() {
        let now = Instant::now();
        let mut m = running(now);
        let close = msg(
            namespace::CONNECTION,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "CLOSE"}),
        );
        assert!(matches!(
            ended(m.on_message(&close, now)),
            SessionEnd::Failure(_)
        ));
    }

    #[test]
    fn media_from_other_apps_is_ignored() {
        let now = Instant::now();
        let mut m = running(now);
        let foreign = msg(
            namespace::MEDIA,
            "someone-else",
            &json!({"type": "LOAD_FAILED", "requestId": 5}),
        );
        assert!(matches!(m.on_message(&foreign, now), Action::Nothing));
        let not_for_us = CastMessage::new_json(
            TRANSPORT,
            "sender-other",
            namespace::MEDIA,
            &json!({"type": "LOAD_FAILED"}),
        )
        .unwrap();
        assert!(matches!(m.on_message(&not_for_us, now), Action::Nothing));
    }

    #[test]
    fn initial_idle_and_loading_are_exempt() {
        let now = Instant::now();
        let mut m = running(now);
        for status in [
            media_status("IDLE", None),
            media_status("LOADING", None),
            media_status("IDLE", Some("FINISHED")),
            media_status("IDLE", Some("INTERRUPTED")),
        ] {
            assert!(matches!(m.on_message(&status, now), Action::Nothing));
        }
    }

    #[test]
    fn idle_with_error_is_always_terminal() {
        let now = Instant::now();
        let mut m = running(now);
        assert!(matches!(
            ended(m.on_message(&media_status("IDLE", Some("ERROR")), now)),
            SessionEnd::Failure(_)
        ));
    }

    #[test]
    fn idle_after_playback_started_is_terminal() {
        let now = Instant::now();
        let cases = [
            ("INTERRUPTED", true),
            ("CANCELLED", true),
            ("FINISHED", false),
        ];
        for (reason, remote) in cases {
            let mut m = running(now);
            assert!(matches!(
                m.on_message(&media_status("PLAYING", None), now),
                Action::Nothing
            ));
            let end = ended(m.on_message(&media_status("IDLE", Some(reason)), now));
            assert_eq!(matches!(end, SessionEnd::Remote(_)), remote, "{reason}");
        }
    }

    #[test]
    fn unknown_player_state_is_tolerated() {
        let now = Instant::now();
        let mut m = running(now);
        assert!(matches!(
            m.on_message(&media_status("SOMETHING_NEW", Some("??")), now),
            Action::Nothing
        ));
    }
}
