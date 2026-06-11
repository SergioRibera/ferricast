//! Receiver-side control plane — accept a remote sender (phone,
//! laptop) and exchange playback commands with it.
//!
//! Separate from the media plane on purpose: every "receiver" protocol
//! we care about (Chromecast, AirPlay, DIAL/YouTube launch) is really
//! two channels — a control connection that carries commands like
//! "LOAD this URL, then PAUSE at 30s" and an *unrelated* media fetch
//! that the receiver performs once it gets the URL. The control
//! session in this module is the first half; pulling and decoding the
//! actual media is [`crate::puller`] + [`crate::decoder`].
//!
//! Chromecast in particular: the sender app on the phone sends LOAD
//! over CASTV2 with an `http(s)://…/.m3u8` URL, and we're expected to
//! HLS-pull it ourselves. So a Chromecast receiver impl wires a
//! `ControlSession` (this trait) to an `HlsPuller`
//! ([`crate::puller::MediaPuller`]) on receipt of `Load`.

use std::collections::HashMap;
use std::net::IpAddr;

use uuid::Uuid;

use crate::error::Result;

/// The remote device that connected to us (the "sender" in
/// Chromecast/AirPlay terminology — confusingly, when *we* are the
/// receiver, the sender is whoever pressed Cast on their phone).
#[derive(Debug, Clone)]
pub struct RemoteSender {
    /// Stable identifier the remote presented at connection setup,
    /// normalised into a UUID. Falls back to a freshly generated
    /// UUID when the protocol doesn't carry one.
    pub id: Uuid,
    /// IP the control connection came in from. Receivers report it
    /// so application code can correlate with mDNS / firewall logs.
    pub addr: IpAddr,
    /// Friendly name the sender app advertised, when available
    /// (Chromecast: sender app's display name; AirPlay: device name).
    pub name: Option<String>,
}

/// Coarse playback state the receiver reports back upstream.
/// Maps cleanly onto Chromecast's `MEDIA_STATUS.playerState` and
/// AirPlay's `playbackState`.
#[derive(Debug, Clone)]
pub enum PlaybackState {
    Idle,
    Buffering,
    Playing,
    Paused,
    Ended,
    Error(String),
}

/// Repeat mode for queued playback. Matches Chromecast
/// `REPEAT_MODE_*` and AirPlay equivalents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatMode {
    Off,
    All,
    Single,
    AllAndShuffle,
}

/// One item in a receiver-side play queue. Senders push these via
/// `QueueLoad` / `QueueInsert`; the receiver runs through them in
/// order, respecting `RepeatMode`.
#[derive(Debug, Clone)]
pub struct QueueItem {
    /// Sender-assigned id, opaque to us. Echoed back in status so
    /// the sender can correlate; receiver MUST NOT reinterpret it.
    pub item_id: i64,
    pub url: String,
    pub content_type: Option<String>,
    pub start_time_us: Option<u64>,
    pub duration_us: Option<u64>,
    pub autoplay: bool,
    /// Arbitrary sender-supplied metadata (title, artwork URL, etc).
    /// Passed through to the GUI for display; receiver doesn't
    /// interpret it.
    pub metadata: HashMap<String, String>,
}

/// Subtitle / alternate-audio track selection. Chromecast
/// `EDIT_TRACKS_INFO` payload.
#[derive(Debug, Clone)]
pub struct TrackSelection {
    /// Track ids (sender-assigned, opaque) the receiver should
    /// enable. Anything not in this list is disabled.
    pub active_track_ids: Vec<i64>,
}

/// Commands a remote sender can issue. Closed enum — every protocol
/// command we accept must be representable here so application code
/// can switch over it exhaustively. Adding a new variant is a
/// breaking change on purpose.
///
/// Coverage target: full Chromecast media namespace
/// (`urn:x-cast:com.google.cast.media`) + receiver namespace
/// (`urn:x-cast:com.google.cast.receiver`) volume/launch. AirPlay
/// and DIAL commands map onto the same set.
#[derive(Debug, Clone)]
pub enum MediaCommand {
    /// Load a media URL and (optionally) start playing it.
    /// Chromecast `LOAD`; AirPlay `play` HTTP request.
    Load {
        url: String,
        content_type: Option<String>,
        start_time_us: Option<u64>,
        autoplay: bool,
        metadata: HashMap<String, String>,
    },
    /// Resume playback.
    Play,
    /// Pause without tearing down the media session.
    Pause,
    /// Stop playback and unload the media. After this the receiver
    /// returns to `Idle`.
    Stop,
    /// Seek to an absolute position from the start of the media.
    Seek { position_us: u64 },
    /// Set per-stream playback volume. `0.0` = mute, `1.0` = unity.
    SetVolume(f32),
    /// Mute / unmute without changing the volume level.
    SetMute(bool),
    /// Set system-level (receiver-wide) volume. Distinct from
    /// `SetVolume` which only affects the active stream.
    SetSystemVolume(f32),
    /// Set system-level mute.
    SetSystemMute(bool),
    /// Sender polled for status. Receiver replies via
    /// [`ControlSession::report_state`] without changing playback.
    GetStatus,
    /// Replace the queue with this list of items.
    /// Chromecast `QUEUE_LOAD`.
    QueueLoad {
        items: Vec<QueueItem>,
        start_index: u32,
        repeat_mode: RepeatMode,
    },
    /// Insert items into the queue. `insert_before` references an
    /// existing `item_id`; `None` appends.
    QueueInsert {
        items: Vec<QueueItem>,
        insert_before: Option<i64>,
    },
    /// Update an existing queue item in place (e.g. retitle).
    QueueUpdate { items: Vec<QueueItem> },
    /// Remove items by id.
    QueueRemove { item_ids: Vec<i64> },
    /// Reorder items. `item_ids` is the new ordering, anchored
    /// before `insert_before`.
    QueueReorder {
        item_ids: Vec<i64>,
        insert_before: Option<i64>,
    },
    /// Skip to the next queue item.
    QueueNext,
    /// Go to the previous queue item.
    QueuePrev,
    /// Jump to a specific queue item by id.
    QueueJump { item_id: i64 },
    /// Change the queue's repeat mode without reloading.
    QueueSetRepeat(RepeatMode),
    /// Update which subtitle / alternate-audio tracks are active.
    /// Chromecast `EDIT_TRACKS_INFO`.
    EditTracks(TrackSelection),
    /// Change playback rate. `1.0` = normal, `2.0` = 2x. Receivers
    /// MAY clamp to a supported range.
    SetPlaybackRate(f32),
    /// Preload media for gapless transition without starting it.
    /// Chromecast `PRELOAD`.
    Preload {
        url: String,
        content_type: Option<String>,
    },
    /// Launch a receiver application by id. Chromecast `LAUNCH`
    /// (`appId` payload); default media receiver = `CC1AD845`.
    LaunchApp { app_id: String },
    /// Stop the currently running receiver application.
    /// Chromecast receiver namespace `STOP`.
    StopApp,
}

/// Bidirectional control channel with one remote sender. Lifetime:
///
/// 1. `accept` — block until a sender opens the control connection
///    and completes the protocol's handshake.
/// 2. Loop: `next_command` to receive a [`MediaCommand`],
///    `report_state` to push status back.
/// 3. `close` — tear down the connection.
///
/// Single-sender at a time on purpose. Cast / AirPlay both serialise
/// to one controller per receiver session; multi-controller scenarios
/// are protocol-specific and out of scope here.
pub trait ControlSession: Send + Sync {
    fn accept(&mut self) -> impl Future<Output = Result<RemoteSender>> + Send;
    fn next_command(&mut self) -> impl Future<Output = Result<MediaCommand>> + Send;
    fn report_state(&mut self, state: PlaybackState) -> impl Future<Output = Result<()>> + Send;
    fn close(&mut self) -> impl Future<Output = Result<()>> + Send;
    fn is_alive(&self) -> bool;
}
