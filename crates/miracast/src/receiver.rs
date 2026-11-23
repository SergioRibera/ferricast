//! Miracast sink: advertises as an MS-MICE display and runs the
//! sink-side RTSP state machine (accept M1/M2, respond to M3/M4/M5,
//! initiate M6/M7, drain RTP MP2T into the demuxer).
//!
//! The `ReceiverProtocol` impl plugs into ferricast-core's push-media
//! flow: on M7 PLAY the control session emits
//! [`MediaCommand::Push`](ferricast_core::MediaCommand) carrying a
//! [`PushedStream`](ferricast_core::push::PushedStream) fed by the RTP
//! depacketiser. The existing receiver pipeline then decodes + renders
//! exactly like the AirPlay mirroring path.
//!
//! Status: scaffolding — `advertise_info` + ReceiverProtocol wiring
//! are in place; the RTSP sink state machine and the RTP → decoder
//! fan-out are tracked as follow-up tasks. The crate compiles with
//! the receiver registered so the GUI can start showing the Miracast
//! sink as discoverable while we finish the state machine.

use std::collections::HashMap;

use bytes::Bytes;
use tokio::sync::mpsc;

use ferricast_core::{
    AdvertiseInfo, Advertiser, Codec, ControlSession, DeviceCapabilities, FerricastError,
    MdnsAdvertiser, MediaCommand, MediaPuller, PlaybackState, ReceiverProtocol, RemoteSender,
    Result,
};

use crate::transport::mice::MICE_SERVICE_TYPE;

pub struct MiracastReceiver {
    pub friendly_name: String,
    pub device_id: String,
}

impl MiracastReceiver {
    #[must_use]
    pub fn new(friendly_name: impl Into<String>, device_id: impl Into<String>) -> Self {
        Self {
            friendly_name: friendly_name.into(),
            device_id: device_id.into(),
        }
    }
}

impl ReceiverProtocol for MiracastReceiver {
    const PROTOCOL: &'static str = "miracast";
    const SUPPORTED_CODECS: &'static [Codec] = &[Codec::H264];

    type Advertiser = MiracastReceiverAdvertiser;
    type Control = MiracastReceiverControl;
    type Puller = MiracastPuller;

    fn create_advertiser(&self) -> Self::Advertiser {
        MiracastReceiverAdvertiser::default()
    }

    fn create_control(&self) -> Result<Self::Control> {
        Ok(MiracastReceiverControl::default())
    }

    fn create_puller(&self) -> Result<Self::Puller> {
        Ok(MiracastPuller::default())
    }

    fn advertise_info(&self) -> AdvertiseInfo {
        let mut txt = HashMap::new();
        // MS-MICE marker: Windows "Cast to" picker filters by mice=1.
        txt.insert("mice".to_string(), "1".to_string());
        txt.insert("rtsp_port".to_string(), "7236".to_string());
        txt.insert("txtvers".to_string(), "1".to_string());
        txt.insert("fn".to_string(), self.friendly_name.clone());
        AdvertiseInfo {
            friendly_name: self.friendly_name.clone(),
            port: 7250,
            capabilities: DeviceCapabilities {
                supported_codecs: vec![Codec::H264],
                max_width: Some(1920),
                max_height: Some(1080),
                max_fps: Some(30),
                ..DeviceCapabilities::default()
            },
            device_id: self.device_id.clone(),
            icon: Bytes::new(),
            txt,
        }
    }
}

#[derive(Default)]
pub struct MiracastReceiverAdvertiser {
    daemon: Option<mdns_sd::ServiceDaemon>,
    fullname: Option<String>,
}

impl Advertiser for MiracastReceiverAdvertiser {
    const PROTOCOL: &'static str = "miracast";

    async fn start(&mut self, info: AdvertiseInfo) -> Result<()> {
        let daemon = mdns_sd::ServiceDaemon::new()
            .map_err(|e| FerricastError::Receiver(format!("mdns-sd: {e}")))?;
        let host = gethostname();
        let service = mdns_sd::ServiceInfo::new(
            MICE_SERVICE_TYPE,
            &info.friendly_name,
            &format!("{host}.local."),
            "",
            info.port,
            info.txt.clone(),
        )
        .map_err(|e| FerricastError::Receiver(format!("mdns service: {e}")))?
        .enable_addr_auto();
        let fullname = service.get_fullname().to_string();
        daemon
            .register(service)
            .map_err(|e| FerricastError::Receiver(format!("mdns register: {e}")))?;
        self.daemon = Some(daemon);
        self.fullname = Some(fullname);
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let (Some(daemon), Some(fullname)) = (self.daemon.take(), self.fullname.take()) {
            let _ = daemon.unregister(&fullname);
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.daemon.is_some()
    }
}

impl MdnsAdvertiser for MiracastReceiverAdvertiser {
    const SERVICE_TYPE: &'static str = MICE_SERVICE_TYPE;
}

fn gethostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| hostname::get().ok().and_then(|h| h.into_string().ok()))
        .unwrap_or_else(|| "ferricast".to_string())
}

// `hostname` isn't a dep; fall back to env.
mod hostname {
    pub fn get() -> std::io::Result<std::ffi::OsString> {
        Err(std::io::Error::other("no hostname crate available"))
    }
}

/// Sink-side control session. Accepts the RTSP TCP, drives the M1-M7
/// handshake with the source, and once PLAY arrives emits a
/// `MediaCommand::Push` with the RTP stream.
///
/// Scaffolding: today `accept` blocks forever and `next_command`
/// returns an Err so the trait compiles. The real sink state machine
/// (listening TCP 7236, reading M1, replying M3 with offered formats,
/// driving M6/M7 to the source once it sends M5 trigger) is the next
/// milestone.
#[derive(Default)]
pub struct MiracastReceiverControl {
    alive: bool,
}

impl ControlSession for MiracastReceiverControl {
    async fn accept(&mut self) -> Result<RemoteSender> {
        Err(FerricastError::Receiver(
            "miracast sink state machine not yet implemented".into(),
        ))
    }

    async fn next_command(&mut self) -> Result<MediaCommand> {
        Err(FerricastError::Receiver(
            "miracast sink state machine not yet implemented".into(),
        ))
    }

    async fn report_state(&mut self, _state: PlaybackState) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.alive = false;
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.alive
    }
}

/// Stub puller — Miracast is push-only, so once the state machine is
/// finished this type delegates to a `PushedStream` built from the
/// RTP depacketiser.
#[derive(Default)]
pub struct MiracastPuller;

impl MediaPuller for MiracastPuller {
    async fn open(
        &mut self,
        _spec: ferricast_core::PullSpec,
    ) -> Result<ferricast_core::MediaInfo> {
        Err(FerricastError::Pull(
            "miracast puller requires PushedStream delegation (pending state machine)".into(),
        ))
    }

    async fn next(&mut self) -> Result<ferricast_core::MediaPacket> {
        Err(FerricastError::Pull("miracast puller not ready".into()))
    }

    async fn seek(&mut self, _position_us: u64) -> Result<()> {
        Err(FerricastError::Pull("miracast is not seekable".into()))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

#[allow(dead_code, reason = "channel shared between sink RTSP task and push stream")]
struct PushBridge {
    tx: mpsc::Sender<ferricast_core::MediaPacket>,
}
