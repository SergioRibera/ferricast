//! Receiver-side media a remote sender *pushes* to us, as opposed to a
//! URL the receiver pulls ([`crate::puller`]).
//!
//! AirPlay screen mirroring is the canonical case: the sender opens the
//! media connections itself (a TCP stream of H.264 access units, UDP
//! audio packets) once the control session set them up. The protocol
//! crate depacketizes and decrypts that traffic, and hands the receiver
//! pipeline a [`PushedStream`] through [`crate::MediaCommand::Push`].
//! The pipeline then consumes it exactly like a pulled stream: a
//! [`PushedStream`] is a [`MediaPuller`] whose packets come from a
//! channel instead of the network.

use std::fmt;

use tokio::sync::mpsc;

use crate::error::{FerricastError, Result};
use crate::puller::{MediaInfo, MediaPacket, MediaPuller, PullSpec};

/// A live stream fed by the protocol through a [`PushSender`].
///
/// [`MediaPuller::open`] ignores the [`PullSpec`] and returns the
/// [`MediaInfo`] the stream was created with. When every sender is
/// dropped the stream ends with [`MediaPacket::Eos`]. Seeking is not
/// possible.
pub struct PushedStream {
    info: MediaInfo,
    packets: mpsc::Receiver<MediaPacket>,
    ended: bool,
}

/// Producer half of a [`PushedStream`].
#[derive(Clone)]
pub struct PushSender {
    packets: mpsc::Sender<MediaPacket>,
}

/// Why [`PushSender::try_push`] did not queue a packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushError {
    /// The consumer is behind and the queue is full; the packet was dropped.
    Full,
    /// The consumer is gone.
    Closed,
}

impl PushedStream {
    /// A stream described by `info` with room for `capacity` queued
    /// packets, and the sender that feeds it.
    #[must_use]
    pub fn channel(info: MediaInfo, capacity: usize) -> (PushSender, Self) {
        let (tx, rx) = mpsc::channel(capacity.max(1));
        (
            PushSender { packets: tx },
            Self {
                info,
                packets: rx,
                ended: false,
            },
        )
    }

    /// What the stream carries.
    #[must_use]
    pub fn info(&self) -> &MediaInfo {
        &self.info
    }
}

impl fmt::Debug for PushedStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PushedStream")
            .field("info", &self.info)
            .field("ended", &self.ended)
            .finish_non_exhaustive()
    }
}

impl PushSender {
    /// Queues `packet` without waiting.
    ///
    /// # Errors
    ///
    /// [`PushError::Full`] when the consumer has fallen behind,
    /// [`PushError::Closed`] once the stream was dropped.
    pub fn try_push(&self, packet: MediaPacket) -> std::result::Result<(), PushError> {
        self.packets.try_send(packet).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => PushError::Full,
            mpsc::error::TrySendError::Closed(_) => PushError::Closed,
        })
    }

    /// Whether the consuming stream was dropped.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.packets.is_closed()
    }
}

impl MediaPuller for PushedStream {
    async fn open(&mut self, _spec: PullSpec) -> Result<MediaInfo> {
        Ok(self.info.clone())
    }

    async fn next(&mut self) -> Result<MediaPacket> {
        if self.ended {
            return Err(FerricastError::Pull("pushed stream already ended".into()));
        }
        match self.packets.recv().await {
            Some(MediaPacket::Eos) | None => {
                self.ended = true;
                Ok(MediaPacket::Eos)
            }
            Some(packet) => Ok(packet),
        }
    }

    async fn seek(&mut self, _position_us: u64) -> Result<()> {
        Err(FerricastError::Pull(
            "a live pushed stream cannot seek".into(),
        ))
    }

    async fn close(&mut self) -> Result<()> {
        self.packets.close();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::AudioCodec;
    use crate::puller::AudioStreamInfo;

    fn info() -> MediaInfo {
        MediaInfo {
            video: None,
            audio: Some(AudioStreamInfo {
                codec: AudioCodec::AacEld,
                sample_rate: 44_100,
                channels: 2,
                source: None,
            }),
            duration_us: None,
            is_live: true,
        }
    }

    #[tokio::test]
    async fn packets_flow_until_the_sender_goes() {
        let (tx, mut stream) = PushedStream::channel(info(), 2);
        let spec = PullSpec {
            url: String::new(),
            headers: std::collections::HashMap::new(),
        };
        assert!(stream.open(spec).await.unwrap().is_live);
        tx.try_push(MediaPacket::Discontinuity).unwrap();
        tx.try_push(MediaPacket::Discontinuity).unwrap();
        assert_eq!(
            tx.try_push(MediaPacket::Discontinuity),
            Err(PushError::Full)
        );
        assert!(matches!(
            stream.next().await.unwrap(),
            MediaPacket::Discontinuity
        ));
        drop(tx);
        assert!(matches!(
            stream.next().await.unwrap(),
            MediaPacket::Discontinuity
        ));
        assert!(matches!(stream.next().await.unwrap(), MediaPacket::Eos));
        assert!(stream.next().await.is_err());
        assert!(stream.seek(0).await.is_err());
    }
}
