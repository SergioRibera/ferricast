//! [`ProgressivePuller`]: plays a single media file (MP4, MOV, MKV, WebM or audio) over HTTP
//! as a [`MediaPuller`].
//!
//! The demuxer is synchronous, so it runs on a worker thread over an
//! [`http::HttpSource`] and hands packets to the async side through a
//! bounded channel, tagged with a seek generation so packets read before
//! a seek are dropped instead of played.

mod http;

use std::sync::mpsc as std_mpsc;

use ferricast_core::{
    AudioStreamInfo, FerricastError, MediaInfo, MediaPacket, MediaPuller, PullSpec, Result,
    VideoStreamInfo,
};
use tokio::sync::{mpsc, oneshot};

use crate::{Demuxer, Packet};
use http::HttpSource;

/// Packets buffered between the worker and the consumer.
const PACKET_CHANNEL_DEPTH: usize = 64;

/// A packet tagged with the seek generation it belongs to.
type Tagged = (u64, Result<MediaPacket>);

enum Cmd {
    Seek {
        position_us: u64,
        generation: u64,
        done: oneshot::Sender<Result<u64>>,
    },
}

/// Pulls one progressive media file (MP4, MOV, Matroska, WebM) over
/// HTTP(S), with range requests when the server supports them.
#[derive(Default)]
pub struct ProgressivePuller {
    rx: Option<mpsc::Receiver<Tagged>>,
    cmd_tx: Option<std_mpsc::Sender<Cmd>>,
    /// Bumped on every seek; packets of older generations are stale.
    generation: u64,
}

impl MediaPuller for ProgressivePuller {
    async fn open(&mut self, spec: PullSpec) -> Result<MediaInfo> {
        self.close().await?;
        let (packet_tx, packet_rx) = mpsc::channel(PACKET_CHANNEL_DEPTH);
        let (cmd_tx, cmd_rx) = std_mpsc::channel();
        let (info_tx, info_rx) = oneshot::channel();
        // Opening reads the container header over the network; it runs
        // on the worker thread too, so no async runtime is ever blocked.
        std::thread::Builder::new()
            .name("ferricast-progressive".into())
            .spawn(move || {
                let opened = HttpSource::open(&spec.url, &spec.headers).and_then(|source| {
                    let ranged = source.is_ranged();
                    let demuxer = Demuxer::from_reader(source)?;
                    tracing::info!(url = %spec.url, ranged, "progressive: opened");
                    Ok(demuxer)
                });
                let demuxer = match opened {
                    Ok(demuxer) => {
                        if info_tx.send(Ok(media_info(&demuxer))).is_err() {
                            return;
                        }
                        demuxer
                    }
                    Err(e) => {
                        let _ = info_tx.send(Err(e));
                        return;
                    }
                };
                Worker {
                    demuxer,
                    packet_tx,
                    cmd_rx,
                    generation: 0,
                }
                .run();
            })
            .map_err(|e| FerricastError::Pull(format!("spawn demuxer thread: {e}")))?;
        let info = info_rx
            .await
            .map_err(|_| FerricastError::Pull("demuxer thread ended".into()))??;
        tracing::debug!(?info, "progressive: media info");
        self.rx = Some(packet_rx);
        self.cmd_tx = Some(cmd_tx);
        self.generation = 0;
        Ok(info)
    }

    async fn next(&mut self) -> Result<MediaPacket> {
        let Some(rx) = self.rx.as_mut() else {
            return Err(FerricastError::Pull(
                "ProgressivePuller::next before open".into(),
            ));
        };
        loop {
            match rx.recv().await {
                Some((generation, packet)) if generation == self.generation => return packet,
                Some(_) => {}
                None => return Err(FerricastError::Pull("demuxer thread ended".into())),
            }
        }
    }

    async fn seek(&mut self, position_us: u64) -> Result<()> {
        let (Some(cmd_tx), Some(rx)) = (self.cmd_tx.as_ref(), self.rx.as_mut()) else {
            return Err(FerricastError::Pull(
                "ProgressivePuller::seek before open".into(),
            ));
        };
        let generation = self.generation + 1;
        let (done_tx, mut done_rx) = oneshot::channel();
        cmd_tx
            .send(Cmd::Seek {
                position_us,
                generation,
                done: done_tx,
            })
            .map_err(|_| FerricastError::Pull("demuxer thread ended".into()))?;
        self.generation = generation;
        // The worker may be parked on a full channel; drain it (every
        // packet in it is stale now) until the seek is acknowledged.
        let landed = loop {
            // Biased: the worker queues the seek's first packet only after
            // acknowledging, so it is never mistaken for a stale one.
            tokio::select! {
                biased;
                done = &mut done_rx => {
                    break done.map_err(|_| FerricastError::Pull("demuxer thread ended".into()))??;
                }
                stale = rx.recv() => {
                    if stale.is_none() {
                        return Err(FerricastError::Pull("demuxer thread ended".into()));
                    }
                }
            }
        };
        tracing::debug!(position_us, landed, "progressive: seeked");
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // Dropping both ends stops the worker at its next send or command
        // check; it may finish a read in flight first.
        self.rx = None;
        self.cmd_tx = None;
        Ok(())
    }
}

fn media_info(demuxer: &Demuxer) -> MediaInfo {
    MediaInfo {
        video: demuxer.video().map(|v| VideoStreamInfo {
            hdr: v.hdr,
            codec: v.codec,
            width: v.width,
            height: v.height,
            fps: 0,
            colour: v.colour,
        }),
        audio: demuxer.audio().map(|a| AudioStreamInfo {
            codec: a.codec,
            sample_rate: a.sample_rate,
            channels: a.channels,
            source: a.source,
        }),
        duration_us: demuxer.duration_us(),
        is_live: false,
    }
}

struct Worker {
    demuxer: Demuxer,
    packet_tx: mpsc::Sender<Tagged>,
    cmd_rx: std_mpsc::Receiver<Cmd>,
    generation: u64,
}

impl Worker {
    fn run(mut self) {
        // Frames ahead of the presentation start (MP4 edit list pre-roll)
        // are decoded but not shown.
        if self
            .demuxer
            .video()
            .is_some_and(|v| v.presentation_start_us > 0)
        {
            self.present_from(0);
        }
        loop {
            loop {
                match self.cmd_rx.try_recv() {
                    Ok(cmd) => self.handle(cmd),
                    Err(std_mpsc::TryRecvError::Empty) => break,
                    Err(std_mpsc::TryRecvError::Disconnected) => return,
                }
            }
            let packet = match self.demuxer.next_packet() {
                Ok(Some(Packet::Video(frame))) => Ok(MediaPacket::Video(frame)),
                Ok(Some(Packet::Audio(frame))) => Ok(MediaPacket::Audio(frame)),
                Ok(None) => Ok(MediaPacket::Eos),
                Err(e) => Err(e),
            };
            let finished = !matches!(packet, Ok(MediaPacket::Video(_) | MediaPacket::Audio(_)));
            if self
                .packet_tx
                .blocking_send((self.generation, packet))
                .is_err()
            {
                return;
            }
            if finished {
                // Nothing more to read until a seek rewinds the demuxer.
                match self.cmd_rx.recv() {
                    Ok(cmd) => self.handle(cmd),
                    Err(_) => return,
                }
            }
        }
    }

    /// Marks what comes before `timestamp_us` (and any pre-roll) as
    /// decode-only.
    fn present_from(&self, timestamp_us: u64) {
        let start = self.demuxer.video().map_or(0, |v| v.presentation_start_us);
        let timestamp_us = timestamp_us.max(start);
        let marker = MediaPacket::PresentFrom { timestamp_us };
        let _ = self.packet_tx.blocking_send((self.generation, Ok(marker)));
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Seek {
                position_us,
                generation,
                done,
            } => {
                self.generation = generation;
                let landed = self.demuxer.seek(position_us);
                let ok = landed.is_ok();
                let _ = done.send(landed);
                if ok {
                    // Playback resumes at the target, not at the keyframe
                    // the demuxer had to go back to.
                    self.present_from(position_us);
                }
            }
        }
    }
}
