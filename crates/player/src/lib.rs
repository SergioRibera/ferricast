mod clock;
mod decoder;
mod limiter;
mod pts;

use std::collections::VecDeque;
use std::path::Path;

use ferricast_core::{FerricastError, Result};
use ferricast_demuxer::Demuxer;

pub use clock::{Clock, Due};
pub use ferricast_core::{CapturedFrame, PixelFormat, RawFrame};
pub use ferricast_demuxer::VideoStream;
pub use limiter::FrameLimiter;

use decoder::Decoder;
use pts::Presentation;

pub struct Player {
    demuxer: Demuxer,
    video: VideoStream,
    decoder: Decoder,
    presentation: Presentation,
    ready: VecDeque<CapturedFrame>,
    drained: bool,
    software_only: bool,
    /// Packets pulled from the demuxer since the last restart.
    packets: usize,
    /// Index (in `packets`) of the last keyframe pulled.
    last_keyframe: usize,
    /// Packets before this index are skipped without decoding while catching
    /// up after a decoder restart.
    skip_until: usize,
}

impl Player {
    pub fn open(path: &Path) -> Result<Self> {
        let demuxer = Demuxer::open(path)?;
        let video = demuxer
            .video()
            .cloned()
            .ok_or_else(|| FerricastError::Demux("no supported video track".into()))?;
        let decoder = open_decoder(&video, false)?;
        let presentation = Presentation::new(video.presentation_start_us);
        Ok(Self {
            demuxer,
            video,
            decoder,
            presentation,
            ready: VecDeque::new(),
            drained: false,
            software_only: false,
            packets: 0,
            last_keyframe: 0,
            skip_until: 0,
        })
    }

    pub fn video(&self) -> &VideoStream {
        &self.video
    }

    pub fn next_frame(&mut self) -> Result<Option<CapturedFrame>> {
        loop {
            if let Some(frame) = self.ready.pop_front() {
                return Ok(Some(frame));
            }
            if self.drained {
                return Ok(None);
            }
            let Some(packet) = self.demuxer.next_frame()? else {
                self.drained = true;
                match self.decoder.flush() {
                    Ok(frames) => {
                        for frame in frames {
                            self.deliver(frame);
                        }
                    }
                    Err(error) => self.recover(error)?,
                }
                continue;
            };
            let index = self.packets;
            self.packets += 1;
            if packet.is_keyframe {
                self.last_keyframe = index;
            }
            if index < self.skip_until {
                continue;
            }
            match self.decoder.decode(packet) {
                Ok(Some(frame)) => self.deliver(frame),
                Ok(None) => {}
                Err(error) => self.recover(error)?,
            }
        }
    }

    pub fn rewind(&mut self) -> Result<()> {
        self.restart()?;
        self.presentation.restart();
        self.skip_until = 0;
        Ok(())
    }

    /// Reopens the demuxer at the start with a fresh decoder, keeping the
    /// presentation state so the caller decides what may be shown again.
    fn restart(&mut self) -> Result<()> {
        self.demuxer.rewind()?;
        self.decoder = open_decoder(&self.video, self.software_only)?;
        self.ready.clear();
        self.drained = false;
        self.packets = 0;
        self.last_keyframe = 0;
        Ok(())
    }

    /// Falls back from the hardware decoder to software without replaying
    /// what was already shown: decoding resumes at the last keyframe at or
    /// before the failing packet, and decoded frames up to the last one
    /// delivered are dropped until playback has caught up.
    fn recover(&mut self, error: FerricastError) -> Result<()> {
        if !self.decoder.is_hardware() {
            return Err(error);
        }
        tracing::warn!(%error, "hardware decoding failed; continuing in software");
        self.software_only = true;
        let resume_at = self.last_keyframe;
        self.restart()?;
        self.skip_until = resume_at;
        self.presentation.resume();
        Ok(())
    }

    /// Every decoder stamps a picture with the timestamp of the packet it was
    /// decoded from, through reordering and dropped packets alike.
    fn deliver(&mut self, mut frame: CapturedFrame) {
        let Some(shown_at) = self.presentation.admit(frame.timestamp_us()) else {
            return;
        };
        match &mut frame {
            CapturedFrame::Cpu(raw) => raw.timestamp_us = shown_at,
            CapturedFrame::Gpu(gpu) => gpu.timestamp_us = shown_at,
        }
        self.ready.push_back(frame);
    }
}

fn open_decoder(stream: &VideoStream, software_only: bool) -> Result<Decoder> {
    Decoder::open(stream.codec, stream.width, stream.height, software_only)
}
