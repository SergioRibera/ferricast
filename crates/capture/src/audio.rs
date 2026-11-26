//! [`NativeAudioCapture`]: system audio over the plugin's audio-only
//! sessions.

use bytes::Bytes;
use ferricast_core::{
    AudioCapture, AudioCaptureConfig, AudioCodec, AudioFrame, AudioMuteHandle, AudioSource,
    FerricastError, Result,
};
use istmo_core::{NativeBuffer, Overflow, SinkPolicy};
use istmo_screen_capture::{
    AudioConfig, AudioMeta, AudioSessionHandle, AudioSource as PluginSource, SampleFormat,
    ScreenCaptureClientExt as _,
};
use tracing::{debug, info, warn};

use crate::plugin::{Plugin, capture_error};

/// Chunks queued between PipeWire and [`AudioCapture::next_frame`]. A
/// full sink refuses new chunks rather than evicting queued ones, so
/// what is delivered stays contiguous (gaps show as `discontinuity`).
const SINK_DEPTH: usize = 32;

/// System audio (or one PipeWire node) as interleaved S16LE PCM, through
/// the istmo screen-capture plugin's audio-only sessions.
///
/// Timestamps are the plugin's `wall_us`: UNIX-epoch microseconds that
/// follow the sample count and re-anchor on gaps, the clock the video
/// frames and the HLS segmenter share.
#[derive(Debug, Default)]
pub struct NativeAudioCapture {
    running: Option<Running>,
}

#[derive(Debug)]
struct Running {
    session: AudioSessionHandle,
    mute: AudioMuteHandle,
    rate: u32,
    channels: u16,
}

impl NativeAudioCapture {
    /// A capture that opens on [`AudioCapture::start`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// The plugin's audio request for `source` and `config`. S16 is what
/// every ferricast audio encoder takes; PipeWire converts rate,
/// channels and format.
fn audio_config(source: &AudioSource, config: AudioCaptureConfig) -> AudioConfig {
    let source = match source {
        AudioSource::DefaultMonitor => PluginSource::System,
        AudioSource::Node(id) => PluginSource::Node(*id),
    };
    let channels = u8::try_from(config.channels).unwrap_or(u8::MAX);
    AudioConfig::of(source)
        .rate(config.sample_rate)
        .channels(channels)
        .format(SampleFormat::S16)
}

/// A chunk's samples as ferricast PCM: the plugin's buffer itself, or
/// silence of the same length while muted (the stream keeps its
/// cadence, so receivers never stall on the audio PID).
fn pcm(meta: &AudioMeta, buffer: NativeBuffer, muted: bool) -> Result<AudioFrame> {
    if meta.format != SampleFormat::S16 {
        return Err(FerricastError::Capture(format!(
            "audio capture: {:?} samples, S16 asked for",
            meta.format
        )));
    }
    let len = buffer.as_bytes().map_or(0, <[u8]>::len);
    let data = if muted {
        Bytes::from(vec![0u8; len])
    } else {
        Bytes::from_owner(Pcm(buffer))
    };
    Ok(AudioFrame {
        codec: AudioCodec::Pcm,
        data,
        timestamp_us: meta.wall_us,
        sample_rate: meta.rate,
        channels: u16::from(meta.channels),
    })
}

/// The bytes of an audio chunk, held as the plugin's buffer.
struct Pcm(NativeBuffer);

impl AsRef<[u8]> for Pcm {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes().unwrap_or_default()
    }
}

impl AudioCapture for NativeAudioCapture {
    async fn start(
        &mut self,
        source: AudioSource,
        config: AudioCaptureConfig,
        mute: AudioMuteHandle,
    ) -> Result<()> {
        self.stop().await?;
        let plugin = Plugin::get()?;
        let session = plugin
            .client()
            .open_audio_session(
                plugin.runtime(),
                audio_config(&source, config),
                SinkPolicy::queue(SINK_DEPTH).with_overflow(Overflow::DropNewest),
            )
            .await
            .map_err(capture_error("opening the audio session"))?;
        let negotiated = session.negotiated();
        info!(
            ?source,
            rate = negotiated.rate,
            channels = negotiated.channels,
            format = ?negotiated.format,
            "audio capture started"
        );
        // The rate and channels are asked for; a different answer would
        // reach the encoder as a different pitch or speed on the
        // receiver unless it follows the chunks' own shape.
        if negotiated.rate != config.sample_rate
            || u16::from(negotiated.channels) != config.channels
        {
            warn!(
                requested_rate = config.sample_rate,
                negotiated_rate = negotiated.rate,
                requested_channels = config.channels,
                negotiated_channels = negotiated.channels,
                "audio capture: PipeWire negotiated another shape than requested"
            );
        }
        self.running = Some(Running {
            rate: negotiated.rate,
            channels: u16::from(negotiated.channels),
            session,
            mute,
        });
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<AudioFrame> {
        let running = self
            .running
            .as_ref()
            .ok_or_else(|| FerricastError::Capture("audio capture: not started".into()))?;
        let Some(chunk) = running.session.audio().recv_async().await else {
            let reason = running.session.audio().end_reason();
            return Err(FerricastError::Capture(format!(
                "audio capture ended: {reason:?}"
            )));
        };
        if chunk.meta.discontinuity {
            debug!(wall_us = chunk.meta.wall_us, "audio capture re-anchored");
        }
        let (meta, buffer) = chunk.into_parts();
        pcm(&meta, buffer, running.mute.is_muted())
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(running) = self.running.take() {
            running
                .session
                .close()
                .await
                .map_err(capture_error("closing the audio session"))?;
        }
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.is_some()
    }

    fn sample_rate(&self) -> u32 {
        self.running.as_ref().map_or(0, |running| running.rate)
    }

    fn channels(&self) -> u16 {
        self.running.as_ref().map_or(0, |running| running.channels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(format: SampleFormat) -> AudioMeta {
        AudioMeta {
            seq: 0,
            pts_ns: 0,
            wall_us: 1_700_000_000_000_000,
            rate: 48_000,
            channels: 2,
            format,
            frames: 2,
            discontinuity: false,
        }
    }

    #[test]
    fn chunks_are_s16_pcm_on_the_wall_clock() {
        let samples = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let frame = pcm(
            &meta(SampleFormat::S16),
            NativeBuffer::from_bytes(samples.clone()),
            false,
        )
        .unwrap();
        assert_eq!(frame.codec, AudioCodec::Pcm);
        assert_eq!(frame.data.as_ref(), samples.as_slice());
        assert_eq!(frame.timestamp_us, 1_700_000_000_000_000);
        assert_eq!((frame.sample_rate, frame.channels), (48_000, 2));
    }

    #[test]
    fn muted_chunks_keep_their_length_and_time() {
        let frame = pcm(
            &meta(SampleFormat::S16),
            NativeBuffer::from_bytes(vec![9u8; 8]),
            true,
        )
        .unwrap();
        assert_eq!(frame.data.as_ref(), [0u8; 8]);
        assert_eq!(frame.timestamp_us, 1_700_000_000_000_000);
    }

    #[test]
    fn float_chunks_are_refused() {
        assert!(
            pcm(
                &meta(SampleFormat::F32),
                NativeBuffer::from_bytes(vec![0u8; 8]),
                false
            )
            .is_err()
        );
    }

    #[test]
    fn the_request_asks_for_s16_at_the_configured_shape() {
        let config = audio_config(
            &AudioSource::Node(77),
            AudioCaptureConfig {
                sample_rate: 44_100,
                channels: 2,
            },
        );
        assert_eq!(config.source, PluginSource::Node(77));
        assert_eq!(
            (config.rate, config.channels, config.format),
            (Some(44_100), Some(2), SampleFormat::S16)
        );
    }
}
