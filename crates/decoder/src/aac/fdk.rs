//! AAC decode through libfdk-aac: AAC-LC, HE-AAC (SBR) and HE-AAC v2
//! (parametric stereo) at their full output rate and channel count.
//!
//! An ADTS header can only name the AAC-LC core, so HE-AAC in ADTS (or
//! in an MP4 whose `AudioSpecificConfig` the demuxer rewrote as ADTS) is
//! signalled implicitly: the SBR and PS data ride in the core's fill
//! elements. libfdk-aac detects both and doubles the rate / upmixes to
//! stereo on its own, which symphonia's LC-only decoder cannot.
//!
//! ADTS headers are stripped here and the access units fed to a
//! raw-transport decoder configured from them: libfdk-aac's own ADTS
//! transport holds every frame back until it has seen the next sync
//! word, which would delay output by one frame and lose the last one.

use fdk_aac::dec::{Decoder, DecoderError, Transport};
use ferricast_core::{AudioDecoderConfig, DecodedAudio, FerricastError, Result};

use super::{Shape, adts};

/// Largest frame libfdk-aac outputs: 2048 samples (SBR) by 8 channels.
const MAX_FRAME_SAMPLES: usize = 2048 * 8;
/// AAC-LC sampling frequency table (ISO/IEC 14496-3 §1.6.3.4).
const RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

pub(super) struct FdkAac {
    /// Core shape the stream metadata stated, for unframed input.
    configured: Option<Shape>,
    /// The decoder and the core shape it was configured for.
    active: Option<(Shape, Decoder)>,
    pcm: Vec<i16>,
}

impl FdkAac {
    pub(super) fn new(config: &AudioDecoderConfig) -> Self {
        let configured = (config.sample_rate > 0 && config.channels > 0).then_some(Shape {
            sample_rate: config.sample_rate,
            channels: config.channels,
        });
        Self {
            configured,
            active: None,
            pcm: vec![0; MAX_FRAME_SAMPLES],
        }
    }

    fn decoder_for(active: &mut Option<(Shape, Decoder)>, shape: Shape) -> Result<&mut Decoder> {
        if active.as_ref().is_none_or(|(s, _)| *s != shape) {
            *active = Some((shape, raw_decoder(shape)?));
        }
        let (_, decoder) = active
            .as_mut()
            .expect("invariant: set just above when missing");
        Ok(decoder)
    }

    /// Decodes every access unit of `data` into `runs` (one run per
    /// stretch of identically shaped output), the first stamped
    /// `timestamp_us` and each next one after the samples before it.
    pub(super) fn decode(
        &mut self,
        data: &[u8],
        timestamp_us: u64,
        runs: &mut Vec<DecodedAudio>,
    ) -> Result<()> {
        let first_run = runs.len();
        for unit in adts::units(data) {
            let unit = unit.map_err(|e| FerricastError::Decode(format!("AAC: {e}")))?;
            let shape = unit.shape.or(self.configured).ok_or_else(|| {
                FerricastError::Decode(
                    "AAC: raw (non-ADTS) input with no sample rate / channels configured".into(),
                )
            })?;
            let decoder = Self::decoder_for(&mut self.active, shape)?;
            let consumed = decoder.fill(unit.payload).map_err(fdk_error)?;
            if consumed < unit.payload.len() {
                return Err(FerricastError::Decode(format!(
                    "libfdk-aac took {consumed} of {} bytes of an access unit",
                    unit.payload.len()
                )));
            }
            match decoder.decode_frame(&mut self.pcm) {
                Ok(()) => {}
                // Nothing to output yet (a decoder still settling).
                Err(e) if e == DecoderError::NOT_ENOUGH_BITS => continue,
                Err(e) => return Err(fdk_error(e)),
            }
            let info = decoder.stream_info();
            let (Ok(sample_rate), Ok(channels), Ok(frame)) = (
                u32::try_from(info.sampleRate),
                u16::try_from(info.numChannels),
                usize::try_from(info.frameSize),
            ) else {
                return Err(FerricastError::Decode(
                    "libfdk-aac reported a negative stream shape".into(),
                ));
            };
            if sample_rate == 0 || channels == 0 {
                continue;
            }
            let samples = (frame * usize::from(channels)).min(self.pcm.len());
            let pcm = &self.pcm[..samples];
            let own_run = runs.len() > first_run;
            match runs.last_mut() {
                Some(run)
                    if own_run && run.sample_rate == sample_rate && run.channels == channels =>
                {
                    run.pcm.extend_from_slice(pcm);
                }
                _ => {
                    let timestamp_us = runs
                        .get(first_run..)
                        .and_then(<[DecodedAudio]>::last)
                        .map_or(timestamp_us, |run| run.timestamp_us + run_duration_us(run));
                    runs.push(DecodedAudio {
                        pcm: pcm.to_vec(),
                        sample_rate,
                        channels,
                        timestamp_us,
                    });
                }
            }
        }
        Ok(())
    }
}

fn run_duration_us(run: &DecodedAudio) -> u64 {
    let frames = (run.pcm.len() / usize::from(run.channels.max(1))) as u64;
    frames * 1_000_000 / u64::from(run.sample_rate.max(1))
}

/// A raw-transport decoder configured with the AAC-LC
/// `AudioSpecificConfig` of `shape`; SBR and PS are found implicitly.
fn raw_decoder(shape: Shape) -> Result<Decoder> {
    let unsupported = || {
        FerricastError::Decode(format!(
            "AAC at {} Hz with {} channels is not supported",
            shape.sample_rate, shape.channels
        ))
    };
    let index = RATES
        .iter()
        .position(|&r| r == shape.sample_rate)
        .ok_or_else(unsupported)?;
    let channels = u8::try_from(shape.channels)
        .ok()
        .filter(|c| (1..=7).contains(c))
        .ok_or_else(unsupported)?;
    // 5 bits object type (2 = LC), 4 bits rate index, 4 bits channels.
    let asc = (2u16 << 11) | ((index as u16) << 7) | (u16::from(channels) << 3);
    let mut decoder = Decoder::new(Transport::Raw);
    decoder.config_raw(&asc.to_be_bytes()).map_err(fdk_error)?;
    Ok(decoder)
}

fn fdk_error(e: DecoderError) -> FerricastError {
    FerricastError::Decode(format!("libfdk-aac: {}", e.message()))
}
