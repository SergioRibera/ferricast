//! AAC decode through libfdk-aac: AAC-LC, HE-AAC (SBR) and HE-AAC v2
//! (parametric stereo) at their full output rate, mono to 7.1.
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
//!
//! The decoder is driven through `fdk-aac-sys` directly: the safe
//! wrapper cannot turn off the output limiter, whose look-ahead delays
//! the audio by its attack time (15 ms) on top of the codec delay.
//! Every channel is output as coded; mixing down for the speakers is
//! up to the sink.

use std::ptr::NonNull;

use fdk_aac_sys as sys;
use ferricast_core::frame::AacChannelConfig;
use ferricast_core::{AudioDecoderConfig, DecodedAudio, FerricastError, Result};

use super::{Shape, adts};

/// Most channels output: 7.1, the widest AAC channel configuration.
const MAX_CHANNELS: usize = 8;
/// Largest frame libfdk-aac outputs: 2048 samples (SBR) by 8 channels.
const MAX_FRAME_SAMPLES: usize = 2048 * MAX_CHANNELS;
/// AAC-LC sampling frequency table (ISO/IEC 14496-3 §1.6.3.4).
const RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// A libfdk-aac decoder instance reading raw access units.
pub(super) struct RawDecoder {
    handle: NonNull<sys::AAC_DECODER_INSTANCE>,
}

// SAFETY: the instance is plain heap state owned by this value; it is
// only used through `&mut self` (or `&self` for the read-only stream
// info), and libfdk-aac keeps no thread-local or global mutable state
// per instance, so moving it to another thread is sound.
unsafe impl Send for RawDecoder {}

impl RawDecoder {
    /// Opens a decoder configured with the `AudioSpecificConfig` `asc`,
    /// limiter off and every coded channel output.
    pub(super) fn open(asc: &[u8]) -> Result<Self> {
        // SAFETY: `aacDecoder_Open` has no preconditions; it returns null
        // on allocation failure, checked below.
        let raw = unsafe { sys::aacDecoder_Open(sys::TRANSPORT_TYPE_TT_MP4_RAW, 1) };
        let handle = NonNull::new(raw).ok_or_else(|| {
            FerricastError::Decode("libfdk-aac: decoder allocation failed".into())
        })?;
        let mut decoder = Self { handle };
        decoder.set_limiter(false)?;
        decoder.set_param(
            sys::AACDEC_PARAM_AAC_PCM_MAX_OUTPUT_CHANNELS,
            MAX_CHANNELS as i32,
        )?;
        let mut conf = asc.as_ptr().cast_mut();
        let len = u32::try_from(asc.len())
            .map_err(|_| FerricastError::Decode("AAC config too long".into()))?;
        // SAFETY: the handle is live; `conf` points at `len` readable
        // bytes, which libfdk-aac only reads (the `*mut` is a C API
        // artefact) and does not keep past the call.
        check(unsafe {
            sys::aacDecoder_ConfigRaw(handle.as_ptr(), &raw mut conf, &raw const len)
        })?;
        Ok(decoder)
    }

    /// Turns the output limiter on or off.
    fn set_limiter(&mut self, enabled: bool) -> Result<()> {
        self.set_param(sys::AACDEC_PARAM_AAC_PCM_LIMITER_ENABLE, i32::from(enabled))
    }

    fn set_param(&mut self, param: sys::AACDEC_PARAM, value: i32) -> Result<()> {
        // SAFETY: the handle is live; `aacDecoder_SetParam` validates
        // the parameter and value itself.
        check(unsafe { sys::aacDecoder_SetParam(self.handle.as_ptr(), param, value) })
    }

    /// Hands `data` to the decoder; returns how many bytes it took.
    pub(super) fn fill(&mut self, data: &[u8]) -> Result<usize> {
        let mut buffer = data.as_ptr().cast_mut();
        let size = u32::try_from(data.len())
            .map_err(|_| FerricastError::Decode("AAC access unit too long".into()))?;
        let mut valid = size;
        // SAFETY: the handle is live; `buffer` points at `size` readable
        // bytes that libfdk-aac copies into its own buffer (it never
        // writes through the pointer), and `valid` is a live `u32` it
        // updates with the bytes left over.
        check(unsafe {
            sys::aacDecoder_Fill(
                self.handle.as_ptr(),
                &raw mut buffer,
                &raw const size,
                &raw mut valid,
            )
        })?;
        Ok(data.len() - valid as usize)
    }

    /// Decodes one frame into `pcm` (interleaved).
    pub(super) fn decode_frame(
        &mut self,
        pcm: &mut [i16],
    ) -> std::result::Result<(), sys::AAC_DECODER_ERROR> {
        let len = i32::try_from(pcm.len()).unwrap_or(i32::MAX);
        // SAFETY: the handle is live and `pcm` is writable for `len`
        // samples, the bound libfdk-aac is given (`INT_PCM` is `i16`).
        let err =
            unsafe { sys::aacDecoder_DecodeFrame(self.handle.as_ptr(), pcm.as_mut_ptr(), len, 0) };
        match err {
            sys::AAC_DECODER_ERROR_AAC_DEC_OK => Ok(()),
            err => Err(err),
        }
    }

    /// Shape of the last decoded frame: `(sample_rate, channels,
    /// samples per channel, output delay in samples)`.
    pub(super) fn stream_info(&self) -> Option<(i32, i32, i32, u32)> {
        // SAFETY: the handle is live; the returned pointer, when not
        // null, points at the instance's own `CStreamInfo`, which stays
        // valid while `&self` borrows the decoder and is only read here.
        let info = unsafe { sys::aacDecoder_GetStreamInfo(self.handle.as_ptr()).as_ref() }?;
        Some((
            info.sampleRate,
            info.numChannels,
            info.frameSize,
            info.outputDelay,
        ))
    }
}

impl Drop for RawDecoder {
    fn drop(&mut self) {
        // SAFETY: the handle came from `aacDecoder_Open` and is closed
        // exactly once, here.
        unsafe { sys::aacDecoder_Close(self.handle.as_ptr()) };
    }
}

fn check(err: sys::AAC_DECODER_ERROR) -> Result<()> {
    match err {
        sys::AAC_DECODER_ERROR_AAC_DEC_OK => Ok(()),
        err => Err(fdk_error(err)),
    }
}

pub(super) fn fdk_error(err: sys::AAC_DECODER_ERROR) -> FerricastError {
    FerricastError::Decode(format!("libfdk-aac error {err:#06x}"))
}

pub(super) struct FdkAac {
    /// Core shape the stream metadata stated, for unframed input.
    configured: Option<Shape>,
    /// The decoder and the core shape it was configured for.
    active: Option<(Shape, RawDecoder)>,
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

    fn decoder_for(
        active: &mut Option<(Shape, RawDecoder)>,
        shape: Shape,
    ) -> Result<&mut RawDecoder> {
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
            let consumed = decoder.fill(unit.payload)?;
            if consumed < unit.payload.len() {
                return Err(FerricastError::Decode(format!(
                    "libfdk-aac took {consumed} of {} bytes of an access unit",
                    unit.payload.len()
                )));
            }
            match decoder.decode_frame(&mut self.pcm) {
                Ok(()) => {}
                // Nothing to output yet (a decoder still settling).
                Err(sys::AAC_DECODER_ERROR_AAC_DEC_NOT_ENOUGH_BITS) => continue,
                Err(e) => return Err(fdk_error(e)),
            }
            let Some((rate, channels, frame, _)) = decoder.stream_info() else {
                continue;
            };
            let (Ok(sample_rate), Ok(channels), Ok(frame)) = (
                u32::try_from(rate),
                u16::try_from(channels),
                usize::try_from(frame),
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

/// The AAC-LC `AudioSpecificConfig` of `shape`: 5 bits object type
/// (2 = LC), 4 bits rate index, 4 bits channel configuration. SBR and
/// PS are found implicitly.
fn lc_config(shape: Shape) -> Result<[u8; 2]> {
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
    let config = AacChannelConfig::from_channels(shape.channels).ok_or_else(unsupported)?;
    let asc = (2u16 << 11) | ((index as u16) << 7) | (u16::from(config.code()) << 3);
    Ok(asc.to_be_bytes())
}

/// A raw-transport decoder for the AAC-LC core of `shape`.
fn raw_decoder(shape: Shape) -> Result<RawDecoder> {
    RawDecoder::open(&lc_config(shape)?)
}

#[cfg(test)]
mod tests {
    use fdk_aac::enc::{AudioObjectType, BitRate, ChannelMode, Encoder, EncoderParams, Transport};

    use super::*;

    const RATE: u32 = 48_000;

    /// One second of stereo silence with a full-scale click at sample
    /// `click`, encoded as raw AAC-LC access units.
    fn encoded_click(click: usize) -> Vec<Vec<u8>> {
        let encoder = Encoder::new(EncoderParams {
            bit_rate: BitRate::Cbr(256_000),
            sample_rate: RATE,
            transport: Transport::Raw,
            channels: ChannelMode::Stereo,
            audio_object_type: AudioObjectType::Mpeg4LowComplexity,
        })
        .unwrap();
        let mut pcm = vec![0i16; RATE as usize * 2];
        for n in click..click + 4 {
            pcm[2 * n] = 30_000;
            pcm[2 * n + 1] = 30_000;
        }
        let frame = encoder.info().unwrap().frameLength as usize * 2;
        let mut units = Vec::new();
        let mut out = vec![0u8; 8192];
        for chunk in pcm.chunks(frame) {
            let info = encoder.encode(chunk, &mut out).unwrap();
            if info.output_size > 0 {
                units.push(out[..info.output_size].to_vec());
            }
        }
        units
    }

    /// Where the click comes out of a decoder with the limiter on or
    /// off, and the output delay the decoder reports.
    fn click_position(units: &[Vec<u8>], limiter: bool) -> (usize, u32) {
        let shape = Shape {
            sample_rate: RATE,
            channels: 2,
        };
        let mut decoder = raw_decoder(shape).unwrap();
        decoder.set_limiter(limiter).unwrap();
        let mut pcm = vec![0i16; MAX_FRAME_SAMPLES];
        let mut left: Vec<i16> = Vec::new();
        let mut delay = 0;
        for unit in units {
            assert_eq!(decoder.fill(unit).unwrap(), unit.len());
            decoder.decode_frame(&mut pcm).unwrap();
            let (_, channels, frame, output_delay) = decoder.stream_info().unwrap();
            assert_eq!(channels, 2);
            delay = output_delay;
            left.extend(pcm[..frame as usize * 2].iter().step_by(2).copied());
        }
        let peak = left
            .iter()
            .enumerate()
            .max_by_key(|&(_, s)| s.unsigned_abs())
            .map(|(i, _)| i)
            .unwrap();
        (peak, delay)
    }

    #[test]
    fn the_limiter_is_off_and_adds_no_delay() {
        let click = 10_000;
        let units = encoded_click(click);
        let (off, off_delay) = click_position(&units, false);
        let (on, on_delay) = click_position(&units, true);
        eprintln!(
            "click at {click}: out at {off} (delay {off_delay}), {on} with limiter ({on_delay})"
        );
        // The limiter delays the output by its look-ahead, 15 ms.
        assert!(on_delay > off_delay, "{on_delay} vs {off_delay}");
        let added = (on_delay - off_delay) as usize;
        assert!(added.abs_diff(720) <= 48, "limiter delay {added}");
        assert!(off < on && (on - off).abs_diff(added) <= 2, "{off} vs {on}");
        // Without it only the codec's own delay is left: the encoder
        // look-ahead (2048 samples) plus the delay the decoder reports.
        let codec = 2048 + off_delay as usize;
        assert!(off >= click && (off - click).abs_diff(codec) <= 2, "{off}");
    }

    #[test]
    fn every_channel_configuration_maps_to_an_lc_config() {
        for (channels, code) in [(1u16, 1u8), (2, 2), (6, 6), (8, 7)] {
            let asc = lc_config(Shape {
                sample_rate: RATE,
                channels,
            })
            .unwrap();
            assert_eq!((asc[1] >> 3) & 0xF, code, "{channels} channels");
            assert!(
                raw_decoder(Shape {
                    sample_rate: RATE,
                    channels
                })
                .is_ok()
            );
        }
        assert!(
            lc_config(Shape {
                sample_rate: RATE,
                channels: 7
            })
            .is_err()
        );
    }
}
