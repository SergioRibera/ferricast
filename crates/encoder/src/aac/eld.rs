//! AAC-ELD encoder (libfdk-aac) for AirPlay screen-mirroring audio.
//!
//! AirPlay receivers decode mirroring audio as AAC-ELD, 44.1 kHz stereo,
//! 480 samples per frame, no SBR, raw access units (UxPlay hands them to
//! its decoder with the AudioSpecificConfig `f8 e8 50 00`). The `fdk-aac`
//! crate's safe wrapper cannot set the 480-sample granule, so this talks
//! to `fdk-aac-sys` directly.
//!
//! Input at any other rate is resampled to 44.1 kHz first. Output
//! timestamps come from [`AudioClock`], anchored on the capture clock.

use std::mem::{self, MaybeUninit};
use std::os::raw::{c_int, c_void};
use std::ptr;

use bytes::Bytes;
use fdk_aac_sys as sys;
use ferricast_core::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioFrame, FerricastError, Result,
};
use tracing::{debug, info};

use super::resample::Resampler;
use crate::audio_clock::{AudioClock, ClockEvent};

/// Sample rate of every AirPlay audio format.
pub const ELD_SAMPLE_RATE: u32 = 44_100;
/// Samples per channel in one AAC-ELD frame.
pub const ELD_FRAME_SAMPLES: usize = 480;
/// AudioSpecificConfig of AAC-ELD 44.1 kHz stereo, 480-sample frames,
/// no SBR: what AirPlay receivers configure their decoder with.
pub const ELD_AUDIO_SPECIFIC_CONFIG: [u8; 4] = [0xf8, 0xe8, 0x50, 0x00];

/// Scratch space for one encoded frame; a 480-sample frame at 256 kbps
/// is about 350 bytes.
const OUTPUT_SCRATCH: usize = 4096;

/// `AACENC_TRANSMUX` value for raw access units (`TT_MP4_RAW`).
const TRANSMUX_RAW: u32 = 0;
/// `AACENC_CHANNELMODE` values (`MODE_1`, `MODE_2`).
const MODE_MONO: u32 = 1;
const MODE_STEREO: u32 = 2;

// UxPlay drops every AAC-ELD frame whose first byte is not 0x80-0x82 or
// 0x8c-0x8e (what Apple's encoder emits). That byte follows the coded
// bandwidth: fdk-aac's automatic bandwidth starts frames at 0x14-0x82
// depending on bitrate, while a fixed 20 kHz bandwidth gives 0x8c-0x8e
// like Apple's from 96 kbps up, and digital silence always codes as 0x14.
/// Coded audio bandwidth.
const BANDWIDTH_HZ: u32 = 20_000;
/// Lowest bitrate at which every frame keeps the full bandwidth.
const MIN_BITRATE_KBPS: u32 = 96;
/// Peak of the noise that replaces digitally silent frames: inaudible
/// (about -90 dBFS) but enough to keep the full-bandwidth frame layout.
const SILENCE_DITHER: i16 = 1;

fn check(what: &str, code: sys::AACENC_ERROR) -> Result<()> {
    if code == sys::AACENC_ERROR_AACENC_OK {
        Ok(())
    } else {
        Err(FerricastError::Encoder(format!(
            "fdk-aac ELD: {what} failed (error {code:#x})"
        )))
    }
}

/// Owned libfdk-aac encoder instance.
struct Handle(sys::HANDLE_AACENCODER);

// SAFETY: the encoder instance has no thread affinity; `Handle` owns it
// exclusively and every call goes through `&mut self`.
unsafe impl Send for Handle {}

impl Handle {
    fn open(channels: u16) -> Result<Self> {
        let mut raw: sys::HANDLE_AACENCODER = ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer; 0 modules = all available.
        check("aacEncOpen", unsafe {
            sys::aacEncOpen(&raw mut raw, 0, u32::from(channels))
        })?;
        Ok(Self(raw))
    }

    fn set(&mut self, param: sys::AACENC_PARAM, value: u32, what: &str) -> Result<()> {
        // SAFETY: `self.0` is a live handle from `aacEncOpen`.
        check(what, unsafe {
            sys::aacEncoder_SetParam(self.0, param, value)
        })
    }

    /// Apply the parameters set so far (`aacEncEncode` with null buffers).
    fn initialize(&mut self) -> Result<()> {
        // SAFETY: null buffers are the documented way to initialise.
        check("initialisation", unsafe {
            sys::aacEncEncode(
                self.0,
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            )
        })
    }

    fn info(&self) -> Result<sys::AACENC_InfoStruct> {
        let mut info = MaybeUninit::<sys::AACENC_InfoStruct>::uninit();
        // SAFETY: `info` is a valid out-pointer that `aacEncInfo` fills
        // completely on success.
        check("aacEncInfo", unsafe {
            sys::aacEncInfo(self.0, info.as_mut_ptr())
        })?;
        // SAFETY: initialised by the successful call above.
        Ok(unsafe { info.assume_init() })
    }

    /// Encode `input` (interleaved S16) into `output`; returns
    /// `(samples consumed, bytes written)`.
    fn encode(&mut self, input: &[i16], output: &mut [u8]) -> Result<(usize, usize)> {
        let to_int = |n: usize| {
            c_int::try_from(n)
                .map_err(|_| FerricastError::Encoder("fdk-aac ELD: buffer too large".into()))
        };
        let mut in_ptr = input.as_ptr().cast_mut().cast::<c_void>();
        let mut in_id = sys::AACENC_BufferIdentifier_IN_AUDIO_DATA as c_int;
        let mut in_size = to_int(mem::size_of_val(input))?;
        let mut in_el = to_int(mem::size_of::<i16>())?;
        let in_desc = sys::AACENC_BufDesc {
            numBufs: 1,
            bufs: &raw mut in_ptr,
            bufferIdentifiers: &raw mut in_id,
            bufSizes: &raw mut in_size,
            bufElSizes: &raw mut in_el,
        };
        let mut out_ptr = output.as_mut_ptr().cast::<c_void>();
        let mut out_id = sys::AACENC_BufferIdentifier_OUT_BITSTREAM_DATA as c_int;
        let mut out_size = to_int(output.len())?;
        let mut out_el = 1;
        let out_desc = sys::AACENC_BufDesc {
            numBufs: 1,
            bufs: &raw mut out_ptr,
            bufferIdentifiers: &raw mut out_id,
            bufSizes: &raw mut out_size,
            bufElSizes: &raw mut out_el,
        };
        let in_args = sys::AACENC_InArgs {
            numInSamples: to_int(input.len())?,
            numAncBytes: 0,
        };
        // SAFETY: plain C struct of integers; all-zero is a valid value.
        let mut out_args: sys::AACENC_OutArgs = unsafe { mem::zeroed() };
        // SAFETY: the descriptors point at locals and at `input` / `output`,
        // all alive for the call; sizes match the buffers. The encoder only
        // reads the input buffer despite the `*mut` in its descriptor.
        check("aacEncEncode", unsafe {
            sys::aacEncEncode(
                self.0,
                &raw const in_desc,
                &raw const out_desc,
                &raw const in_args,
                &raw mut out_args,
            )
        })?;
        let consumed = usize::try_from(out_args.numInSamples).unwrap_or(0);
        let written = usize::try_from(out_args.numOutBytes).unwrap_or(0);
        Ok((consumed, written.min(output.len())))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from `aacEncOpen` and is closed only here.
        unsafe {
            sys::aacEncClose(&raw mut self.0);
        }
    }
}

/// State that exists once [`AudioEncoder::configure`] ran.
struct Configured {
    handle: Handle,
    /// Rate and channel count of the PCM handed to `encode`.
    input_rate: u32,
    channels: u16,
    /// Converts input to 44.1 kHz when it arrives at another rate.
    resampler: Option<Resampler>,
    /// Interleaved 44.1 kHz PCM waiting for a full frame.
    pcm: Vec<i16>,
    /// Scratch for the resampler output of one chunk.
    converted: Vec<i16>,
    clock: AudioClock,
    scratch: Vec<u8>,
    /// State of the generator behind [`SILENCE_DITHER`].
    dither: u32,
}

/// AAC-ELD encoder producing raw 480-sample access units at 44.1 kHz.
#[derive(Default)]
pub struct FdkEldEncoder {
    state: Option<Configured>,
    pending: Vec<AudioFrame>,
}

impl FdkEldEncoder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn open(config: &AudioEncoderConfig) -> Result<Configured> {
        let mode = match config.channels {
            1 => MODE_MONO,
            2 => MODE_STEREO,
            other => {
                return Err(FerricastError::Encoder(format!(
                    "fdk-aac ELD: unsupported channel count {other} (1 or 2 supported)"
                )));
            }
        };
        let mut handle = Handle::open(config.channels)?;
        handle.set(
            sys::AACENC_PARAM_AACENC_AOT,
            sys::AUDIO_OBJECT_TYPE_AOT_ER_AAC_ELD as u32,
            "AOT",
        )?;
        handle.set(sys::AACENC_PARAM_AACENC_SBR_MODE, 0, "SBR mode")?;
        handle.set(
            sys::AACENC_PARAM_AACENC_SAMPLERATE,
            ELD_SAMPLE_RATE,
            "sample rate",
        )?;
        handle.set(sys::AACENC_PARAM_AACENC_CHANNELMODE, mode, "channel mode")?;
        handle.set(
            sys::AACENC_PARAM_AACENC_GRANULE_LENGTH,
            ELD_FRAME_SAMPLES as u32,
            "granule length",
        )?;
        handle.set(sys::AACENC_PARAM_AACENC_BITRATEMODE, 0, "bitrate mode")?;
        let bitrate_kbps = config.bitrate_kbps.max(MIN_BITRATE_KBPS);
        handle.set(
            sys::AACENC_PARAM_AACENC_BITRATE,
            bitrate_kbps.saturating_mul(1000),
            "bitrate",
        )?;
        handle.set(sys::AACENC_PARAM_AACENC_TRANSMUX, TRANSMUX_RAW, "transport")?;
        handle.set(sys::AACENC_PARAM_AACENC_AFTERBURNER, 1, "afterburner")?;
        handle.set(
            sys::AACENC_PARAM_AACENC_BANDWIDTH,
            BANDWIDTH_HZ,
            "bandwidth",
        )?;
        handle.initialize()?;

        let info = handle.info()?;
        let asc_len = usize::try_from(info.confSize)
            .unwrap_or(0)
            .min(info.confBuf.len());
        let asc = &info.confBuf[..asc_len];
        if info.frameLength as usize != ELD_FRAME_SAMPLES {
            return Err(FerricastError::Encoder(format!(
                "fdk-aac ELD: frame length {} instead of {ELD_FRAME_SAMPLES}",
                info.frameLength
            )));
        }
        info!(
            input_rate = config.sample_rate,
            channels = config.channels,
            bitrate_kbps,
            asc = %hex_bytes(asc),
            "AAC-ELD encoder backend: fdk-aac (libfdk-aac)"
        );
        let resampler = (config.sample_rate != ELD_SAMPLE_RATE).then(|| {
            info!(
                from = config.sample_rate,
                to = ELD_SAMPLE_RATE,
                "AAC-ELD: resampling capture audio"
            );
            Resampler::new(config.sample_rate, ELD_SAMPLE_RATE, config.channels)
        });
        Ok(Configured {
            handle,
            input_rate: config.sample_rate,
            channels: config.channels,
            resampler,
            pcm: Vec::with_capacity(ELD_FRAME_SAMPLES * usize::from(config.channels) * 4),
            converted: Vec::new(),
            clock: AudioClock::new(ELD_SAMPLE_RATE),
            scratch: vec![0; OUTPUT_SCRATCH],
            dither: 0x2545_f491,
        })
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl Configured {
    /// Replace the first `len` (all-zero) samples with [`SILENCE_DITHER`]-level
    /// noise.
    fn dither_silence(&mut self, len: usize) {
        for s in &mut self.pcm[..len] {
            // xorshift32: cheap, and its low bits are noise enough here.
            self.dither ^= self.dither << 13;
            self.dither ^= self.dither >> 17;
            self.dither ^= self.dither << 5;
            *s = match self.dither % 3 {
                0 => -SILENCE_DITHER,
                1 => 0,
                _ => SILENCE_DITHER,
            };
        }
    }

    fn push(&mut self, frame: &AudioFrame, pending: &mut Vec<AudioFrame>) -> Result<()> {
        let channels = usize::from(self.channels.max(1));
        let samples: Vec<i16> = frame
            .data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let samples = samples
            .get(..samples.len() / channels * channels)
            .unwrap_or_default();
        self.converted.clear();
        let chunk: &[i16] = match self.resampler.as_mut() {
            Some(r) => {
                r.process(samples, &mut self.converted);
                &self.converted
            }
            None => samples,
        };

        let buffered = (self.pcm.len() / channels) as u64;
        let event = self.clock.observe(
            frame.timestamp_us,
            buffered,
            (chunk.len() / channels) as u64,
        );
        match event {
            ClockEvent::Resynced { gap_us } => {
                debug!(
                    gap_us,
                    "AAC-ELD: capture gap, re-anchoring output timestamps"
                );
            }
            ClockEvent::Trimmed {
                ahead_us,
                report: true,
                ..
            } => info!(
                ahead_us,
                "AAC-ELD: output clock ahead of capture, discarding input samples"
            ),
            ClockEvent::Anchored | ClockEvent::Continuous | ClockEvent::Trimmed { .. } => {}
        }
        let skip = usize::try_from(event.skip())
            .map_or(chunk.len(), |s| s.saturating_mul(channels).min(chunk.len()));
        self.pcm.extend_from_slice(&chunk[skip..]);

        let per_frame = ELD_FRAME_SAMPLES * channels;
        while self.pcm.len() >= per_frame {
            if self.pcm[..per_frame].iter().all(|&s| s == 0) {
                self.dither_silence(per_frame);
            }
            let (consumed, written) = self
                .handle
                .encode(&self.pcm[..per_frame], &mut self.scratch)?;
            if consumed == 0 && written == 0 {
                break;
            }
            self.pcm.drain(..consumed.min(per_frame));
            if written > 0 {
                pending.push(AudioFrame {
                    codec: AudioCodec::AacEld,
                    data: Bytes::copy_from_slice(&self.scratch[..written]),
                    timestamp_us: self.clock.next_timestamp_us(),
                    sample_rate: ELD_SAMPLE_RATE,
                    channels: self.channels,
                });
                self.clock.advance(ELD_FRAME_SAMPLES as u64);
            }
        }
        Ok(())
    }
}

impl AudioEncoder for FdkEldEncoder {
    fn configure(&mut self, config: &AudioEncoderConfig) -> Result<()> {
        if config.codec != AudioCodec::AacEld {
            return Err(FerricastError::Encoder(format!(
                "FdkEldEncoder: only AAC-ELD is supported, got {:?}",
                config.codec
            )));
        }
        self.state = Some(Self::open(config)?);
        self.pending.clear();
        Ok(())
    }

    fn encode(&mut self, frame: &AudioFrame) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Err(FerricastError::Encoder(
                "FdkEldEncoder::encode called before configure()".into(),
            ));
        };
        if frame.codec != AudioCodec::Pcm {
            return Err(FerricastError::Encoder(format!(
                "fdk-aac ELD: expected PCM input, got {:?}",
                frame.codec
            )));
        }
        if frame.sample_rate != state.input_rate || frame.channels != state.channels {
            return Err(FerricastError::Encoder(format!(
                "fdk-aac ELD: input shape changed mid-stream: configured {}Hz/{}ch, got {}Hz/{}ch",
                state.input_rate, state.channels, frame.sample_rate, frame.channels
            )));
        }
        state.push(frame, &mut self.pending)
    }

    fn take_output(&mut self) -> Vec<AudioFrame> {
        mem::take(&mut self.pending)
    }

    fn codec_config(&self) -> Vec<u8> {
        ELD_AUDIO_SPECIFIC_CONFIG.to_vec()
    }

    fn flush(self) -> Result<Vec<AudioFrame>> {
        Ok(self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(sample_rate: u32) -> AudioEncoderConfig {
        AudioEncoderConfig {
            codec: AudioCodec::AacEld,
            sample_rate,
            channels: 2,
            bitrate_kbps: 128,
        }
    }

    fn tone(rate: u32, frames: usize, start: usize) -> Bytes {
        let mut out = Vec::with_capacity(frames * 4);
        for i in start..start + frames {
            let v = ((2.0 * std::f64::consts::PI * 440.0 * i as f64 / f64::from(rate)).sin()
                * 8000.0) as i16;
            out.extend_from_slice(&v.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        Bytes::from(out)
    }

    fn encode_seconds(rate: u32, chunk: usize) -> Vec<AudioFrame> {
        let mut enc = FdkEldEncoder::new();
        enc.configure(&config(rate)).expect("configure");
        let mut out = Vec::new();
        let total = rate as usize;
        let mut done = 0;
        while done < total {
            let ts = 1_000_000_000 + (done as u64 * 1_000_000) / u64::from(rate);
            enc.encode(&AudioFrame {
                codec: AudioCodec::Pcm,
                data: tone(rate, chunk, done),
                timestamp_us: ts,
                sample_rate: rate,
                channels: 2,
            })
            .expect("encode");
            out.extend(enc.take_output());
            done += chunk;
        }
        out
    }

    #[test]
    fn emits_the_airplay_audio_specific_config() {
        let mut enc = FdkEldEncoder::new();
        enc.configure(&config(ELD_SAMPLE_RATE)).expect("configure");
        let info = enc.state.as_ref().unwrap().handle.info().unwrap();
        assert_eq!(
            &info.confBuf[..info.confSize as usize],
            &ELD_AUDIO_SPECIFIC_CONFIG
        );
    }

    #[test]
    fn frames_are_480_samples_apart() {
        let frames = encode_seconds(ELD_SAMPLE_RATE, 1024);
        // About one second (rounded up to whole chunks) of 480-sample frames.
        assert!((85..=94).contains(&frames.len()), "{} frames", frames.len());
        for pair in frames.windows(2) {
            let step = pair[1].timestamp_us - pair[0].timestamp_us;
            assert!((10_883..=10_885).contains(&step), "step {step}");
        }
        assert!(
            frames
                .iter()
                .all(|f| f.codec == AudioCodec::AacEld && !f.data.is_empty())
        );
    }

    /// First bytes UxPlay accepts for AAC-ELD frames.
    const ACCEPTED_FIRST_BYTES: [u8; 6] = [0x80, 0x81, 0x82, 0x8c, 0x8d, 0x8e];

    #[test]
    fn frames_pass_the_receiver_first_byte_check() {
        let mut seed = 0x1234_5678_u32;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            f64::from(seed >> 16) / 32768.0 - 1.0
        };
        type Signal = Box<dyn FnMut(f64) -> (f64, f64)>;
        let signals: [(&str, Signal); 4] = [
            ("silence", Box::new(|_| (0.0, 0.0))),
            (
                "tone",
                Box::new(|t| {
                    let v = (t * 440.0 * std::f64::consts::TAU).sin() * 6000.0;
                    (v, v)
                }),
            ),
            (
                "quiet",
                Box::new(|t| {
                    let v = (t * 440.0 * std::f64::consts::TAU).sin() * 60.0;
                    (v, -v)
                }),
            ),
            (
                "noise",
                Box::new(move |_| (noise() * 8000.0, noise() * 8000.0)),
            ),
        ];
        for (name, mut signal) in signals {
            for kbps in [32, 96, 128, 256] {
                let mut enc = FdkEldEncoder::new();
                enc.configure(&AudioEncoderConfig {
                    bitrate_kbps: kbps,
                    ..config(ELD_SAMPLE_RATE)
                })
                .expect("configure");
                let mut frames = Vec::new();
                for chunk in 0..100_u64 {
                    let mut pcm = Vec::with_capacity(960 * 4);
                    for i in 0..960 {
                        let t = (chunk * 960 + i) as f64 / f64::from(ELD_SAMPLE_RATE);
                        let (l, r) = signal(t);
                        pcm.extend_from_slice(&(l as i16).to_le_bytes());
                        pcm.extend_from_slice(&(r as i16).to_le_bytes());
                    }
                    enc.encode(&AudioFrame {
                        codec: AudioCodec::Pcm,
                        data: Bytes::from(pcm),
                        timestamp_us: 1_000_000 + chunk * 21_768,
                        sample_rate: ELD_SAMPLE_RATE,
                        channels: 2,
                    })
                    .expect("encode");
                    frames.extend(enc.take_output());
                }
                let rejected: Vec<u8> = frames
                    .iter()
                    .map(|f| f.data[0])
                    .filter(|b| !ACCEPTED_FIRST_BYTES.contains(b))
                    .collect();
                assert!(
                    rejected.is_empty(),
                    "{name} at {kbps} kbps: rejected first bytes {rejected:02x?}"
                );
            }
        }
    }

    #[test]
    fn resamples_48k_input() {
        let frames = encode_seconds(48_000, 960);
        assert!((85..=94).contains(&frames.len()), "{} frames", frames.len());
        assert!(frames.iter().all(|f| f.sample_rate == ELD_SAMPLE_RATE));
    }
}
