//! Packets of each codec as VideoToolbox samples, and the format
//! description VideoToolbox decodes them against.
//!
//! - H.264 / H.265: Annex B in, length-prefixed (4-byte) NAL units out;
//!   the description comes from the in-band parameter sets.
//! - VP9: frames (or superframes) pass through; the description carries a
//!   `vpcC` record built from the latest keyframe header.
//! - AV1: temporal units pass through as sized OBUs without temporal
//!   delimiters; the description carries an `av1C` record with the latest
//!   sequence header.

use std::ptr::{self, NonNull};

use ferricast_codec::bitstream::split_annex_b;
use ferricast_core::{Codec, CodecError, CodecResult, CodecTarget, ColourDescription, Operation};
use objc2_core_foundation::{CFData, CFDictionary, CFRetained, CFString, CFType};
use objc2_core_media::{
    CMFormatDescription, CMVideoCodecType, CMVideoFormatDescriptionCreate,
    CMVideoFormatDescriptionCreateFromH264ParameterSets,
    CMVideoFormatDescriptionCreateFromHEVCParameterSets,
    kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms, kCMVideoCodecType_AV1,
    kCMVideoCodecType_H264, kCMVideoCodecType_HEVC, kCMVideoCodecType_VP9,
};
use tracing::debug;

use super::{BACKEND, fail};
use crate::platform::apple::os_status;
use ferricast_codec::bitstream::{av1, h264, hevc, vp9};

/// The VideoToolbox codec type of `codec`.
pub(super) fn codec_type(codec: Codec) -> CodecResult<CMVideoCodecType> {
    match codec {
        Codec::H264 => Ok(kCMVideoCodecType_H264),
        Codec::H265 => Ok(kCMVideoCodecType_HEVC),
        Codec::Vp9 => Ok(kCMVideoCodecType_VP9),
        Codec::Av1 => Ok(kCMVideoCodecType_AV1),
        other => Err(CodecError::BackendCannotHandle {
            backend: BACKEND,
            target: CodecTarget::decode(other),
        }),
    }
}

/// A stream configuration as VideoToolbox describes it.
pub(super) struct Format {
    pub(super) description: CFRetained<CMFormatDescription>,
    /// Luma bit depth: 8 decodes to NV12, more to P010.
    pub(super) depth: u8,
    /// Bumped whenever the configuration changes.
    pub(super) generation: u64,
}

/// One codec's packets turned into samples.
pub(super) enum Stream {
    Nal(Nal),
    Vp9(Vp9),
    Av1(Av1),
}

impl Stream {
    pub(super) fn new(codec: Codec) -> CodecResult<Self> {
        Ok(match codec {
            Codec::H264 | Codec::H265 => Self::Nal(Nal::new(codec)),
            Codec::Vp9 => Self::Vp9(Vp9::default()),
            Codec::Av1 => Self::Av1(Av1::default()),
            other => {
                return Err(CodecError::BackendCannotHandle {
                    backend: BACKEND,
                    target: CodecTarget::decode(other),
                });
            }
        })
    }

    pub(super) fn codec(&self) -> Codec {
        match self {
            Self::Nal(nal) => nal.codec,
            Self::Vp9(_) => Codec::Vp9,
            Self::Av1(_) => Codec::Av1,
        }
    }

    /// The sample to decode from `packet`, after taking in any new
    /// configuration it carries; `None` while there is nothing to decode
    /// (configuration only, or nothing decodable before it).
    pub(super) fn sample(
        &mut self,
        packet: &[u8],
        colour: Option<ColourDescription>,
    ) -> CodecResult<Option<Vec<u8>>> {
        match self {
            Self::Nal(nal) => nal.sample(packet),
            Self::Vp9(vp9) => vp9.sample(packet, colour),
            Self::Av1(av1) => av1.sample(packet),
        }
    }

    /// The configuration the latest sample decodes against.
    pub(super) fn format(&self) -> Option<&Format> {
        match self {
            Self::Nal(nal) => nal.format.as_ref(),
            Self::Vp9(vp9) => vp9.format.as_ref(),
            Self::Av1(av1) => av1.format.as_ref(),
        }
    }
}

/// The next generation after `format`'s.
fn next_generation(format: Option<&Format>) -> u64 {
    format.map_or(1, |format| format.generation + 1)
}

/// Where an H.264 / H.265 NAL unit goes.
enum NalKind {
    ParameterSet(usize),
    Skip,
    Slice,
}

fn classify(codec: Codec, nal: &[u8]) -> NalKind {
    let Some(&header) = nal.first() else {
        return NalKind::Skip;
    };
    match codec {
        Codec::H265 => match (header >> 1) & 0x3f {
            32 => NalKind::ParameterSet(0),
            33 => NalKind::ParameterSet(1),
            34 => NalKind::ParameterSet(2),
            // Access unit delimiter, end of sequence / bitstream.
            35..=37 => NalKind::Skip,
            _ => NalKind::Slice,
        },
        _ => match header & 0x1f {
            7 => NalKind::ParameterSet(0),
            8 => NalKind::ParameterSet(1),
            9..=11 => NalKind::Skip,
            _ => NalKind::Slice,
        },
    }
}

/// H.264 / H.265.
pub(super) struct Nal {
    codec: Codec,
    /// The latest parameter sets, by slot (SPS, PPS; VPS, SPS, PPS).
    sets: Vec<Option<Vec<u8>>>,
    /// The parameter sets `format` was built from.
    built_from: Vec<Vec<u8>>,
    format: Option<Format>,
}

impl Nal {
    fn new(codec: Codec) -> Self {
        let slots = if codec == Codec::H265 { 3 } else { 2 };
        Self {
            codec,
            sets: vec![None; slots],
            built_from: Vec::new(),
            format: None,
        }
    }

    fn sample(&mut self, packet: &[u8]) -> CodecResult<Option<Vec<u8>>> {
        let mut slices = Vec::new();
        for nal in split_annex_b(packet) {
            match classify(self.codec, nal) {
                NalKind::ParameterSet(slot) => {
                    if let Some(entry) = self.sets.get_mut(slot) {
                        *entry = Some(nal.to_vec());
                    }
                }
                NalKind::Skip => {}
                NalKind::Slice => slices.push(nal),
            }
        }
        if slices.is_empty() {
            return Ok(None);
        }
        let Some(sets) = self.sets.iter().cloned().collect::<Option<Vec<_>>>() else {
            debug!(codec = %self.codec, "VideoToolbox: slice before the parameter sets; skipped");
            return Ok(None);
        };
        if self.format.is_none() || self.built_from != sets {
            let sps = &sets[if self.codec == Codec::H265 { 1 } else { 0 }];
            let depth = match self.codec {
                Codec::H265 => hevc::sps_bit_depth(sps).map(|(luma, _)| luma),
                _ => h264::sps_bit_depth(sps),
            }
            .unwrap_or(8);
            self.format = Some(Format {
                description: parameter_set_description(self.codec, &sets)?,
                depth,
                generation: next_generation(self.format.as_ref()),
            });
            self.built_from = sets;
        }
        let mut avcc = Vec::with_capacity(slices.iter().map(|nal| nal.len() + 4).sum());
        for nal in slices {
            let len = u32::try_from(nal.len())
                .map_err(|_| fail(Operation::Decode, "NAL unit too long"))?;
            avcc.extend_from_slice(&len.to_be_bytes());
            avcc.extend_from_slice(nal);
        }
        Ok(Some(avcc))
    }
}

/// A format description from Annex-B-free parameter sets, for 4-byte
/// NAL lengths.
fn parameter_set_description(
    codec: Codec,
    sets: &[Vec<u8>],
) -> CodecResult<CFRetained<CMFormatDescription>> {
    let mut pointers: Vec<NonNull<u8>> = sets
        .iter()
        .map(|set| {
            NonNull::new(set.as_ptr().cast_mut())
                .ok_or_else(|| fail(Operation::Decode, "an empty parameter set"))
        })
        .collect::<CodecResult<_>>()?;
    let mut sizes: Vec<usize> = sets.iter().map(Vec::len).collect();
    let mut raw: *const CMFormatDescription = ptr::null();
    let pointers_ptr = NonNull::new(pointers.as_mut_ptr())
        .ok_or_else(|| fail(Operation::Decode, "no parameter sets"))?;
    let sizes_ptr = NonNull::new(sizes.as_mut_ptr())
        .ok_or_else(|| fail(Operation::Decode, "no parameter sets"))?;
    // SAFETY: `pointers` / `sizes` describe `sets.len()` live parameter
    // sets; the out pointer is a valid local.
    let status = unsafe {
        match codec {
            Codec::H265 => CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                None,
                sets.len(),
                pointers_ptr,
                sizes_ptr,
                4,
                None,
                NonNull::from(&mut raw),
            ),
            _ => CMVideoFormatDescriptionCreateFromH264ParameterSets(
                None,
                sets.len(),
                pointers_ptr,
                sizes_ptr,
                4,
                NonNull::from(&mut raw),
            ),
        }
    };
    created(status, raw, "its parameter sets are not accepted")
}

/// A format description of `codec_type` carrying `record` as the
/// sample description extension atom `atom` (`vpcC`, `av1C`).
fn atom_description(
    codec: Codec,
    width: u32,
    height: u32,
    atom: &str,
    record: &[u8],
) -> CodecResult<CFRetained<CMFormatDescription>> {
    let codec_type = codec_type(codec)?;
    let (width, height) = (
        i32::try_from(width).map_err(|_| fail(Operation::Decode, "width exceeds i32"))?,
        i32::try_from(height).map_err(|_| fail(Operation::Decode, "height exceeds i32"))?,
    );
    let name = CFString::from_str(atom);
    let data = CFData::from_bytes(record);
    let atoms = CFDictionary::<CFString, CFType>::from_slices(&[&*name], &[data.as_ref()]);
    // SAFETY: CoreMedia exports this key for the life of the process.
    let atoms_key = unsafe { kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms };
    let extensions =
        CFDictionary::<CFString, CFType>::from_slices(&[atoms_key], &[(*atoms).as_ref()]);
    let mut raw: *const CMFormatDescription = ptr::null();
    // SAFETY: the dictionary is live for the call; the out pointer is a
    // valid local.
    let status = unsafe {
        CMVideoFormatDescriptionCreate(
            None,
            codec_type,
            width,
            height,
            Some(extensions.as_ref()),
            NonNull::from(&mut raw),
        )
    };
    created(status, raw, "its configuration is not accepted")
}

/// The description a create call returned, as a codec error when it
/// refused.
fn created(
    status: i32,
    raw: *const CMFormatDescription,
    refused: &str,
) -> CodecResult<CFRetained<CMFormatDescription>> {
    if status != 0 {
        return Err(CodecError::UnsupportedStream {
            backend: BACKEND,
            reason: format!("{refused}: {}", os_status(status)),
        });
    }
    let raw = NonNull::new(raw.cast_mut())
        .ok_or_else(|| fail(Operation::Decode, "no format description was created"))?;
    // SAFETY: the create call handed over a +1 reference.
    Ok(unsafe { CFRetained::from_raw(raw) })
}

/// VP9.
#[derive(Default)]
pub(super) struct Vp9 {
    keyframe: Option<vp9::Keyframe>,
    format: Option<Format>,
}

impl Vp9 {
    fn sample(
        &mut self,
        packet: &[u8],
        colour: Option<ColourDescription>,
    ) -> CodecResult<Option<Vec<u8>>> {
        if packet.is_empty() {
            return Ok(None);
        }
        if let Some(keyframe) = vp9::keyframe(packet)
            && self.keyframe != Some(keyframe)
        {
            if keyframe.subsampling_x != keyframe.subsampling_y || !keyframe.subsampling_x {
                return Err(CodecError::UnsupportedStream {
                    backend: BACKEND,
                    reason: format!("VP9 profile {} with 4:2:2 / 4:4:4 chroma", keyframe.profile),
                });
            }
            let record = keyframe.vpcc(Some(keyframe.colour(colour)));
            self.format = Some(Format {
                description: atom_description(
                    Codec::Vp9,
                    keyframe.width,
                    keyframe.height,
                    "vpcC",
                    &record,
                )?,
                depth: keyframe.bit_depth,
                generation: next_generation(self.format.as_ref()),
            });
            self.keyframe = Some(keyframe);
        }
        if self.format.is_none() {
            debug!("VideoToolbox: VP9 frame before the first keyframe; skipped");
            return Ok(None);
        }
        Ok(Some(packet.to_vec()))
    }
}

/// AV1.
#[derive(Default)]
pub(super) struct Av1 {
    /// The sized sequence header OBU `format` was built from.
    sequence: Vec<u8>,
    format: Option<Format>,
}

impl Av1 {
    fn sample(&mut self, packet: &[u8]) -> CodecResult<Option<Vec<u8>>> {
        let obus = av1::obus(packet)
            .ok_or_else(|| fail(Operation::Decode, "a malformed AV1 temporal unit"))?;
        if let Some(sequence) = obus.iter().find(|obu| obu.kind == av1::SEQUENCE_HEADER) {
            let sized = sequence.sized();
            if sized != self.sequence {
                let header = av1::SequenceHeader::parse(sequence.payload)
                    .ok_or_else(|| fail(Operation::Decode, "a malformed AV1 sequence header"))?;
                if header.monochrome || !(header.subsampling_x && header.subsampling_y) {
                    return Err(CodecError::UnsupportedStream {
                        backend: BACKEND,
                        reason: format!("AV1 profile {} that is not 4:2:0 colour", header.profile),
                    });
                }
                self.format = Some(Format {
                    description: atom_description(
                        Codec::Av1,
                        header.max_width,
                        header.max_height,
                        "av1C",
                        &header.av1c(&sized),
                    )?,
                    depth: header.bit_depth,
                    generation: next_generation(self.format.as_ref()),
                });
                self.sequence = sized;
            }
        }
        if self.format.is_none() {
            debug!("VideoToolbox: AV1 temporal unit before the sequence header; skipped");
            return Ok(None);
        }
        let sample = av1::sample(&obus);
        Ok((!sample.is_empty()).then_some(sample))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_units_are_sorted_into_parameter_sets_and_slices() {
        assert!(matches!(
            classify(Codec::H264, &[0x67]),
            NalKind::ParameterSet(0)
        ));
        assert!(matches!(
            classify(Codec::H264, &[0x68]),
            NalKind::ParameterSet(1)
        ));
        assert!(matches!(classify(Codec::H264, &[0x09]), NalKind::Skip));
        assert!(matches!(classify(Codec::H264, &[0x65]), NalKind::Slice));
        assert!(matches!(
            classify(Codec::H265, &[0x40]),
            NalKind::ParameterSet(0)
        ));
        assert!(matches!(
            classify(Codec::H265, &[0x44]),
            NalKind::ParameterSet(2)
        ));
        assert!(matches!(classify(Codec::H265, &[0x46]), NalKind::Skip));
        assert!(matches!(classify(Codec::H265, &[0x26]), NalKind::Slice));
    }
}
