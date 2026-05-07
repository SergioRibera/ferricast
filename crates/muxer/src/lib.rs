//! Container muxers used by the casting pipelines.
//!
//! Currently the only consumer is HLS, which always writes
//! [`mpeg_ts::MpegTs`] segments. The trait is kept narrow on purpose
//! — every method maps 1:1 onto what the segmenter needs.

use ferricast_core::{EncodedFrame, FerricastError};

pub mod mpeg_ts;

/// One muxed elementary stream, emitting bytes through [`Self::drain`].
///
/// A muxer is a *single-use* state machine: instantiate, [`config`] once
/// with the codec's parameter sets, push frames with [`add_frame`], then
/// [`drain`] to consume the buffered bytes. For HLS, a fresh muxer is
/// built per segment so each segment is independently decodable.
pub trait Muxer {
    /// Provide the codec parameter sets (e.g. H.264 SPS+PPS in Annex B
    /// form) that must precede each random-access point.
    fn config(&mut self, parameter_sets: Vec<u8>) -> Result<(), FerricastError>;

    /// Append one encoded frame.
    ///
    /// `pts_90k` and `dts_90k` are MPEG-2 system clock ticks (90 kHz).
    /// For codecs without frame reordering (H.264 Baseline, used here)
    /// `dts_90k == pts_90k`.
    fn add_frame(
        &mut self,
        frame: &EncodedFrame,
        pts_90k: u64,
        dts_90k: u64,
    ) -> Result<(), FerricastError>;

    /// Take the bytes accumulated so far. Subsequent calls return the
    /// next chunk; calling after the final frame returns the rest.
    fn drain(&mut self) -> Vec<u8>;
}
