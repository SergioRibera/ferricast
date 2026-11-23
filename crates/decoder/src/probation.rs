//! Hardware decoders can accept a configuration and still reject the
//! bitstream: NVDEC only opens its decode session when the parser reaches
//! the first sequence header, and a VA-API driver may fail on the first
//! slices, so a stream it cannot handle (below the minimum size,
//! unsupported profile or chroma format) fails on the first `decode()`,
//! after `configure()` already succeeded. Until a
//! backend has produced its first picture, the facades keep the packets
//! they fed it, so a software decoder can take over from the start of
//! the stream instead of the whole session failing.

use ferricast_core::EncodedFrame;

/// A backend that has not produced a picture after this many packets
/// is decoding, just slowly to start; probation ends and nothing more
/// is kept.
const MAX_PROBATION_PACKETS: usize = 64;

#[derive(Default)]
pub(crate) struct Probation {
    packets: Vec<EncodedFrame>,
    settled: bool,
}

impl Probation {
    /// Keeps `frame` for a possible replay while still on probation.
    pub(crate) fn record(&mut self, frame: &EncodedFrame) {
        if self.settled {
            return;
        }
        if self.packets.len() == MAX_PROBATION_PACKETS {
            self.settle();
            return;
        }
        self.packets.push(frame.clone());
    }

    /// The backend produced a picture: from here on its errors are
    /// ordinary decode errors, not a rejected stream.
    pub(crate) fn settle(&mut self) {
        self.settled = true;
        self.packets = Vec::new();
    }

    /// Packets to replay into a software decoder, or `None` once the
    /// backend has proven itself. Ends probation either way.
    pub(crate) fn take_replay(&mut self) -> Option<Vec<EncodedFrame>> {
        if self.settled {
            return None;
        }
        self.settled = true;
        Some(std::mem::take(&mut self.packets))
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use ferricast_core::Codec;

    use super::*;

    fn packet(timestamp_us: u64) -> EncodedFrame {
        EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from_static(&[0, 0, 0, 1, 0x65]),
            timestamp_us,
            is_keyframe: true,
            duration_us: None,
            pts_dts: (timestamp_us, timestamp_us),
        }
    }

    fn timestamps(packets: &[EncodedFrame]) -> Vec<u64> {
        packets.iter().map(|p| p.timestamp_us).collect()
    }

    #[test]
    fn replays_every_packet_fed_before_the_failure() {
        let mut probation = Probation::default();
        for ts in [0, 33, 66] {
            probation.record(&packet(ts));
        }
        let replay = probation.take_replay().unwrap();
        assert_eq!(timestamps(&replay), [0, 33, 66]);
        assert!(probation.take_replay().is_none());
    }

    #[test]
    fn a_decoded_picture_ends_probation() {
        let mut probation = Probation::default();
        probation.record(&packet(0));
        probation.settle();
        probation.record(&packet(33));
        assert!(probation.take_replay().is_none());
    }

    #[test]
    fn a_long_silent_start_ends_probation() {
        let mut probation = Probation::default();
        for ts in 0..=MAX_PROBATION_PACKETS as u64 {
            probation.record(&packet(ts));
        }
        assert!(probation.take_replay().is_none());
    }
}
