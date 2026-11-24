//! Streaming sample-rate converter for interleaved S16 PCM.
//!
//! Four-point cubic Hermite interpolation. Cheap and free of the
//! zipper noise of linear interpolation; adequate for converting
//! between the 44.1 / 48 kHz family when the capture side could not
//! deliver the codec's rate directly.

/// Converts one interleaved PCM stream from `in_rate` to `out_rate`,
/// keeping its phase across calls.
pub(crate) struct Resampler {
    channels: usize,
    /// Input frames advanced per output frame (`in_rate / out_rate`).
    step: f64,
    /// Position of the next output frame, in input frames from the
    /// start of `history`.
    pos: f64,
    /// Interleaved input not fully consumed yet, with one frame of
    /// look-behind before `pos`.
    history: Vec<f32>,
}

impl Resampler {
    pub(crate) fn new(in_rate: u32, out_rate: u32, channels: u16) -> Self {
        let channels = usize::from(channels.max(1));
        Self {
            channels,
            step: f64::from(in_rate.max(1)) / f64::from(out_rate.max(1)),
            // One silent frame of look-behind so the first output sample
            // has a left neighbour.
            pos: 1.0,
            history: vec![0.0; channels],
        }
    }

    /// Resample `input` (interleaved) and append the result to `out`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "frame counts of a PCM chunk are far below 2^52"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "the Hermite coefficients keep their textbook names"
    )]
    pub(crate) fn process(&mut self, input: &[i16], out: &mut Vec<i16>) {
        self.history.extend(input.iter().map(|&s| f32::from(s)));
        let frames = self.history.len() / self.channels;
        let ch = self.channels;
        // Output frame at `pos` needs input frames floor(pos) - 1 ..= floor(pos) + 2.
        while self.pos + 2.0 < frames as f64 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "pos is positive and bounded by the history length"
            )]
            let i = self.pos as usize;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the fractional part is in [0, 1)"
            )]
            let t = (self.pos - i as f64) as f32;
            for c in 0..ch {
                let at = |k: usize| self.history[k * ch + c];
                let (y0, y1, y2, y3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
                let a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
                let b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                let c1 = -0.5 * y0 + 0.5 * y2;
                let v = ((a * t + b) * t + c1) * t + y1;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "clamped to the i16 range first"
                )]
                out.push(v.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
            }
            self.pos += self.step;
        }
        // Drop input that no future output frame can reach, keeping one
        // frame of look-behind.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "pos is positive and bounded by the history length"
        )]
        let keep_from = (self.pos as usize).saturating_sub(1).min(frames);
        if keep_from > 0 {
            self.history.drain(..keep_from * ch);
            self.pos -= keep_from as f64;
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test signal generation: sample values are bounded by construction"
)]
mod tests {
    use super::*;

    #[test]
    fn output_length_follows_the_rate_ratio() {
        let mut r = Resampler::new(48_000, 44_100, 2);
        let mut out = Vec::new();
        // One second in uneven chunks.
        for chunk in [1024_usize, 333, 4096, 48_000 - 1024 - 333 - 4096] {
            r.process(&vec![0_i16; chunk * 2], &mut out);
        }
        let frames = out.len() / 2;
        assert!((44_095..=44_100).contains(&frames), "{frames} frames");
    }

    #[test]
    fn preserves_a_low_tone() {
        let tone = |rate: f64, n: usize| {
            (0..n)
                .flat_map(|i| {
                    let v = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / rate).sin() * 10_000.0;
                    let v = v as i16;
                    [v, v]
                })
                .collect::<Vec<i16>>()
        };
        let mut r = Resampler::new(48_000, 44_100, 2);
        let mut out = Vec::new();
        r.process(&tone(48_000.0, 4800), &mut out);
        let expected = tone(44_100.0, out.len() / 2);
        // Skip the start-up frame; the rest tracks the ideal tone.
        let worst = out
            .iter()
            .zip(&expected)
            .skip(8)
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .unwrap();
        assert!(worst < 60, "max error {worst}");
    }
}
