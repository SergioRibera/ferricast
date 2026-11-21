//! Pure playlist logic of the HLS puller: variant choice, live-edge
//! start, per-segment key and byte-range resolution, AES-128
//! decryption. Kept free of I/O so it is unit-testable.

use aes::Aes128;
use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockDecrypt, KeyInit};
use ferricast_core::{FerricastError, Result};
use m3u8_rs::{AlternativeMediaType, KeyMethod, MasterPlaylist, MediaPlaylist, VariantStream};
use url::Url;

use super::fetch::ByteRange;

/// Picks the variant to play from a master playlist.
///
/// Preference order: variants whose codecs we can decode, then
/// variants with muxed audio (no external `EXT-X-MEDIA` audio
/// playlist, which this puller does not fetch), then the highest
/// bandwidth. Returns the variant and whether its audio lives in a
/// separate rendition (and will therefore be missing).
pub(super) fn choose_variant(master: &MasterPlaylist) -> Option<(&VariantStream, bool)> {
    let external_audio = |v: &VariantStream| {
        v.audio.as_ref().is_some_and(|group| {
            master.alternatives.iter().any(|alt| {
                alt.media_type == AlternativeMediaType::Audio
                    && &alt.group_id == group
                    && alt.uri.is_some()
            })
        })
    };
    let decodable = |v: &VariantStream| {
        v.codecs.as_deref().is_none_or(|codecs| {
            codecs
                .split(',')
                .map(str::trim)
                .any(|c| c.starts_with("avc1") || c.starts_with("hvc1") || c.starts_with("hev1"))
        })
    };
    master
        .variants
        .iter()
        .filter(|v| !v.is_i_frame)
        .max_by_key(|v| (decodable(v), !external_audio(v), v.bandwidth))
        .map(|v| (v, external_audio(v)))
}

/// Rejects playlists using features this puller cannot play, with a
/// message naming the feature.
pub(super) fn check_supported(pl: &MediaPlaylist) -> Result<()> {
    for seg in &pl.segments {
        if let Some(key) = &seg.key {
            match &key.method {
                KeyMethod::None => {}
                KeyMethod::AES128 => {
                    if key.uri.is_none() {
                        return Err(FerricastError::Pull(
                            "EXT-X-KEY METHOD=AES-128 without URI".into(),
                        ));
                    }
                    if key.keyformat.as_deref().is_some_and(|f| f != "identity") {
                        return Err(FerricastError::Pull(format!(
                            "unsupported HLS stream: DRM key format {:?}",
                            key.keyformat
                        )));
                    }
                }
                KeyMethod::SampleAES => {
                    return Err(FerricastError::Pull(
                        "unsupported HLS stream: SAMPLE-AES encryption".into(),
                    ));
                }
                KeyMethod::Other(m) => {
                    return Err(FerricastError::Pull(format!(
                        "unsupported HLS stream: encryption method {m}"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Index of the segment a live client should start at: the last one
/// that starts at least three target durations before the end of the
/// playlist (RFC 8216 §6.3.3).
pub(super) fn live_start_index(pl: &MediaPlaylist) -> usize {
    let window = 3.0 * pl.target_duration as f64;
    let mut tail = 0.0;
    for (idx, seg) in pl.segments.iter().enumerate().rev() {
        tail += f64::from(seg.duration);
        if tail >= window {
            return idx;
        }
    }
    0
}

/// Index of the VOD segment containing `pos_us` and the time that
/// segment starts at.
pub(super) fn segment_for_position(pl: &MediaPlaylist, pos_us: u64) -> Option<(usize, u64)> {
    let mut start_us: u64 = 0;
    for (idx, seg) in pl.segments.iter().enumerate() {
        let end_us = start_us + secs_to_us(f64::from(seg.duration));
        if pos_us < end_us {
            return Some((idx, start_us));
        }
        start_us = end_us;
    }
    None
}

pub(super) fn total_duration_us(pl: &MediaPlaylist) -> Option<u64> {
    if !pl.end_list {
        return None;
    }
    let secs: f64 = pl.segments.iter().map(|s| f64::from(s.duration)).sum();
    Some(secs_to_us(secs))
}

fn secs_to_us(secs: f64) -> u64 {
    // Durations are non-negative and far below u64::MAX µs.
    (secs.max(0.0) * 1_000_000.0) as u64
}

/// Encryption parameters of one segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SegmentKey {
    pub(super) uri: Url,
    pub(super) iv: [u8; 16],
}

/// Everything needed to fetch and decode one segment.
#[derive(Debug, Clone)]
pub(super) struct SegmentRef {
    pub(super) url: Url,
    pub(super) range: Option<ByteRange>,
    pub(super) key: Option<SegmentKey>,
    pub(super) discontinuity: bool,
    /// Initialization segment (`EXT-X-MAP`) of fragmented MP4 media;
    /// `None` for MPEG-TS.
    pub(super) map: Option<InitSegment>,
}

/// Where an `EXT-X-MAP` initialization segment is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct InitSegment {
    pub(super) url: Url,
    pub(super) range: Option<ByteRange>,
}

/// Resolves segment `idx` of `pl` against the playlist's (final) URL.
/// The key and byte-range offset carry over from earlier segments,
/// so they are found by scanning from the start.
pub(super) fn resolve_segment(pl: &MediaPlaylist, base: &Url, idx: usize) -> Result<SegmentRef> {
    let seg = pl
        .segments
        .get(idx)
        .ok_or_else(|| FerricastError::Pull(format!("segment index {idx} out of range")))?;
    let url = base
        .join(&seg.uri)
        .map_err(|e| FerricastError::Pull(format!("segment URL join {}: {e}", seg.uri)))?;

    let mut next_offset = 0u64;
    let mut range = None;
    let mut key = None;
    let mut map = None;
    for s in &pl.segments[..=idx] {
        if let Some(k) = &s.key {
            key = Some(k);
        }
        // An EXT-X-MAP applies to every segment after it until the next.
        if let Some(m) = &s.map {
            map = Some(m);
        }
        range = s.byte_range.as_ref().map(|br| {
            let offset = br.offset.unwrap_or(next_offset);
            next_offset = offset + br.length;
            ByteRange {
                offset,
                length: br.length,
            }
        });
    }

    let key = match key {
        Some(k) if k.method == KeyMethod::AES128 => {
            let uri = k
                .uri
                .as_deref()
                .ok_or_else(|| FerricastError::Pull("EXT-X-KEY without URI".into()))?;
            let uri = base
                .join(uri)
                .map_err(|e| FerricastError::Pull(format!("key URL join {uri}: {e}")))?;
            let iv = match k.iv.as_deref() {
                Some(hex) => parse_iv(hex)?,
                // No IV attribute: the media sequence number,
                // big-endian, is the IV (RFC 8216 §5.2).
                None => u128::from(pl.media_sequence + idx as u64).to_be_bytes(),
            };
            Some(SegmentKey { uri, iv })
        }
        _ => None,
    };

    let map = map
        .map(|m| -> Result<InitSegment> {
            let url = base
                .join(&m.uri)
                .map_err(|e| FerricastError::Pull(format!("EXT-X-MAP URL join {}: {e}", m.uri)))?;
            let range = m.byte_range.as_ref().map(|br| ByteRange {
                offset: br.offset.unwrap_or(0),
                length: br.length,
            });
            Ok(InitSegment { url, range })
        })
        .transpose()?;

    Ok(SegmentRef {
        url,
        range,
        key,
        discontinuity: seg.discontinuity,
        map,
    })
}

fn parse_iv(hex: &str) -> Result<[u8; 16]> {
    let digits = hex
        .strip_prefix("0x")
        .or_else(|| hex.strip_prefix("0X"))
        .unwrap_or(hex);
    let value = u128::from_str_radix(digits, 16)
        .map_err(|e| FerricastError::Pull(format!("EXT-X-KEY IV {hex}: {e}")))?;
    Ok(value.to_be_bytes())
}

/// Decrypts an AES-128-CBC segment with PKCS#7 padding.
pub(super) fn decrypt_aes128(key: &[u8; 16], iv: &[u8; 16], data: &[u8]) -> Result<Vec<u8>> {
    if data.is_empty() || !data.len().is_multiple_of(16) {
        return Err(FerricastError::Pull(format!(
            "encrypted segment length {} is not a multiple of 16",
            data.len()
        )));
    }
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let mut out = Vec::with_capacity(data.len());
    let mut prev = *iv;
    for block in data.chunks_exact(16) {
        let mut buf = GenericArray::clone_from_slice(block);
        cipher.decrypt_block(&mut buf);
        out.extend(buf.iter().zip(prev).map(|(b, p)| b ^ p));
        prev.copy_from_slice(block);
    }
    let pad = usize::from(*out.last().unwrap_or(&0));
    if pad == 0
        || pad > 16
        || out[out.len() - pad..]
            .iter()
            .any(|&b| usize::from(b) != pad)
    {
        return Err(FerricastError::Pull(
            "bad PKCS#7 padding — wrong AES-128 key?".into(),
        ));
    }
    out.truncate(out.len() - pad);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes::cipher::BlockEncrypt;

    fn media(text: &str) -> MediaPlaylist {
        m3u8_rs::parse_media_playlist_res(text.as_bytes()).unwrap()
    }

    #[test]
    fn live_start_is_three_target_durations_from_the_end() {
        let mut text = String::from("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:40\n");
        for i in 0..12 {
            text.push_str(&format!("#EXTINF:2.0,\ns{i}.ts\n"));
        }
        let pl = media(&text);
        // 3 × 2 s = 6 s → the last three segments.
        assert_eq!(live_start_index(&pl), 9);
    }

    #[test]
    fn short_live_playlist_starts_at_the_beginning() {
        let pl = media("#EXTM3U\n#EXT-X-TARGETDURATION:6\n#EXTINF:6.0,\na.ts\n");
        assert_eq!(live_start_index(&pl), 0);
    }

    #[test]
    fn key_and_byte_range_carry_over() {
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:7\n\
             #EXT-X-KEY:METHOD=AES-128,URI=\"k.bin\"\n\
             #EXT-X-BYTERANGE:100@0\n#EXTINF:4.0,\nall.ts\n\
             #EXT-X-BYTERANGE:50\n#EXTINF:4.0,\nall.ts\n",
        );
        let base = Url::parse("http://host/dir/pl.m3u8").unwrap();
        let second = resolve_segment(&pl, &base, 1).unwrap();
        assert_eq!(second.url.as_str(), "http://host/dir/all.ts");
        assert_eq!(
            second.range,
            Some(ByteRange {
                offset: 100,
                length: 50
            })
        );
        let key = second.key.unwrap();
        assert_eq!(key.uri.as_str(), "http://host/dir/k.bin");
        assert_eq!(key.iv, 8u128.to_be_bytes());
    }

    #[test]
    fn fmp4_maps_carry_over_and_sample_aes_is_rejected() {
        let fmp4 = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MAP:URI=\"init.mp4\"\n#EXTINF:4.0,\na.m4s\n\
             #EXTINF:4.0,\nb.m4s\n",
        );
        assert!(check_supported(&fmp4).is_ok());
        let base = Url::parse("http://host/dir/index.m3u8").unwrap();
        let second = resolve_segment(&fmp4, &base, 1).unwrap();
        assert_eq!(
            second.map.map(|m| m.url.to_string()).as_deref(),
            Some("http://host/dir/init.mp4")
        );
        let sample_aes = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"k\"\n\
             #EXTINF:4.0,\na.ts\n",
        );
        assert!(check_supported(&sample_aes).is_err());
    }

    #[test]
    fn variant_with_muxed_audio_is_preferred() {
        let text = "#EXTM3U\n\
            #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aud\",NAME=\"en\",URI=\"audio.m3u8\"\n\
            #EXT-X-STREAM-INF:BANDWIDTH=9000000,CODECS=\"avc1.640028\",AUDIO=\"aud\"\n\
            hi.m3u8\n\
            #EXT-X-STREAM-INF:BANDWIDTH=3000000,CODECS=\"avc1.640028,mp4a.40.2\"\n\
            mid.m3u8\n\
            #EXT-X-STREAM-INF:BANDWIDTH=12000000,CODECS=\"vp09.00.10.08\"\n\
            vp9.m3u8\n";
        let master = m3u8_rs::parse_master_playlist_res(text.as_bytes()).unwrap();
        let (v, external) = choose_variant(&master).unwrap();
        assert_eq!(v.uri, "mid.m3u8");
        assert!(!external);
    }

    #[test]
    fn aes128_cbc_round_trip() {
        let key = [7u8; 16];
        let iv = [9u8; 16];
        let plain: Vec<u8> = (0..40).collect();
        // Encrypt with PKCS#7 by hand.
        let mut padded = plain.clone();
        let pad = 16 - plain.len() % 16;
        padded.extend(std::iter::repeat_n(pad as u8, pad));
        let cipher = Aes128::new(GenericArray::from_slice(&key));
        let mut prev = iv;
        let mut enc = Vec::new();
        for block in padded.chunks_exact(16) {
            let mut buf = GenericArray::clone_from_slice(block);
            buf.iter_mut().zip(prev).for_each(|(b, p)| *b ^= p);
            cipher.encrypt_block(&mut buf);
            prev.copy_from_slice(&buf);
            enc.extend_from_slice(&buf);
        }
        assert_eq!(decrypt_aes128(&key, &iv, &enc).unwrap(), plain);
        assert!(decrypt_aes128(&[0u8; 16], &iv, &enc).is_err());
    }

    #[test]
    fn position_maps_to_segment_start() {
        let pl = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4.0,\na.ts\n#EXTINF:4.0,\nb.ts\n\
             #EXT-X-ENDLIST\n",
        );
        assert_eq!(segment_for_position(&pl, 5_000_000), Some((1, 4_000_000)));
        assert_eq!(segment_for_position(&pl, 9_000_000), None);
        assert_eq!(total_duration_us(&pl), Some(8_000_000));
    }
}
