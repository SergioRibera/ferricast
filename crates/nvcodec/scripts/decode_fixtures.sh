#!/usr/bin/env bash
# Regenerates the NVDEC test clips under tests/fixtures/ with FFmpeg, then
# the reference hashes the decode tests compare against (FFmpeg's own decode
# of each clip, see `reference_hashes` in tests/decode.rs).
#
# Clips are FFmpeg `testsrc2` pictures, small enough to commit:
#   *.h264 / *.hevc   Annex B with an access unit delimiter in front of every
#                     access unit (how the tests split packets), plus a
#                     `.pts` sidecar: each packet's presentation time in
#                     microseconds, in decode order
#   *.ivf             VP8 / VP9 / AV1: one frame (VP8 / VP9, superframes
#                     included) or temporal unit (AV1) per IVF frame
#
# Needs FFmpeg with libx264, libx265, libvpx and libsvtav1. Without ffmpeg on
# PATH but with Nix, it re-runs itself inside `nix shell nixpkgs#ffmpeg-full`.
# Run it from the workspace devshell (cargo is needed for the hashes):
#
#   nix develop -c crates/nvcodec/scripts/decode_fixtures.sh
set -euo pipefail

die() {
    echo "decode_fixtures.sh: $*" >&2
    exit 1
}

if [[ -z ${FERRICAST_FFMPEG_DIR:-} ]] && ! command -v ffmpeg >/dev/null 2>&1; then
    if command -v nix >/dev/null 2>&1 && [[ -z ${FERRICAST_FIXTURES_REEXEC:-} ]]; then
        # The devshell's LD_LIBRARY_PATH breaks the glibc of nixpkgs binaries;
        # ffmpeg runs without it, cargo keeps it.
        ffmpeg_bin=$(env -u LD_LIBRARY_PATH nix build --no-link --print-out-paths nixpkgs#ffmpeg-full.bin)/bin
        exec env FERRICAST_FIXTURES_REEXEC=1 FERRICAST_FFMPEG_DIR="$ffmpeg_bin" "$0" "$@"
    fi
    die "ffmpeg not found on PATH"
fi

crate=$(cd "$(dirname "$0")/.." && pwd)
out="$crate/tests/fixtures"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$out"

ff() {
    env -u LD_LIBRARY_PATH "${FERRICAST_FFMPEG_DIR:+$FERRICAST_FFMPEG_DIR/}ffmpeg" \
        -hide_banner -loglevel error -y "$@"
}
probe() {
    env -u LD_LIBRARY_PATH "${FERRICAST_FFMPEG_DIR:+$FERRICAST_FFMPEG_DIR/}ffprobe" \
        -hide_banner -loglevel error "$@"
}
src() { # size frames
    echo -f lavfi -i "testsrc2=size=$1:rate=30" -frames:v "$2"
}

# Encodes into Matroska (for the timestamps), then extracts Annex B and the
# decode-order presentation times in microseconds (Matroska counts in ms).
# annexb <name> <bsf> <ffmpeg args...>
annexb() {
    local name=$1 bsf=$2
    shift 2
    ff "$@" "$work/$name.mkv"
    ff -i "$work/$name.mkv" -c copy -bsf:v "$bsf" -f "${name##*.}" "$work/$name"
    probe -select_streams v -show_entries packet=pts -of csv=p=0 "$work/$name.mkv" |
        awk '{ print $1 * 1000 }' >"$work/$name.pts"
}

x264=(-c:v libx264 -preset medium -crf 32 -pix_fmt yuv420p)
x265_params=aud=1:bframes=3:keyint=30:log-level=error

# H.264 with B-frames (pyramid), the timestamp-order test.
annexb h264_bframes.h264 h264_mp4toannexb $(src 320x240 30) "${x264[@]}" \
    -bf 3 -x264-params aud=1:keyint=30:b-pyramid=normal
# H.264 cropped on all four edges: coded 400x304, displayed 392x294 from
# (8, 6) (x264 crops the bottom to 300, the bitstream filter adds left / top).
annexb h264_crop.h264 h264_mp4toannexb,h264_metadata=crop_left=8:crop_top=6 \
    $(src 400x300 10) "${x264[@]}" -bf 2 -x264-params aud=1
# Three H.264 sequences of different sizes back to back: 320x240, then
# smaller (reconfigure), then larger (new decoder).
: >"$work/res_change.h264"
: >"$work/res_change.h264.pts"
offset=0
for size in 320x240 176x144 640x360; do
    annexb "seg_$size.h264" h264_mp4toannexb $(src "$size" 10) "${x264[@]}" -bf 2 \
        -x264-params aud=1
    cat "$work/seg_$size.h264" >>"$work/res_change.h264"
    awk -v o="$offset" '{ print $1 + o }' "$work/seg_$size.h264.pts" >>"$work/res_change.h264.pts"
    offset=$((offset + 1000000))
done

annexb hevc_main8.hevc hevc_mp4toannexb $(src 256x144 30) -c:v libx265 -preset medium -crf 32 \
    -pix_fmt yuv420p -x265-params "$x265_params"
annexb hevc_main10_hdr10.hevc hevc_mp4toannexb $(src 256x144 30) -c:v libx265 -preset medium \
    -crf 32 -pix_fmt yuv420p10le -color_primaries bt2020 -color_trc smpte2084 \
    -colorspace bt2020nc -x265-params "$x265_params:hdr10=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400"
annexb hevc_444.hevc hevc_mp4toannexb $(src 256x144 10) -c:v libx265 -preset medium -crf 32 \
    -pix_fmt yuv444p -x265-params "$x265_params"

vpx=(-deadline good -cpu-used 4 -b:v 200k)
ff $(src 256x144 30) -c:v libvpx "${vpx[@]}" -pix_fmt yuv420p -auto-alt-ref 1 "$work/vp8.ivf"
ff $(src 256x144 30) -c:v libvpx-vp9 "${vpx[@]}" -pix_fmt yuv420p "$work/vp9_profile0.ivf"
ff $(src 256x144 30) -c:v libvpx-vp9 "${vpx[@]}" -pix_fmt yuv420p10le -profile:v 2 \
    "$work/vp9_profile2.ivf"
svt=(-c:v libsvtav1 -preset 8 -crf 45 -svtav1-params keyint=30)
ff $(src 256x144 30) "${svt[@]}" -pix_fmt yuv420p "$work/av1_8bit.ivf"
ff $(src 256x144 30) -c:v libsvtav1 -preset 8 -crf 45 -pix_fmt yuv420p10le \
    -svtav1-params keyint=30:color-primaries=9:transfer-characteristics=16:matrix-coefficients=9 \
    "$work/av1_10bit.ivf"

for clip in h264_bframes.h264 h264_crop.h264 res_change.h264 hevc_main8.hevc \
    hevc_main10_hdr10.hevc hevc_444.hevc; do
    cp "$work/$clip" "$work/$clip.pts" "$out/"
done
cp "$work"/*.ivf "$out/"

FERRICAST_NVDEC_WRITE_REFERENCES=1 cargo test -p ferricast-nvcodec --test decode \
    -- --ignored --exact reference_hashes
ls -l "$out"
