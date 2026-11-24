//! Decode throughput of one Annex B H.264 / HEVC file, in host and device
//! mode:
//!
//! ```text
//! cargo run --release -p ferricast-nvcodec --example nvdec_throughput -- clip.h264 [rounds]
//! ```
//!
//! The file is fed in 64 KiB chunks (no access unit splitting needed), so the
//! decoder runs without the one-picture-per-packet promise.
#![allow(clippy::print_stdout, reason = "an example reporting its measurements")]

use std::time::Instant;

use ferricast_nvcodec::cuda::{Context, Device as CudaDevice};
use ferricast_nvcodec::decode::{Codec, Decoder, DecoderConfig, Device, Host, Packet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: nvdec_throughput <file.h264|file.hevc> [rounds]")?;
    let rounds: u32 = args.next().map_or(Ok(3), |rounds| rounds.parse())?;
    let extension = std::path::Path::new(&path).extension();
    let codec = if extension.is_some_and(|extension| {
        extension.eq_ignore_ascii_case("hevc") || extension.eq_ignore_ascii_case("h265")
    }) {
        Codec::Hevc
    } else {
        Codec::H264
    };
    let data = std::fs::read(&path)?;
    let context = Context::primary(CudaDevice::get(0)?)?;
    let config = DecoderConfig::new(codec).with_whole_pictures(false);
    for round in 0..rounds {
        let mut host = Decoder::<Host>::new(&context, config)?;
        let start = Instant::now();
        let mut frames = 0_u32;
        let mut checksum = 0_u64;
        for chunk in data.chunks(64 << 10) {
            host.decode(Packet::untimed(chunk))?;
            while let Some(frame) = host.next_frame() {
                frames += 1;
                checksum = checksum.wrapping_add(u64::from(frame.data()[0]));
            }
        }
        host.flush()?;
        while let Some(frame) = host.next_frame() {
            frames += 1;
            checksum = checksum.wrapping_add(u64::from(frame.data()[0]));
        }
        let host_fps = f64::from(frames) / start.elapsed().as_secs_f64();

        let mut device = Decoder::<Device>::new(&context, config)?;
        let start = Instant::now();
        let mut device_frames = 0_u32;
        let mut last = None;
        for chunk in data.chunks(64 << 10) {
            device.decode(Packet::untimed(chunk))?;
            while let Some(frame) = device.next_frame() {
                device_frames += 1;
                last = Some(frame);
            }
        }
        device.flush()?;
        while let Some(frame) = device.next_frame() {
            device_frames += 1;
            last = Some(frame);
        }
        if let Some(frame) = &last {
            frame.synchronize()?;
        }
        let device_fps = f64::from(device_frames) / start.elapsed().as_secs_f64();
        let size = last
            .as_ref()
            .map(|frame| (frame.info().width, frame.info().height));
        println!(
            "round {round}: {frames} frames {size:?}: host {host_fps:.0} fps, device {device_fps:.0} fps (checksum {checksum})"
        );
    }
    Ok(())
}
