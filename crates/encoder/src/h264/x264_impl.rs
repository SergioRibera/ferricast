use bytes::Bytes;
use ferricast_core::{
    Codec, EncodedFrame, FerricastError, PixelFormat, Result, VideoEncoder,
};
use x264::{Colorspace, Encoder, Image, Preset, Setup, Tune};

#[derive(Default)]
pub struct X264H264Encoder {
    encoder: Option<Encoder>,
    colorspace: Option<Colorspace>,
    frame_index: i64,
    fps: i64,
}


unsafe impl Send for X264H264Encoder {}
unsafe impl Sync for X264H264Encoder {}

impl VideoEncoder for X264H264Encoder {
    const CODEC: Codec = Codec::H264;

    fn configure(&mut self, config: &ferricast_core::EncoderConfig) -> ferricast_core::Result<()> {
        let colorspace = match config.pixel_format {
            PixelFormat::Bgra => Colorspace::BGRA,
            PixelFormat::Nv12 => Colorspace::NV12,
            PixelFormat::Rgba => {
                // This will probably fail
                Colorspace::RGB
            },
            PixelFormat::I420 => Colorspace::I420,
        };

        // `ultrafast` + `zerolatency` is what every live encoder under
        // the sun uses: drops B-frames, disables lookahead, sizes the
        // VBV for one-frame buffering. Without it x264 defaults to
        // `medium` with a multi-frame lookahead — far too slow for
        // real-time screen casting.
        //
        // `scenecut_threshold(0)` pins keyframes to exactly every
        // `max_keyframe_interval` frames so HLS segment cadence is
        // predictable. With scenecuts on, an idle desktop yields ≥4 s
        // segments and a busy one yields fragmentary < 1 s ones.
        let keyint = config.keyframe_interval.max(1) as i32;
        let encoder = Setup::preset(Preset::Ultrafast, Tune::None, false, true)
            .fps(config.fps, 1)
            .baseline()
            .bitrate(config.bitrate_kbps as i32)
            .max_keyframe_interval(keyint)
            .min_keyframe_interval(keyint)
            .scenecut_threshold(0)
            .build(colorspace, config.width as i32, config.height as i32)
            .map_err(|_| FerricastError::Encoder("Cannot create encoder".to_string()))?;

        self.encoder = Some(encoder);
        self.colorspace = Some(colorspace);
        self.fps = config.fps as i64;

        Ok(())
    }
    fn get_headers(&mut self) -> Result<Vec<u8>> {
        Ok(self.encoder.as_mut().unwrap().headers().map_err(|_| FerricastError::Encoder("Cannot get headers".to_string()))?.entirety().to_vec())
    }
    fn encode(&mut self, frame: ferricast_core::CapturedFrame) -> ferricast_core::Result<ferricast_core::EncodedFrame> {
        // x264 only consumes CPU bytes — if the source produced a
        // GpuFrame, force a readback through its attached importer.
        // For VA-API or any future GPU-direct encoder we'd match on
        // the variant instead of always materialising a `RawFrame`.
        let frame = frame.into_cpu()?;
        let image = match self.colorspace.unwrap() {
            Colorspace::BGRA => Image::bgra(frame.width as i32, frame.height as i32, &frame.data),
            Colorspace::NV12 => return Err(FerricastError::Encoder("Unimplemented colorspace: nv12".to_string())),
            Colorspace::RGB => Image::rgb(frame.width as i32, frame.height as i32, &frame.data),
            Colorspace::I420 => return Err(FerricastError::Encoder("Unimplemented colorspace: i420".to_string())),
            _ => unimplemented!(),
        };

        let encoder = self.encoder.as_mut().unwrap();
       
        let (data, p) = encoder.encode(self.frame_index, image).map_err(|_| FerricastError::Encoder("Cannot encode frame".to_string()))?;
        self.frame_index += 1;

        let data = data.entirety().to_vec();


        Ok(EncodedFrame {
            codec: Codec::H264,
            data: Bytes::from({
                if p.keyframe() {
                    //let mut d = Vec::new();
                    //d.extend(self.get_headers()?);
                    //d.extend(data);

                    data
                } else {
                    data
                }
            }),
            timestamp_us: frame.timestamp_us,
            duration_us: None,
            is_keyframe: p.keyframe(),
            pts_dts: (p.pts() as u64, p.dts() as u64),
        })
    }
    fn flush(self) -> ferricast_core::Result<Vec<ferricast_core::EncodedFrame>> {
        let encoder = self.encoder.unwrap();
        let mut frames = Vec::new();
        let mut flush = encoder.flush();

        while let Some(Ok((data, p))) = flush.next() {
            frames.push(EncodedFrame {
                codec: Codec::H264,
                data: Bytes::from(data.entirety().to_vec()),
                timestamp_us: 0,
                is_keyframe: p.keyframe(),
                duration_us: None,
                pts_dts: (p.pts() as u64, p.dts() as u64),
            });
        }
        
        Ok(frames)
    }
} 
