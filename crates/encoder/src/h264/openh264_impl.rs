use bytes::Bytes;
use ferricast_core::{Codec, EncodedFrame, FerricastError, H264Profile, PixelFormat, VideoEncoder};
use openh264::{OpenH264API, encoder::{BitRate, Encoder, EncoderConfig, FrameType, Profile}, formats::{BgraSliceU8, RgbaSliceU8, YUVBuffer}};

pub const OPEN_H264_THREADS_VAR: &'static str = "FERRICAST_OPEN_H264_THREADS";


#[derive(Default)]
pub struct OpenH264Encoder {
    pub encoder: Option<Encoder>,
    pub frame_count: usize,
    pub fps: usize,
}


impl VideoEncoder for OpenH264Encoder {
    const CODEC: Codec = Codec::H264;
    

    fn configure(&mut self, config: &ferricast_core::EncoderConfig) -> ferricast_core::Result<()> {
         let api = OpenH264API::from_source();
          
         let profile = match config.max_h264_profile.unwrap_or(ferricast_core::H264Profile::Baseline) {
            H264Profile::Baseline => Profile::Baseline,
            H264Profile::Main => Profile::Main,
            H264Profile::High => Profile::High,
         };
        
         let encoder_config = EncoderConfig::new()
             .profile(profile)
             .num_threads(1)
             .bitrate(BitRate::from_bps(config.bitrate_kbps * 1000));
        let encoder = Encoder::with_api_config(api, encoder_config).map_err(|e| FerricastError::Encoder(format!("Cannot create openh264 encoder {:?}", e)))?;        

        self.fps = config.fps as usize;
        self.encoder = Some(encoder);

        Ok(())
    }
    fn encode(&mut self, frame: ferricast_core::CapturedFrame) -> ferricast_core::Result<ferricast_core::EncodedFrame> {
        let frame = frame.into_cpu().unwrap();
        let encoder = self.encoder.as_mut().expect("Ferricast(Openh264) bug: use of an encoder that has not been configured");
        
        let yuv_buffer = match frame.format {
            PixelFormat::Bgra => {
                let bgra = BgraSliceU8::new(&frame.data, (frame.width as usize, frame.height as usize));

                YUVBuffer::from_rgb_source(bgra)
            },
            PixelFormat::Rgba => {
                let rgba = RgbaSliceU8::new(&frame.data, (frame.width as usize, frame.height as usize));

                YUVBuffer::from_rgb_source(rgba)
            },
            PixelFormat::Nv12 => {
                YUVBuffer::from_vec(frame.data.to_vec(), frame.width as usize, frame.height as usize)
            },
            _ => unimplemented!(),
        };
  
        let encoded = encoder.encode(&yuv_buffer).map_err(|_| FerricastError::Encoding("Cannot encode frame".to_string()))?;
        let data = encoded.to_vec();

        let pts = (self.frame_count * self.fps) / self.fps;
        self.frame_count += 1;


        Ok(EncodedFrame {  
            codec: Codec::H264,
            data: Bytes::from(data),
            timestamp_us: 0,
            is_keyframe: encoded.frame_type() == FrameType::IDR,
            duration_us: None, 
            pts_dts: (pts as u64, pts as u64)
        })
        
    }
    fn flush(self) -> ferricast_core::Result<Vec<ferricast_core::EncodedFrame>> {
        Ok(vec![])
    }
    fn get_headers(&mut self) -> ferricast_core::Result<Vec<u8>> {
        // TODO!
        Ok(vec![]) 
    }
    fn request_keyframe(&mut self) {
        let encoder = self.encoder.as_mut().expect("Ferricast(Openh264) bug: use of an encoder that has not been configured");
    
        encoder.force_intra_frame();
    }
}
