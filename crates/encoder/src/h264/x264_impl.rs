use bytes::Bytes;
use ferricast_core::{Codec, EncodedFrame, FerricastError, H264Profile, VideoEncoder};
use x264::{Colorspace, Encoder, Image};


#[derive(Default)]
pub struct X264Encoder {
    pub encoder: Option<Encoder>,
    pub frame_count: usize,
    pub fps: usize,
    pub sps_pps: Vec<u8>
}

unsafe impl Sync for X264Encoder {}
unsafe impl Send for X264Encoder {}

impl VideoEncoder for X264Encoder
{
    const CODEC: Codec = Codec::H264;
    
    fn configure(&mut self, config: &ferricast_core::EncoderConfig) -> ferricast_core::Result<()> {
        let builder = Encoder::builder();
        let builder = match config.max_h264_profile.unwrap_or(H264Profile::High) {
            H264Profile::Baseline => builder.baseline(),
            H264Profile::Main => builder.main(),
            H264Profile::High => builder.high(),
        };

        let mut encoder = builder.fps(config.fps, 1).build(Colorspace::BGRA, config.width as _, config.height as _).map_err(|_| FerricastError::Encoding("Cannot create encoder".to_string()))?;

    

        let header = encoder.headers().map_err(|_| FerricastError::Encoding("Cannot get sps/pps".to_string()))?.entirety().to_vec(); 
        
        self.sps_pps = header;
        self.fps = config.fps as usize;
        self.encoder = Some(encoder);

        Ok(())
    }
    fn encode(&mut self, frame: ferricast_core::CapturedFrame) -> ferricast_core::Result<ferricast_core::EncodedFrame> {
        let timestamp_us = frame.timestamp_us();
        let frame = frame.into_cpu()?;
        let encoder = self.encoder.as_mut().expect("Ferricast(X264) bug: use of an encoder that has not been configured");
    
        let image = Image::bgra(frame.width as i32, frame.height as i32, &frame.data);

        let pts = (self.frame_count * 1000) / self.fps;
        self.frame_count += 1;

        let (data, plane) = encoder.encode(pts as i64, image).map_err(|_| FerricastError::Encoding("Cannot encode frame".to_string()))?;

        let content = data.entirety();
  
        let mut final_payload = Vec::new();

        if plane.keyframe() {
            final_payload.extend_from_slice(&self.sps_pps);
        }

        final_payload.extend(content);




        Ok(EncodedFrame {  
            codec: Codec::H264,
            data: Bytes::from(final_payload),
            timestamp_us,
            is_keyframe: plane.keyframe(),
            duration_us: Some(1_000_000 / self.fps as u64),
            pts_dts: (plane.pts() as u64, plane.dts() as u64)
        })
        
    }
    fn flush(self) -> ferricast_core::Result<Vec<ferricast_core::EncodedFrame>> {
        Ok(vec![])
    }
    fn get_headers(&mut self) -> ferricast_core::Result<Vec<u8>> {
        Ok(self.sps_pps.clone())
    }
    fn request_keyframe(&mut self) {
        tracing::warn!("X264 backend can't request keyframe");
    }
}

