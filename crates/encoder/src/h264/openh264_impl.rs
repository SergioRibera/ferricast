use bytes::Bytes;
use ferricast_core::{Codec, EncodedFrame, FerricastError, H264Profile, PixelFormat, VideoEncoder};
use openh264::{OpenH264API, encoder::{BitRate, Encoder, EncoderConfig, FrameType, Profile}, formats::{BgraSliceU8, RgbaSliceU8, YUVBuffer}};

pub const OPEN_H264_THREADS_VAR: &'static str = "FERRICAST_OPEN_H264_THREADS";


#[derive(Default)]
pub struct OpenH264Encoder {
    pub encoder: Option<Encoder>,
    pub frame_count: usize,
    pub fps: usize,
    pub sps_pps: Vec<u8>
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
             .skip_frames(false)
             .usage_type(openh264::encoder::UsageType::CameraVideoRealTime)
             
             .bitrate(BitRate::from_bps(config.bitrate_kbps * 1000));
        let mut encoder = Encoder::with_api_config(api, encoder_config).map_err(|e| FerricastError::Encoder(format!("Cannot create openh264 encoder {:?}", e)))?;
        
        let empty_frame = YUVBuffer::new(config.width as usize, config.height as usize);
        let encoded = encoder.encode(&empty_frame).map_err(|_| FerricastError::Encoding("Cannot encode empty frame".to_string()))?;
        let data = encoded.to_vec();

        let sps_pps = extract_sps_pps(&data);
        if sps_pps.is_empty() {
            return Err(FerricastError::Encoding("Sps/pps not found in first keyframe".to_string()));
        }


        tracing::info!("SPS/PPS found!");
        self.sps_pps = sps_pps;


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

        let pts = (self.frame_count * 1000) / self.fps;
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
        if self.sps_pps.is_empty() {
            tracing::warn!("No keyframe has been generated to obtain the sps/pps, resulting in an empty Vec");
        }
        Ok(self.sps_pps.clone()) 
    }
    fn request_keyframe(&mut self) {
        let encoder = self.encoder.as_mut().expect("Ferricast(Openh264) bug: use of an encoder that has not been configured");
    
        encoder.force_intra_frame();
    }
}

pub fn extract_sps_pps(annex_b: &[u8]) -> Vec<u8> {
    let positions = find_start_codes(annex_b);
    if positions.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    for (i, &(start, sc_len)) in positions.iter().enumerate() {
        let nal_start = start + sc_len;
        if nal_start >= annex_b.len() {
            continue;
        }
        let nal_type = annex_b[nal_start] & 0x1f;
        if nal_type != 7 && nal_type != 8 {
            continue;
        }
        let nal_end = positions
            .get(i + 1)
            .map(|(s, _)| *s)
            .unwrap_or(annex_b.len());
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&annex_b[nal_start..nal_end]);
    }
    out
}

fn find_start_codes(buf: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= buf.len() {
        if buf[i] == 0 && buf[i + 1] == 0 {
            if buf[i + 2] == 1 {
                out.push((i, 3));
                i += 3;
                continue;
            }
            if i + 4 <= buf.len() && buf[i + 2] == 0 && buf[i + 3] == 1 {
                out.push((i, 4));
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    out
}


