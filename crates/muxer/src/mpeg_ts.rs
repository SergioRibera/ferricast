use libmpegts::mux::{Multiplexer, MuxFrame, MuxService, MuxStream};

use crate::Muxer;

const H264: u8 = 0x1b;
const AAC: u8 = 0x0f;

pub struct MpegTs {
    mux: Option<Multiplexer>,
    video_index: usize,
}

impl Default for MpegTs {
    fn default() -> Self {
        Self {
            mux: None,
            video_index: 0,
        }
    }
}

impl Muxer for MpegTs {
    fn config(&mut self, sps_pps: Vec<u8>) -> Result<(), ferricast_core::FerricastError> {
        let mut mux = Multiplexer::new(1);
    
        mux.add_service(&MuxService {
            program_number: 1,
            pmt_pid: 256,
            pcr_pid: 101,
            program_descriptors: Vec::new(),
            service_descriptors: Vec::new(),
            streams: vec![
                MuxStream {
                    stream_type: H264,
                    elementary_pid: 101,
                    stream_descriptors: Vec::new(),
                }, 
                MuxStream {
                    stream_type: AAC,
                    elementary_pid: 102,
                    stream_descriptors: Vec::new(),
                }
            ],
        });



        let index =  mux.stream_index(101).unwrap();
        mux.push_frame(index, MuxFrame {
            data: sps_pps,
            is_key_frame: true,
            pts_dts: None,
        });

        // SAFE UNWRAP!
        self.video_index = index;
        self.mux = Some(mux);


        Ok(()) 
    }
    fn add_frame(&mut self, frame: ferricast_core::EncodedFrame) -> Result<(), ferricast_core::FerricastError> {
        let mux = self.mux.as_mut().unwrap();

        mux.push_frame(self.video_index, MuxFrame {
            data: frame.data.to_vec(),
            is_key_frame: frame.is_keyframe,
            pts_dts: Some((frame.pts_dts.0, Some(frame.pts_dts.1)).into()),
        });

        Ok(())
    }
    fn drain(&mut self) -> Vec<u8> {
        let mut v = Vec::new();
        let mux = self.mux.as_mut().unwrap();

        while mux.drain(&mut v) != 0 {}

        v
    }
}
