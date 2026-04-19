use ferricast_core::{EncodedFrame, FerricastError};

pub trait Muxer {
    fn config(&mut self, sps_pps: Vec<u8>) -> Result<(), FerricastError>;
    fn add_frame(&mut self, frame: EncodedFrame) -> Result<(), FerricastError>;
    fn drain(&mut self) -> Vec<u8>;
    fn add_audio(&mut self) -> Result<(), FerricastError> {
        unimplemented!()
    }
}

pub mod mpeg_ts;
