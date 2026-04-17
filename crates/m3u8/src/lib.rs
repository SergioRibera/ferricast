use std::io::Write;

use thiserror::Error;

const HEADER: &[u8] = b"#EXTM3U\n";

#[repr(u8)]
pub enum M3u8Version {
    V3 = 3,
}

pub struct M3u8Writer {
    target_duration: u8,
    version: u8,
    segments: Vec<(u8, String)>
}

impl Default for M3u8Writer {
    fn default() -> Self {
        Self {
            target_duration: 10,
            version: M3u8Version::V3 as u8,
            segments: Vec::new(),
        }
    }
}

impl M3u8Writer {
    pub fn set_target_duration(mut self, duration: u8) -> Self {
        self.target_duration = duration;

        self
    }

    pub fn set_version(mut self, version: M3u8Version) -> Self {
        self.version = version as u8;

        self
    }

    pub fn add_segment(mut self, duration: u8, url: String) -> Result<Self, M3u8Error> { 
        if duration == self.target_duration {
            return Err(M3u8Error::InvalidSegment);
        }

        self.segments.push((duration, url));        
        
        Ok(self)
    } 


    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), M3u8Error> {
        writer.write_all(HEADER)?;
        writer.write_all(format!("#EXT-X-TARGETDURATION:{}\n", self.target_duration).as_bytes())?;
        writer.write_all(format!("#EXT-X-VERSION:{}\n", self.version).as_bytes())?;

        for segment in &self.segments {
            writer.write_all(format!("#EXTINF:{},\n{}\n", segment.0, segment.1).as_bytes())?;
        }
        
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum M3u8Error {
    #[error("IO Error {0}")]
    IoError(#[from] std::io::Error), 

    #[error("Invalid Segment")]
    InvalidSegment,
}
