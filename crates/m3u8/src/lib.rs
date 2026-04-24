use ferricast_core::FerricastError;
use thiserror::Error;

const HEADER: &[u8] = b"#EXTM3U\n";

#[repr(u8)]
pub enum M3u8Version {
    V3 = 3,
}

pub struct M3u8Writer {
    target_duration: u8,
    version: u8,
    segments: Vec<(u8, String)>,
    media_seq: Option<u64>,
}

impl Default for M3u8Writer {
    fn default() -> Self {
        Self {
            target_duration: 10,
            version: M3u8Version::V3 as u8,
            segments: Vec::new(),
            media_seq: None,
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

    pub fn add_segment(mut self, duration: u8, url: String) -> Result<Self, FerricastError> { 
        if duration == self.target_duration {
            return Err(FerricastError::M3u8("Invalid segment duration".to_string()));
        }

        self.segments.push((duration, url));        
        
        Ok(self)
    } 

    pub fn set_media_seq(mut self, media_seq: u64) -> Self {
        self.media_seq = Some(media_seq);
        self
    }

    pub fn to_string(&self) -> Result<String, FerricastError> {
        let mut v = Vec::new();
        self.write(&mut v)?;
        

        Ok(String::from_utf8(v).map_err(|_| FerricastError::M3u8("Invalid m3u8".to_string()))?)
    }

    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> Result<(), FerricastError> {
        writer.write_all(HEADER)?;
        writer.write_all(format!("#EXT-X-TARGETDURATION:{}\n", self.target_duration).as_bytes())?;
        writer.write_all(format!("#EXT-X-VERSION:{}\n", self.version).as_bytes())?;

        if let Some(seq) = self.media_seq  {
            writer.write_all(format!("#EXT-X-MEDIA-SEQUENCE:{}\n", seq).as_bytes())?;
        }

        for segment in &self.segments {
            writer.write_all(format!("#EXTINF:{},\n{}\n", segment.0, segment.1).as_bytes())?;
        }
        
        Ok(())
    }
}

