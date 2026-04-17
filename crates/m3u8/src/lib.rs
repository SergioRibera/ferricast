use std::io::Write;

use thiserror::Error;

const HEADER: &[u8] = b"#EXTM3U";

pub struct M3u8Writer;


impl M3u8Writer {
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), M3u8Error> {
        writer.write_all(HEADER)?;
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum M3u8Error {
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error), 
}
