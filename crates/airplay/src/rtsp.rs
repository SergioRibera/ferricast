use std::sync::{Arc, atomic::AtomicU64};

use ferricast_core::FerricastError;
use tokio::io::{AsyncWrite, AsyncWriteExt};

pub struct RtspManager(AtomicU64);

impl RtspManager {
    pub fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    pub fn builder(&self) -> RtspReqBuilder {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        RtspReqBuilder::new(self.0.load(std::sync::atomic::Ordering::SeqCst))
    }
}


pub struct RtspReqBuilder {
    cseq: u64,
    method: Method,
    path: String,
    content_type: String,
    body: Vec<u8>,
    headers: Vec<(String, String)>
}

impl RtspReqBuilder {
    fn new(cseq: u64) -> Self {
        Self {
            cseq,
            method: Method::POST,
            path: String::new(),
            content_type: String::new(),
            body: Vec::new(),
            headers: vec![
               ("User-Agent".to_string(), "AirPlay/381.13".to_string()),
               ("X-Apple-HKP".to_string(), "3".to_string()),
               ("X-Apple-Client-Name".to_string(), "Ferricast Airplay".to_string()) 
            ],
        } 
    } 
    pub fn path(mut self, path: String) -> Self {
        self.path = path;

        self
    }
    pub fn method(mut self, method: Method) -> Self {
        self.method = method;

        self
    }
    pub fn content_type(mut self, content_type: String) -> Self {
        self.content_type = content_type;

        self
    }
    pub fn body(mut self, body: Vec<u8>) -> Self {
        self.body = body;

        self
    }
    pub fn header(mut self, header: (String, String)) -> Self {
        self.headers.push(header);

        self
    }
    pub async fn write<T: AsyncWriteExt + Unpin>(self, writer: &mut T) -> Result<(), FerricastError> {
        let mut a = format!("{:?} {} RTSP/1.0\r\nCSeq: {}\r\n", self.method, self.path, self.cseq);

        if !self.content_type.is_empty() {
            a.push_str(format!("Content-Type: {}\r\n", self.content_type).as_str());
        }

        for header in self.headers {
            a.push_str(format!("{}: {}\r\n", header.0, header.1).as_str());
        }

        a.push_str("\r\n");

        writer.write(a.as_bytes()).await?;

        if !self.body.is_empty() {
            writer.write(&self.body).await?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub enum Method {
    POST,
}
