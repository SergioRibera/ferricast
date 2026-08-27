use std::{collections::HashMap, sync::{Arc, atomic::AtomicU64}};

use ferricast_core::FerricastError;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

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
              // ("X-Apple-Client-Name".to_string(), "Ferricast Airplay".to_string()) 
            ],
        } 
    } 
    pub fn path(mut self, path: String) -> Self {
        self.path = path;

        self
    }
    pub fn post(mut self) -> Self {
        self.method = Method::POST;

        self
    }
    pub fn options(mut self) -> Self {
        self.method = Method::OPTIONS;

        self
    }
    pub fn content_type(mut self, content_type: String) -> Self {
        self.content_type = content_type;

        self
    }
    pub fn body(mut self, body: Vec<u8>) -> Self {
        self.headers.push(("Content-Length".to_string(), body.len().to_string()));


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
    OPTIONS,
}


#[derive(Debug, Clone)]
pub struct RtspResponse {
    status_line: StatusLine,
    headers: HashMap<String, String>,
    content: Option<Vec<u8>>
}

impl RtspResponse {
    pub async fn read<T: AsyncReadExt + Unpin>(buf: &mut BufReader<T>) -> Result<Self, FerricastError> {
        let mut status_line: Option<StatusLine> = None;
        let mut headers: HashMap<String, String> = HashMap::new();

        


        loop {
            let mut line = String::new();        
    
            match buf.read_line(&mut line).await {
                Ok(0) => break, 
                Ok(_) => {
                    if line == "\r\n" {
                        break;
                    }

                    if status_line.is_none() {
                        status_line = Some(StatusLine::read(&line)?);
                        continue;
                    }

                    if let Some((name, value)) = line.split_once(':') {
                        headers.insert(name.trim().to_string(), value.trim().to_string());
                    } else {
                       return Err(FerricastError::Rtsp("Invalid RTSP header format".to_string())); 
                    }

                },
                Err(_) => break,
            }
        }

        let content = { 
            if let Some(v) = headers.get("Content-Length") {
                let len = v.parse::<usize>()
                    .map_err(|_| FerricastError::Rtsp("Invalid Content-Length Header".to_string()))?;


                
                let mut content = vec![0_u8; len];

                buf.read(&mut content).await?;

                Some(content)
            } else {
                None
            } 
        };


        
        let status_line = status_line.ok_or(FerricastError::Rtsp("Invalid RTSP Response".to_string()))?;

        Ok(Self { status_line, headers, content })
    }

    pub fn is_ok(&self) -> Result<(), FerricastError> {
        if self.is_success() {
            return Ok(());
        }

        Err(FerricastError::Rtsp(format!("RTSP Response failed with code {}", self.status_line.status_code)))
    }

    pub fn is_success(&self) -> bool {
        (200..=299).contains(&self.status_line.status_code)
    }
}

#[derive(Debug, Clone)]
pub struct StatusLine {
    pub status_code: u16,
    pub description: String,
}

impl StatusLine {
    pub fn read(status_line: &str) -> Result<Self, FerricastError> {
        let mut splited = status_line.split(" ");

        let header = splited.next().unwrap_or_default();

        if header != "RTSP/1.0" {
            return Err(FerricastError::Rtsp("Invalid Rtsp version".to_string()));
        }

        let status_code = splited.next().unwrap_or_default().parse::<u16>()
            .map_err(|e| FerricastError::Rtsp(format!("Invalid status code, {e}")))?;

        let description = splited.next().unwrap_or_default().to_string();

        Ok(Self { status_code, description })
    }
}
