use ferricast_muxer::Muxer;
use ferricast_muxer::mpeg_ts::MpegTs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use tokio::net::ToSocketAddrs;

use ferricast_core::{FerricastError, ScreenCapture, VideoEncoder};
use tokio::net::TcpListener;

pub struct HlsServer<S: ScreenCapture, E: VideoEncoder> {
    listener: TcpListener,
    encoder: E,
    capture: S,
    muxer: MpegTs
}

impl<S: ScreenCapture, E: VideoEncoder> HlsServer<S, E> {
    pub async fn listen<A: ToSocketAddrs>(addr: A, mut encoder: E, mut capture: S) -> Result<Self, FerricastError> {
        let listener = TcpListener::bind(addr).await?;
        let mut muxer = MpegTs::default();
        muxer.config(encoder.get_headers()?)?;

        Ok(Self {
            listener,
            encoder,
            capture,
            muxer,
        })
    }
    pub async fn serve(&mut self) -> Result<(), FerricastError> {
        let (mut socket, _) = self.listener.accept().await?;


        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut req = httparse::Request::new(&mut headers);
        let mut req_text = String::new();

        {
            let mut buf = BufReader::new(&mut socket);
            let mut lines = buf.lines();

             while let Some(line) = lines.next_line().await? {
                if line.is_empty() {
                    req_text.push_str("\r\n");
                    break;
                }


                req_text.push_str(&line);
                req_text.push_str("\r\n");

             }
                    
        }

       // socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/x-mpegurl\r\nContent-Length: {}\r\n\r\n{}", em.len(), em).as_bytes()).await?;



         
        Ok(())
    }
}
