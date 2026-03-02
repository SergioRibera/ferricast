use ferricast_core::{Device, EncodedFrame, FerricastError, Result, StreamConfig};

const DEFAULT_APP_NAME: &str = "Ferricast";

pub struct DialSession {
    app_name: String,
    http_client: reqwest::Client,
    alive: bool,
}

impl Default for DialSession {
    fn default() -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .expect("failed to build reqwest client");

        Self {
            app_name: DEFAULT_APP_NAME.to_owned(),
            http_client,
            alive: false,
        }
    }
}

impl ferricast_core::CastSession for DialSession {
    async fn connect(&mut self, device: &Device) -> Result<()> {
        if device.protocol != "dial" {
            return Err(FerricastError::Protocol(format!(
                "DialSession cannot connect to a {:?} device",
                device.protocol
            )));
        }

        Ok(())
    }

    async fn setup_stream(&mut self, config: &StreamConfig) -> Result<()> {
        Ok(())
    }

    async fn send_frame(&mut self, frame: &EncodedFrame) -> Result<()> {
        if !self.alive {
            return Err(FerricastError::NoActiveSession);
        }
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn is_alive(&self) -> bool {
        self.alive
    }
}
