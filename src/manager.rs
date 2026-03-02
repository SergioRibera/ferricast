use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::{mpsc, Mutex, RwLock};
use uuid::Uuid;

use ferricast_core::{
    CaptureConfig, CaptureSource, CastSession, Device, Discovery, DiscoveryEvent, EncodedFrame,
    FerricastError, ProtocolHandler, Result, ScreenCapture, StreamConfig, VideoEncoder,
};

type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait ErasedDiscovery: Send + Sync {
    fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> BoxFut<'_, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
}

impl<T: Discovery> ErasedDiscovery for T {
    fn start(&mut self, tx: mpsc::Sender<DiscoveryEvent>) -> BoxFut<'_, Result<()>> {
        Box::pin(Discovery::start(self, tx))
    }
    fn stop(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(Discovery::stop(self))
    }
}

trait ErasedSession: Send + Sync {
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<()>>;
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>>;
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>>;
    fn stop(&mut self) -> BoxFut<'_, Result<()>>;
    fn is_alive(&self) -> bool;
}

impl<T: CastSession> ErasedSession for T {
    fn connect<'a>(&'a mut self, device: &'a Device) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::connect(self, device))
    }
    fn setup_stream<'a>(&'a mut self, config: &'a StreamConfig) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::setup_stream(self, config))
    }
    fn send_frame<'a>(&'a mut self, frame: &'a EncodedFrame) -> BoxFut<'a, Result<()>> {
        Box::pin(CastSession::send_frame(self, frame))
    }
    fn stop(&mut self) -> BoxFut<'_, Result<()>> {
        Box::pin(CastSession::stop(self))
    }
    fn is_alive(&self) -> bool {
        CastSession::is_alive(self)
    }
}

struct RegisteredProtocol {
    protocol: &'static str,
    create_discovery: Box<dyn Fn() -> Box<dyn ErasedDiscovery> + Send + Sync>,
    create_session: Box<dyn Fn() -> Result<Box<dyn ErasedSession>> + Send + Sync>,
}

struct ActiveStream {
    device: Device,
    cancel_tx: mpsc::Sender<()>,
}

#[derive(Debug, Clone)]
pub enum ManagerEvent {
    DeviceFound(Device),
    DeviceLost(Uuid),
    StreamStarted {
        device_id: Uuid,
        device_name: String,
    },
    StreamStopped {
        device_id: Uuid,
    },
    StreamError {
        device_id: Uuid,
        message: String,
    },
    DiscoveryError {
        protocol: &'static str,
        message: String,
    },
}

pub struct StreamManager {
    protocols: Vec<RegisteredProtocol>,
    devices: Arc<RwLock<HashMap<Uuid, Device>>>,
    active_streams: Arc<Mutex<HashMap<Uuid, ActiveStream>>>,
    discovery_handles: Vec<Box<dyn ErasedDiscovery>>,
    event_tx: mpsc::Sender<ManagerEvent>,
    event_rx: Option<mpsc::Receiver<ManagerEvent>>,
    running: bool,
}

impl Default for StreamManager {
    fn default() -> Self {
        let (event_tx, event_rx) = mpsc::channel(256);
        Self {
            protocols: Vec::new(),
            devices: Arc::new(RwLock::new(HashMap::new())),
            active_streams: Arc::new(Mutex::new(HashMap::new())),
            discovery_handles: Vec::new(),
            event_tx,
            event_rx: Some(event_rx),
            running: false,
        }
    }
}

impl StreamManager {
    pub fn register<H>(&mut self)
    where
        H: ProtocolHandler + Clone + Default + 'static,
    {
        let h_sess = H::default();
        let h_disc = h_sess.clone();

        self.protocols.push(RegisteredProtocol {
            protocol: H::PROTOCOL,
            create_discovery: Box::new(move || {
                Box::new(h_disc.create_discovery()) as Box<dyn ErasedDiscovery>
            }),
            create_session: Box::new(move || {
                let session = h_sess.create_session()?;
                Ok(Box::new(session) as Box<dyn ErasedSession>)
            }),
        });
    }

    pub fn take_event_rx(&mut self) -> Option<mpsc::Receiver<ManagerEvent>> {
        self.event_rx.take()
    }

    pub fn registered_protocols(&self) -> Vec<&'static str> {
        self.protocols.iter().map(|p| p.protocol).collect()
    }

    pub async fn start_discovery(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }

        let (disc_tx, mut disc_rx) = mpsc::channel::<DiscoveryEvent>(256);

        for proto in &self.protocols {
            let mut discovery = (proto.create_discovery)();
            let tx = disc_tx.clone();
            match discovery.start(tx).await {
                Ok(()) => {
                    self.discovery_handles.push(discovery);
                }
                Err(e) => {
                    tracing::warn!(
                        protocol = proto.protocol,
                        %e,
                        "Failed to start discovery, skipping"
                    );
                    let _ = self
                        .event_tx
                        .send(ManagerEvent::DiscoveryError {
                            protocol: proto.protocol,
                            message: e.to_string(),
                        })
                        .await;
                }
            }
        }

        let devices = self.devices.clone();
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            while let Some(event) = disc_rx.recv().await {
                match event {
                    DiscoveryEvent::DeviceFound(device) => {
                        let id = device.id;
                        tracing::info!(name = %device.name, protocol = device.protocol, "Device found");
                        devices.write().await.insert(id, device.clone());
                        let _ = event_tx.send(ManagerEvent::DeviceFound(device)).await;
                    }
                    DiscoveryEvent::DeviceLost(id) => {
                        tracing::info!(?id, "Device lost");
                        devices.write().await.remove(&id);
                        let _ = event_tx.send(ManagerEvent::DeviceLost(id)).await;
                    }
                    DiscoveryEvent::Error { protocol, message } => {
                        tracing::warn!(protocol, %message, "Discovery error");
                        let _ = event_tx
                            .send(ManagerEvent::DiscoveryError { protocol, message })
                            .await;
                    }
                }
            }
        });

        self.running = true;
        Ok(())
    }

    pub async fn stop_discovery(&mut self) -> Result<()> {
        for discovery in &mut self.discovery_handles {
            discovery.stop().await?;
        }
        self.discovery_handles.clear();
        self.running = false;
        Ok(())
    }

    pub async fn devices(&self) -> Vec<Device> {
        self.devices.read().await.values().cloned().collect()
    }

    pub async fn start_stream(
        &self,
        device_id: Uuid,
        source: CaptureSource,
        mut capture: impl ScreenCapture + 'static,
        mut encoder: impl VideoEncoder + 'static,
        config: StreamConfig,
    ) -> Result<()> {
        let device = {
            let devices = self.devices.read().await;
            devices
                .get(&device_id)
                .cloned()
                .ok_or_else(|| FerricastError::DeviceNotFound(device_id.to_string()))?
        };

        let proto = self
            .protocols
            .iter()
            .find(|p| p.protocol == device.protocol)
            .ok_or_else(|| {
                FerricastError::Protocol(format!("No handler for {}", device.protocol))
            })?;

        // Start capture first so the portal picker shows before connecting.
        let capture_config = CaptureConfig {
            fps: config.fps,
            width: Some(config.width),
            height: Some(config.height),
            show_cursor: true,
        };

        capture.start(source, capture_config).await?;
        tracing::info!(device_id = %device_id, "Capture started, connecting to device");

        let mut session = (proto.create_session)()?;
        session.connect(&device).await?;
        session.setup_stream(&config).await?;

        let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);

        let active_streams = self.active_streams.clone();
        let event_tx = self.event_tx.clone();
        let device_name = device.name.clone();
        let did = device.id;

        tokio::spawn(async move {

            let _ = event_tx
                .send(ManagerEvent::StreamStarted {
                    device_id: did,
                    device_name,
                })
                .await;

            loop {
                tokio::select! {
                    _ = cancel_rx.recv() => {
                        tracing::info!(?did, "Stream cancelled");
                        break;
                    }
                    frame_result = capture.next_frame() => {
                        match frame_result {
                            Ok(raw_frame) => {
                                match encoder.encode(&raw_frame) {
                                    Ok(encoded) => {
                                        if let Err(e) = session.send_frame(&encoded).await {
                                            tracing::error!(%e, "Failed to send frame");
                                            let _ = event_tx.send(ManagerEvent::StreamError {
                                                device_id: did,
                                                message: e.to_string(),
                                            }).await;
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(%e, "Encoding error");
                                        let _ = event_tx.send(ManagerEvent::StreamError {
                                            device_id: did,
                                            message: e.to_string(),
                                        }).await;
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!(%e, "Capture error");
                                let _ = event_tx.send(ManagerEvent::StreamError {
                                    device_id: did,
                                    message: e.to_string(),
                                }).await;
                                break;
                            }
                        }
                    }
                }
            }

            let _ = session.stop().await;
            let _ = capture.stop().await;
            active_streams.lock().await.remove(&did);
            let _ = event_tx
                .send(ManagerEvent::StreamStopped { device_id: did })
                .await;
        });

        self.active_streams.lock().await.insert(
            device_id,
            ActiveStream {
                device,
                cancel_tx,
            },
        );

        Ok(())
    }

    pub async fn stop_stream(&self, device_id: Uuid) -> Result<()> {
        let stream = self.active_streams.lock().await.remove(&device_id);
        if let Some(stream) = stream {
            let _ = stream.cancel_tx.send(()).await;
            tracing::info!(name = %stream.device.name, "Stopping stream");
            Ok(())
        } else {
            Err(FerricastError::NoActiveSession)
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        let ids: Vec<Uuid> = self.active_streams.lock().await.keys().copied().collect();
        for id in ids {
            let _ = self.stop_stream(id).await;
        }
        self.stop_discovery().await
    }
}
