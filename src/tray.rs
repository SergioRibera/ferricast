use std::sync::{Arc, mpsc as std_mpsc};

use traysys::{MenuItemBuilder, TrayIcon, TrayMenu, TrayMenuBuilder};
use uuid::Uuid;

use crate::manager::ManagerEvent;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Hash, PartialEq)]
pub enum TrayAction {
    Device(Uuid),
    NoDevices,
    About,
    Quit,
}

pub struct FerricastTray {
    tray: Arc<TrayMenu<TrayAction>>,
    action_tx: std_mpsc::Sender<TrayAction>,
    action_rx: std_mpsc::Receiver<TrayAction>,
}

impl FerricastTray {
    pub fn new() -> traysys::Result<Self>
    where
        Self: Sized,
    {
        let (action_tx, action_rx) = std_mpsc::channel();

        let builder = TrayMenuBuilder::default()
            .icon(create_default_icon())
            .name(format!("Ferricast v{VERSION}"))
            .add_item(
                TrayAction::About,
                ("About", false, {
                    let tx = action_tx.clone();
                    move || {
                        let _ = tx.send(TrayAction::About);
                    }
                }),
            )
            .add_item(
                TrayAction::Quit,
                ("Quit", false, {
                    let tx = action_tx.clone();
                    move || {
                        let _ = tx.send(TrayAction::Quit);
                    }
                }),
            )
            .add_separator()
            .add_label(TrayAction::NoDevices, "Searching for devices...", None);

        let tray = Arc::new(builder.build());
        Ok(Self {
            tray,
            action_tx,
            action_rx,
        })
    }

    pub fn run(&self) {
        self.tray.start();
    }

    pub fn close(&self) {
        self.tray.close();
    }

    pub fn try_recv_action(&self) -> Option<TrayAction> {
        self.action_rx.try_recv().ok()
    }

    pub fn handle_manager_event(&self, event: &ManagerEvent) {
        match event {
            ManagerEvent::DeviceFound(device) => {
                let label = device.name.clone();
                let device_id = device.id;
                let tx = self.action_tx.clone();
                let icon = protocol_icon(device.protocol);

                self.tray.remove(TrayAction::NoDevices);

                if !self.tray.contains(&TrayAction::Device(device_id)) {
                    self.tray.update(move |menu| {
                        menu.add_item(
                            TrayAction::Device(device_id),
                            MenuItemBuilder::default()
                                .enabled(true)
                                .label(label.clone())
                                .icon(icon.clone())
                                .action({
                                    let tx = tx.clone();
                                    move || {
                                        let _ = tx.send(TrayAction::Device(device_id));
                                    }
                                }),
                        )
                    });
                }
            }
            ManagerEvent::DeviceLost(device_id) => {
                self.tray.remove(TrayAction::Device(*device_id));
            }
            ManagerEvent::StreamStarted { device_name, .. } => {
                tracing::info!(%device_name, "Stream started");
            }
            ManagerEvent::StreamStopped { device_id } => {
                tracing::info!(?device_id, "Stream stopped");
            }
            _ => {}
        }
    }
}

fn create_default_icon() -> TrayIcon {
    let size = 16u32;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let in_circle = ((x as f32 - 7.5).powi(2) + (y as f32 - 7.5).powi(2)).sqrt() < 7.0;
            if in_circle {
                rgba.extend_from_slice(&[0xF7, 0x6D, 0x27, 0xFF]);
            } else {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
    }
    TrayIcon::from_static(size, size, rgba)
}

fn protocol_icon(protocol: &str) -> TrayIcon {
    let svg = match protocol {
        "chromecast" => include_str!("../assets/chromecast.svg"),
        "airplay" => include_str!("../assets/airplay.svg"),
        "dial" => include_str!("../assets/dial.svg"),
        "miracast" => include_str!("../assets/miracast.svg"),
        _ => include_str!("../assets/chromecast.svg"),
    };
    TrayIcon::from_svg(svg, 16).unwrap_or_else(|_| create_default_icon())
}
