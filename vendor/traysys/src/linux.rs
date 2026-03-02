use crate::{TrayIcon, ID};

use super::MenuEntry;
use ksni::menu::{CheckmarkItem, StandardItem};
use ksni::Handle;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Tray<I: ID>(TrayData<I>, Arc<Mutex<Option<Handle<TrayData<I>>>>>);

#[derive(Clone)]
struct TrayData<I: ID> {
    name: Arc<str>,
    items: Arc<Mutex<Vec<MenuEntry<I>>>>,
    icon: Option<TrayIcon>,
}

impl<I: ID> ksni::Tray for TrayData<I> {
    fn id(&self) -> String {
        self.name.clone().to_string()
    }

    fn title(&self) -> String {
        self.name.clone().to_string()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        self.icon
            .as_ref()
            .map(|i| vec![i.into()])
            .unwrap_or_default()
    }
    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        self.items
            .lock()
            .unwrap()
            .iter()
            .map(ToKsni::to_ksni)
            .collect()
    }
}

impl<I: ID> Tray<I> {
    pub fn new(
        name: Arc<str>,
        items: Arc<Mutex<Vec<MenuEntry<I>>>>,
        icon: Option<TrayIcon>,
    ) -> Self {
        let data = TrayData { name, items, icon };
        Self(data.clone(), Arc::new(Mutex::new(None)))
    }

    pub fn start(self) {
        let service = ksni::TrayService::new(self.0.clone());
        let handle = service.handle();
        service.spawn();
        *self.1.lock().unwrap() = Some(handle);
    }

    pub fn update(&mut self) {
        if let Some(handle) = self.1.lock().unwrap().as_ref() {
            handle.update(|_| {});
        }
    }

    pub fn close(&self) {
        if let Some(handle) = self.1.lock().unwrap().as_ref() {
            handle.shutdown();
        }
    }

    pub fn get_name(&self) -> Arc<str> {
        self.0.name.clone()
    }

    pub fn get_icon(&self) -> Option<TrayIcon> {
        self.0.icon.clone()
    }

    pub fn set_name(&mut self, name: Option<Arc<str>>) {
        if let Some(name) = name {
            self.0 = TrayData {
                name,
                ..self.0.clone()
            };
        }
    }

    pub fn set_icon(&mut self, icon: Option<TrayIcon>) {
        self.0 = TrayData {
            icon,
            ..self.0.clone()
        };
    }
}

trait ToKsni<I: ID>: Sized {
    fn to_ksni(&self) -> ksni::MenuItem<TrayData<I>>;
}

impl<I: ID> ToKsni<I> for MenuEntry<I> {
    fn to_ksni(&self) -> ksni::MenuItem<TrayData<I>> {
        match self {
            MenuEntry::Item(_id, item) => {
                let mut ksni_item = StandardItem {
                    label: item.label.clone(),
                    enabled: item.enabled,
                    visible: true,
                    activate: {
                        let action = item.action.clone();
                        Box::new(move |_| action())
                    },
                    ..Default::default()
                };

                if let Some(icon) = item.icon.clone() {
                    ksni_item.icon_data = icon.data_raw.to_vec();
                }

                ksni::MenuItem::Standard(ksni_item)
            }
            MenuEntry::Separator => ksni::MenuItem::Separator,
            MenuEntry::Label(_id, icon, label) => {
                let mut ksni_item = StandardItem {
                    enabled: false,
                    label: label.clone(),
                    visible: true,
                    ..Default::default()
                };

                if let Some(icon) = icon.clone() {
                    ksni_item.icon_data = icon.data_raw.to_vec();
                }

                ksni::MenuItem::Standard(ksni_item)
            }
            MenuEntry::LabelCheck(_id, checked, icon, label) => {
                let mut ksni_item = CheckmarkItem {
                    enabled: false,
                    label: label.clone(),
                    visible: true,
                    checked: *checked,
                    ..Default::default()
                };

                if let Some(icon) = icon.clone() {
                    ksni_item.icon_data = icon.data_raw.to_vec();
                }

                ksni::MenuItem::Checkmark(ksni_item)
            }
            MenuEntry::SubMenu(_id, enabled, icon, label, items) => {
                let mut ksni_item = ksni::menu::SubMenu {
                    label: label.clone(),
                    enabled: *enabled,
                    visible: true,
                    submenu: items.iter().map(ToKsni::to_ksni).collect(),
                    ..Default::default()
                };

                if let Some(icon) = icon.clone() {
                    ksni_item.icon_data = icon.data_raw.to_vec();
                }

                ksni::MenuItem::SubMenu(ksni_item)
            }
        }
    }
}

impl Into<ksni::Icon> for &TrayIcon {
    fn into(self) -> ksni::Icon {
        let data = self
            .data
            .chunks_exact(4)
            .flat_map(|v| {
                if let [r, g, b, a] = *v {
                    return [a, r, g, b];
                }
                [v[0], v[1], v[2], v[3]]
            })
            .collect();

        ksni::Icon {
            width: self.width as _,
            height: self.height as _,
            data,
        }
    }
}
