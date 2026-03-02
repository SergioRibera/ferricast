use std::hash::Hash;
use std::sync::{Arc, Mutex};

mod builder;
mod error;
mod icon;
mod item;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
mod system_icon;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
use linux::Tray;
#[cfg(target_os = "macos")]
use macos::Tray;
#[cfg(target_os = "windows")]
use windows::Tray;

pub use builder::TrayMenuBuilder;
pub use error::Result;
pub use icon::TrayIcon;
pub use item::*;
pub use system_icon::SystemIcon;

pub trait ID: Hash + PartialEq + Clone + Send + Sync + 'static {}

impl<T: Hash + PartialEq + Clone + Send + Sync + 'static> ID for T {}

/// The main tray menu.
pub struct TrayMenu<I: ID> {
    items: Arc<Mutex<Vec<MenuEntry<I>>>>,

    platform: Arc<Mutex<Tray<I>>>,
}

unsafe impl<I: ID> Send for TrayMenu<I> {}
unsafe impl<I: ID> Sync for TrayMenu<I> {}

impl<I: ID> TrayMenu<I> {
    pub fn new(
        name: Option<Arc<str>>,
        items: Vec<MenuEntry<I>>,
        icon: Option<TrayIcon>,
        min_width: Option<f64>,
    ) -> Self {
        let items = Arc::new(Mutex::new(items));
        Self {
            items: items.clone(),
            #[cfg(target_os = "macos")]
            platform: Arc::new(Mutex::new(Tray::new(
                name.unwrap_or_default(),
                items,
                icon,
                min_width,
            ))),
            #[cfg(not(target_os = "macos"))]
            platform: Arc::new(Mutex::new(Tray::new(name.unwrap_or_default(), items, icon))),
        }
    }

    pub fn start(&self) {
        #[cfg(target_os = "macos")]
        let mut platform = {
            let platform_guard = self.platform.lock().unwrap();
            platform_guard.clone()
        };
        #[cfg(not(target_os = "macos"))]
        let platform = {
            let platform_guard = self.platform.lock().unwrap();
            platform_guard.clone()
        };
        platform.start();
    }

    pub fn contains(&self, id: &I) -> bool {
        self.items.lock().unwrap().iter().any(|me| match me {
            MenuEntry::Separator => false,
            MenuEntry::Label(i, ..)
            | MenuEntry::LabelCheck(i, ..)
            | MenuEntry::Item(i, ..)
            | MenuEntry::SubMenu(i, ..) => i == id,
        })
    }

    /// Removes with the given id.
    pub fn remove(&self, id: I) {
        {
            let mut items = self.items.lock().unwrap();
            items.retain(|item| match item {
                MenuEntry::Item(item_id, _) => *item_id != id,
                MenuEntry::Label(item_id, _, _) => *item_id != id,
                MenuEntry::LabelCheck(item_id, _, _, _) => *item_id != id,
                MenuEntry::SubMenu(item_id, _, _, _, _) => *item_id != id,
                MenuEntry::Separator => true,
            });
        }

        #[cfg(not(target_os = "windows"))]
        {
            let mut platform = self.platform.lock().unwrap();
            platform.update();
        }
    }

    pub fn update(
        &self,
        modifier: impl Fn(TrayMenuBuilder<I, true>) -> TrayMenuBuilder<I, true> + 'static,
    ) {
        let mut platform = self.platform.lock().unwrap();
        {
            let mut items = self.items.lock().unwrap();
            let submenu = TrayMenuBuilder::<I, true> {
                name: Some(platform.get_name()),
                icon: platform.get_icon(),
                items: items.clone(),
                min_width: None,
            };
            let traymenu = modifier(submenu);

            *items = traymenu.items;

            platform.set_name(traymenu.name);
            platform.set_icon(traymenu.icon);
        }
        #[cfg(not(target_os = "windows"))]
        platform.update();
    }

    pub fn update_icon(&self, id: I, new_icon: TrayIcon) {
        {
            let mut items = self.items.lock().unwrap();
            builder::update(
                id,
                &mut items,
                Arc::new({
                    let new_icon = new_icon.clone();
                    move |old_icon| {
                        old_icon.replace(new_icon.clone());
                    }
                }),
                Arc::new(|_| {}),
                Arc::new(|_| {}),
                Arc::new({
                    let new_icon = new_icon.clone();
                    move |item| {
                        item.icon.replace(new_icon.clone());
                    }
                }),
                Arc::new(|_| {}),
                Arc::new(move |_| {}),
            );
        }

        #[cfg(not(target_os = "windows"))]
        {
            let mut platform = self.platform.lock().unwrap();
            platform.update();
        }
    }

    /// Updates the label of a menu item (stub for dynamic updates).
    pub fn update_label<S: Into<String>>(&self, id: I, new_label: S) {
        let new_label = new_label.into();
        {
            let mut items = self.items.lock().unwrap();
            builder::update(
                id,
                &mut items,
                Arc::new(|_| {}),
                Arc::new(|_| {}),
                Arc::new(|_| {}),
                Arc::new({
                    let new_label = new_label.clone();
                    move |item| item.label = new_label.clone()
                }),
                Arc::new(move |old_label| *old_label = new_label.clone()),
                Arc::new(move |_| {}),
            );
        }

        #[cfg(not(target_os = "windows"))]
        {
            let mut platform = self.platform.lock().unwrap();
            platform.update();
        }
    }

    pub fn update_enabled(&self, id: I, enabled: bool) {
        {
            let mut items = self.items.lock().unwrap();
            builder::update(
                id,
                &mut items,
                Arc::new(|_| {}),
                Arc::new(move |item_enabled| *item_enabled = enabled),
                Arc::new(move |_| {}),
                Arc::new(move |item| item.enabled = enabled),
                Arc::new(move |_| {}),
                Arc::new(move |_| {}),
            );
        }

        #[cfg(not(target_os = "windows"))]
        {
            let mut platform = self.platform.lock().unwrap();
            platform.update();
        }
    }

    pub fn update_submenu(
        &self,
        id: I,
        modifier: impl Fn(TrayMenuBuilder<I, true>) -> TrayMenuBuilder<I, true> + 'static,
    ) {
        {
            let mut items = self.items.lock().unwrap();
            builder::update(
                id.clone(),
                &mut items,
                Arc::new(|_| {}),
                Arc::new(move |_| {}),
                Arc::new(move |_| {}),
                Arc::new(move |_item| {}),
                Arc::new(move |_old_label| {}),
                Arc::new(move |old_submenu| {
                    let submenu = TrayMenuBuilder::<I, true> {
                        name: None,
                        icon: None,
                        items: old_submenu.clone(),
                        min_width: None,
                    };
                    let submenu = modifier(submenu);
                    *old_submenu = submenu.items.clone();
                }),
            );
        }

        #[cfg(not(target_os = "windows"))]
        {
            let mut platform = self.platform.lock().unwrap();
            platform.update();
        }
    }

    pub fn close(&self) {
        {
            let platform = self.platform.lock().unwrap();
            platform.close();
        }
    }
}
