use std::fmt::Debug;
use std::{fmt::Display, sync::Arc};

use crate::{SystemIcon, TrayIcon, ID};

/// Represents a single menu item in the tray.
#[derive(Clone, Default)]
pub struct MenuItemBuilder {
    pub label: Option<String>,
    pub enabled: Option<bool>,
    pub checked: Option<bool>,
    pub icon: Option<TrayIcon>,
    pub action: Option<Arc<dyn Fn() + Send + Sync + 'static>>,
}

/// Represents a single menu item in the tray.
#[derive(Clone)]
pub struct MenuItem {
    pub(crate) label: String,
    pub(crate) enabled: bool,
    pub(crate) checked: bool,
    pub(crate) icon: Option<TrayIcon>,
    pub(crate) action: Arc<dyn Fn() + Send + Sync + 'static>,
}

impl Debug for MenuItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MenuItem")
            .field("enabled", &self.enabled)
            .field("icon", &self.icon)
            .field("checked", &self.checked)
            .field("label", &self.label)
            .finish()
    }
}

/// Represents either a menu item or a submenu.
#[derive(Clone, Debug)]
pub enum MenuEntry<I: ID> {
    Separator,
    Label(I, Option<TrayIcon>, String),
    LabelCheck(I, bool, Option<TrayIcon>, String),
    Item(I, MenuItem),
    SubMenu(I, bool, Option<TrayIcon>, String, Vec<MenuEntry<I>>),
}

impl MenuItemBuilder {
    #[must_use]
    pub fn label<S: Into<String>>(mut self, label: S) -> Self {
        self.label = Some(label.into());
        self
    }

    #[must_use]
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = Some(enabled);
        self
    }

    #[must_use]
    pub fn checked(mut self, checked: bool) -> Self {
        self.checked = Some(checked);
        self
    }

    #[must_use]
    pub fn icon(mut self, icon: TrayIcon) -> Self {
        self.icon = Some(icon);
        self
    }

    #[must_use]
    pub fn system_icon(mut self, system_icon: SystemIcon) -> Self {
        if let Some(icon) = system_icon.to_tray_icon() {
            self.icon = Some(icon);
        }
        self
    }

    #[must_use]
    pub fn system_icon_for(mut self, system_icon: SystemIcon, supported_os: &[&str]) -> Self {
        #[cfg(target_os = "macos")]
        let current_os = "macos";
        #[cfg(target_os = "linux")]
        let current_os = "linux";
        #[cfg(target_os = "windows")]
        let current_os = "windows";

        if supported_os.contains(&current_os) {
            if let Some(icon) = system_icon.to_tray_icon() {
                self.icon = Some(icon);
            }
        }
        self
    }

    #[cfg(feature = "svg")]
    #[must_use]
    pub fn icon_svg(mut self, svg_data: &str, size: u32) -> Self {
        if let Ok(icon) = TrayIcon::from_svg(svg_data.as_bytes(), size) {
            self.icon = Some(icon);
        }
        self
    }

    #[cfg(feature = "svg")]
    #[must_use]
    pub fn icon_with_fallback(
        mut self,
        svg_data: Option<&str>,
        system_icon: SystemIcon,
        size: u32,
    ) -> Self {
        if let Some(svg) = svg_data {
            if let Ok(icon) = TrayIcon::from_svg(svg.as_bytes(), size) {
                self.icon = Some(icon);
                return self;
            }
        }
        if let Some(icon) = system_icon.to_tray_icon() {
            self.icon = Some(icon);
        }
        self
    }

    #[must_use]
    pub fn action<F: Fn() + Send + Sync + 'static>(mut self, action: F) -> Self {
        self.action = Some(Arc::new(action));
        self
    }

    #[must_use]
    pub fn build_cloned(&self) -> MenuItem {
        MenuItem {
            label: self.label.clone().unwrap_or_default(),
            icon: self.icon.clone(),
            enabled: self.enabled.unwrap_or(true),
            checked: self.checked.unwrap_or_default(),
            action: self.action.clone().unwrap_or_else(|| Arc::new(|| {})),
        }
    }

    #[must_use]
    pub fn build(self) -> MenuItem {
        MenuItem {
            icon: self.icon.clone(),
            label: self.label.unwrap_or_default(),
            enabled: self.enabled.unwrap_or(true),
            checked: self.checked.unwrap_or_default(),
            action: self.action.unwrap_or_else(|| Arc::new(|| {})),
        }
    }
}

pub trait ToEntryData {
    fn to_entry_data(self) -> MenuItemBuilder;
}

impl ToEntryData for String {
    fn to_entry_data(self) -> MenuItemBuilder {
        MenuItemBuilder {
            label: Some(self),
            ..Default::default()
        }
    }
}

impl ToEntryData for &str {
    fn to_entry_data(self) -> MenuItemBuilder {
        MenuItemBuilder {
            label: Some(self.into()),
            ..Default::default()
        }
    }
}

impl ToEntryData for MenuItemBuilder {
    fn to_entry_data(self) -> MenuItemBuilder {
        self
    }
}

impl<S> ToEntryData for (S, bool)
where
    S: Display,
{
    fn to_entry_data(self) -> MenuItemBuilder {
        let (label, enabled) = self;
        MenuItemBuilder {
            label: Some(label.to_string()),
            enabled: Some(enabled),
            ..Default::default()
        }
    }
}

impl<S, F> ToEntryData for (S, F)
where
    S: Display,
    F: Fn() + Send + Sync + 'static,
{
    fn to_entry_data(self) -> MenuItemBuilder {
        let (label, action) = self;
        MenuItemBuilder {
            label: Some(label.to_string()),
            action: Some(Arc::new(action)),
            ..Default::default()
        }
    }
}

impl<S, B, F> ToEntryData for (S, B, F)
where
    S: Display,
    B: Into<bool>,
    F: Fn() + Send + Sync + 'static,
{
    fn to_entry_data(self) -> MenuItemBuilder {
        let (label, checked, action) = self;
        MenuItemBuilder {
            label: Some(label.to_string()),
            checked: Some(checked.into()),
            action: Some(Arc::new(action)),
            ..Default::default()
        }
    }
}

impl<S, B, F> ToEntryData for (S, B, B, F)
where
    S: Display,
    B: Into<bool>,
    F: Fn() + Send + Sync + 'static,
{
    fn to_entry_data(self) -> MenuItemBuilder {
        let (label, enabled, checked, action) = self;
        MenuItemBuilder {
            label: Some(label.to_string()),
            enabled: Some(enabled.into()),
            checked: Some(checked.into()),
            action: Some(Arc::new(action)),
            ..Default::default()
        }
    }
}

impl<I: ID, D: ToEntryData> From<(I, D)> for MenuEntry<I> {
    fn from((id, data): (I, D)) -> Self {
        let item = data.to_entry_data();

        if item.checked.is_some()
            && item.label.is_some()
            && item.enabled.is_none()
            && item.action.is_none()
        {
            let item = item.build_cloned();
            return MenuEntry::LabelCheck(id, item.checked, None, item.label);
        }

        if item.label.is_some() && item.enabled.is_none() && item.action.is_none() {
            let item = item.build_cloned();
            return MenuEntry::Label(id, None, item.label);
        }

        MenuEntry::Item(id, item.build())
    }
}

impl From<MenuItemBuilder> for MenuItem {
    fn from(builder: MenuItemBuilder) -> Self {
        builder.build()
    }
}

// impl<F: Fn(), I: ID, D: ToEntryData<F>> From<(I, D)> for MenuEntry<I> {
//     fn from((id, data): (I, D)) -> Self {
//         let item = data.to_entry_data();

//         if item.checked.is_some()
//             && item.label.is_some()
//             && item.enabled.is_none()
//             && item.action.is_none()
//         {
//             let item = item.build_cloned();
//             return MenuEntry::LabelCheck(id, item.checked, item.label);
//         }

//         if item.label.is_some() && item.enabled.is_none() && item.action.is_none() {
//             let item = item.build_cloned();
//             return MenuEntry::Label(id, item.label);
//         }

//         MenuEntry::Item(id, item.build())
//     }
// }
