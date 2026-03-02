use std::sync::Arc;

use crate::{MenuEntry, MenuItem, MenuItemBuilder, ToEntryData, TrayIcon, TrayMenu, ID};

/// The main tray menu.
pub struct TrayMenuBuilder<I: ID, const SUBMENU: bool = false> {
    pub(crate) name: Option<Arc<str>>,
    pub(crate) icon: Option<TrayIcon>,
    pub(crate) items: Vec<MenuEntry<I>>,
    pub(crate) min_width: Option<f64>,
}

impl<I: ID> Default for TrayMenuBuilder<I> {
    fn default() -> Self {
        Self {
            icon: None,
            name: None,
            items: Vec::new(),
            min_width: None,
        }
    }
}

pub(crate) fn update<I: ID>(
    id: I,
    items: &mut Vec<MenuEntry<I>>,
    cb_icon: Arc<dyn Fn(&mut Option<TrayIcon>)>,
    cb_enabled: Arc<dyn Fn(&mut bool)>,
    cb_checked: Arc<dyn Fn(&mut bool)>,
    cb_item: Arc<dyn Fn(&mut MenuItem)>,
    cb_label: Arc<dyn Fn(&mut String)>,
    cb_submenu: Arc<dyn Fn(&mut Vec<MenuEntry<I>>)>,
) {
    for item in items.iter_mut() {
        match item {
            MenuEntry::Item(item_id, menu_item) if id == *item_id => {
                cb_item(menu_item);
                break;
            }
            MenuEntry::Label(item_id, icon, old_label) if id == *item_id => {
                cb_label(old_label);
                cb_icon(icon);
                break;
            }
            MenuEntry::LabelCheck(item_id, old_checked, icon, old_label) if id == *item_id => {
                cb_icon(icon);
                cb_label(old_label);
                cb_checked(old_checked);
                break;
            }
            MenuEntry::SubMenu(item_id, enabled, icon, label, submenu) if id == *item_id => {
                cb_icon(icon);
                cb_enabled(enabled);
                cb_label(label);
                cb_submenu(submenu);
                break;
            }
            MenuEntry::SubMenu(_, _, icon, _label, submenu) => {
                cb_icon(icon);
                update(
                    id.clone(),
                    submenu,
                    cb_icon.clone(),
                    cb_enabled.clone(),
                    cb_checked.clone(),
                    cb_item.clone(),
                    cb_label.clone(),
                    cb_submenu.clone(),
                );
                continue;
            }
            MenuEntry::Item(_, _) => {}
            MenuEntry::Label(_, _, _) => {}
            MenuEntry::Separator => {}
            MenuEntry::LabelCheck(_, _, _, _) => {}
        }
    }
}

impl<I: ID, const SUBMENU: bool> TrayMenuBuilder<I, SUBMENU> {
    /// Add Menu Entry
    pub fn add<M: Into<MenuEntry<I>>>(mut self, entry: M) -> Self {
        self.items.push(entry.into());
        self
    }

    pub fn contains(&self, id: &I) -> bool {
        self.items.iter().any(|me| match me {
            MenuEntry::Separator => false,
            MenuEntry::Label(i, ..)
            | MenuEntry::LabelCheck(i, ..)
            | MenuEntry::Item(i, ..)
            | MenuEntry::SubMenu(i, ..) => i == id,
        })
    }

    /// Adds a menu item to the tray.
    pub fn add_item<M>(mut self, id: I, item: M) -> Self
    where
        M: ToEntryData,
    {
        self.items.push(MenuEntry::from((id, item)));
        self
    }

    /// Adds a menu item to the tray.
    pub fn add_checked<S>(mut self, id: I, checked: bool, label: S, icon: Option<TrayIcon>) -> Self
    where
        S: Into<String>,
    {
        self.items
            .push(MenuEntry::LabelCheck(id, checked, icon, label.into()));
        self
    }

    /// Adds a separator to the tray.
    pub fn add_separator(mut self) -> Self {
        self.items.push(MenuEntry::Separator);
        self
    }

    /// Adds a label to the tray.
    pub fn add_label<S>(mut self, id: I, label: S, icon: Option<TrayIcon>) -> Self
    where
        S: Into<String>,
    {
        self.items.push(MenuEntry::Label(id, icon, label.into()));
        self
    }

    /// Adds a submenu to the tray.
    pub fn add_submenu<S, F>(
        mut self,
        id: I,
        enabled: bool,
        label: S,
        icon: Option<TrayIcon>,
        create: F,
    ) -> Self
    where
        S: Into<String>,
        F: Fn(TrayMenuBuilder<I, true>) -> TrayMenuBuilder<I, true> + Send + Sync,
    {
        let label = label.into();
        let submenu = TrayMenuBuilder::<I, true> {
            name: Some(label.clone().into()),
            icon: None,
            items: Default::default(),
            min_width: None,
        };
        let submenu = create(submenu);
        self.items
            .push(MenuEntry::SubMenu(id, enabled, icon, label, submenu.items));
        self
    }
}

impl<I: ID> TrayMenuBuilder<I> {
    /// Sets the name of the tray.
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        let name = name.into();
        self.name = Some(name.into());
        self
    }

    /// Sets the icon of the tray.
    pub fn icon(mut self, icon: TrayIcon) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Sets the minimum width of the menu in pixels.
    pub fn min_width(mut self, width: f64) -> Self {
        self.min_width = Some(width);
        self
    }

    /// Builds the tray menu and starts it (stub for now).
    pub fn build(self) -> TrayMenu<I> {
        TrayMenu::new(self.name, self.items, self.icon, self.min_width)
    }
}

impl<I: ID> TrayMenuBuilder<I, true> {
    /// Clears the items of the submenu.
    pub fn clear(mut self) -> Self {
        self.items.clear();
        self
    }

    /// Sets the name of the tray.
    pub fn new_name<S: Into<String>>(mut self, name: S) -> Self {
        let name = name.into();
        self.name = Some(name.into());
        self
    }

    /// Removes with the given id.
    pub fn remove(mut self, id: I) -> Self {
        self.items.retain(|item| match item {
            MenuEntry::Item(item_id, _) => *item_id != id,
            MenuEntry::Label(item_id, _, _) => *item_id != id,
            MenuEntry::LabelCheck(item_id, _, _, _) => *item_id != id,
            MenuEntry::SubMenu(item_id, _, _, _, _) => *item_id != id,
            MenuEntry::Separator => true,
        });
        self
    }

    /// Updates the item with the given id.
    pub fn update_item<M: Into<MenuItemBuilder>>(mut self, id: I, item: M) -> Self {
        let item: MenuItemBuilder = item.into();
        update(
            id,
            &mut self.items,
            Arc::new(|_item_icon| {}),
            Arc::new(|_item_enabled| {}),
            Arc::new(|_item_checked| {}),
            Arc::new({
                let item = item.clone();
                move |old_item| {
                    if let Some(enabled) = item.enabled {
                        old_item.enabled = enabled;
                    }
                    if let Some(checked) = item.checked {
                        old_item.checked = checked;
                    }
                    if let Some(label) = item.label.as_ref() {
                        old_item.label = label.clone();
                    }
                    if let Some(action) = item.action.clone() {
                        old_item.action = action;
                    }
                }
            }),
            Arc::new(move |old_label| {
                if let Some(label) = item.label.clone() {
                    *old_label = label;
                }
            }),
            Arc::new(|_old_submenu| {}),
        );
        self
    }

    // Update Label.
    pub fn update_label<S>(mut self, id: I, label: S) -> Self
    where
        S: Into<String>,
    {
        let label = label.into();
        update(
            id,
            &mut self.items,
            Arc::new(|_item_icon| {}),
            Arc::new(|_item_enabled| {}),
            Arc::new(|_item_checked| {}),
            Arc::new({
                let label = label.clone();
                move |item| {
                    item.label = label.clone();
                }
            }),
            Arc::new(move |old_label| *old_label = label.clone()),
            Arc::new(|_old_submenu| {}),
        );
        self
    }

    /// Update icon
    pub fn update_icon(mut self, id: I, new_icon: TrayIcon) -> Self {
        update(
            id,
            &mut self.items,
            Arc::new({
                let new_icon = new_icon.clone();
                move |old_icon| {
                    old_icon.replace(new_icon.clone());
                }
            }),
            Arc::new(|_item_enabled| {}),
            Arc::new(|_item_checked| {}),
            Arc::new({
                let new_icon = new_icon.clone();
                move |item| {
                    item.icon.replace(new_icon.clone());
                }
            }),
            Arc::new(move |_old_label| {}),
            Arc::new(|_old_submenu| {}),
        );
        self
    }

    /// Update submenu in tray.
    pub fn update_submenu<S, F>(mut self, id: I, enabled: bool, label: S, create: F) -> Self
    where
        S: Into<String>,
        F: Fn(TrayMenuBuilder<I, true>) -> TrayMenuBuilder<I, true> + Send + Sync,
    {
        let label = label.into();
        let submenu = TrayMenuBuilder::<I, true> {
            name: Some(label.clone().into()),
            icon: None,
            items: Default::default(),
            min_width: None,
        };

        update(
            id.clone(),
            &mut self.items,
            Arc::new(|_item_icon| {}),
            Arc::new(move |old_enabled| *old_enabled = enabled),
            Arc::new(|_item_checked| {}),
            Arc::new({
                let label = label.clone();
                move |item| {
                    item.enabled = enabled;
                    item.label = label.clone();
                }
            }),
            Arc::new({
                let label = label.clone();
                move |old_label| *old_label = label.clone()
            }),
            Arc::new({
                let submenu = create(submenu);
                move |old_submenu| *old_submenu = submenu.items.clone()
            }),
        );

        self
    }
}
