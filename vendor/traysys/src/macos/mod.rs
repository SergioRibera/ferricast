use crate::{MenuEntry, TrayIcon, ID};
use cocoa::appkit::{NSApplicationActivationPolicyAccessory, NSMenuItem, NSStatusBar};
use cocoa::base::{id, nil, NO, YES};
use cocoa::foundation::{NSAutoreleasePool, NSString};
use objc::declare::ClassDecl;
use objc::runtime::{Object, Sel};
use objc::{class, msg_send, sel, sel_impl};
use std::sync::{Arc, Mutex};

mod callback;
pub mod sf_symbols;

use callback::Callback;

const MENU_ICON_HEIGHT: f64 = 13.0;
const TRAY_ICON_HEIGHT: f64 = 18.0;
const NSWINDOW_COLLECTION_BEHAVIOR_CAN_JOIN_ALL_SPACES: u64 = 1 << 0;

#[repr(C)]
#[derive(Debug)]
struct NSPoint {
    x: CGFloat,
    y: CGFloat,
}

type CGFloat = f64;

#[repr(C)]
struct NSRect {
    origin: NSPoint,
    size: NSSize,
}

#[repr(C)]
struct NSSize {
    width: f64,
    height: f64,
}

impl NSSize {
    fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

impl<I: ID> Drop for Tray<I> {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self._pool, release];
        }
    }
}

#[derive(Clone)]
pub struct Tray<I: ID> {
    name: Arc<str>,
    icon: Option<TrayIcon>,
    items: Arc<Mutex<Vec<MenuEntry<I>>>>,
    min_width: Option<f64>,
    app: id,
    menu: id,
    status_item: id,
    _objc_instance: id,
    _pool: id,
}

unsafe impl<I: ID> Sync for Tray<I> {}
unsafe impl<I: ID> Send for Tray<I> {}

impl<I: ID> Tray<I> {
    pub fn new(
        name: Arc<str>,
        items: Arc<Mutex<Vec<MenuEntry<I>>>>,
        icon: Option<TrayIcon>,
        min_width: Option<f64>,
    ) -> Self {
        unsafe {
            let superclass = class!(NSObject);
            let mut decl = ClassDecl::new("MacOSIcon", superclass).unwrap();

            decl.add_method(
                sel!(statusItemClicked:),
                Self::status_item_clicked as extern "C" fn(&Object, _, id),
            );

            let objc_class = decl.register();
            let objc_instance: id = msg_send![objc_class, new];

            let _pool: id = NSAutoreleasePool::new(nil);
            let app: id = msg_send![class!(NSApplication), sharedApplication];
            let _: () = msg_send![app, setActivationPolicy: NSApplicationActivationPolicyAccessory];

            let status_bar: id = NSStatusBar::systemStatusBar(nil).autorelease();
            let status_item: id = msg_send![status_bar, statusItemWithLength: -1.0];

            let tooltip: id = NSString::alloc(nil).init_str(name.as_ref());
            let _: () = msg_send![status_item, setToolTip: tooltip];

            let button: id = msg_send![status_item, button];
            if !button.is_null() {
                let window: id = msg_send![button, window];
                if !window.is_null() {
                    let behavior: u64 = msg_send![window, collectionBehavior];
                    let new_behavior = behavior | NSWINDOW_COLLECTION_BEHAVIOR_CAN_JOIN_ALL_SPACES;
                    let _: () = msg_send![window, setCollectionBehavior: new_behavior];
                }
            }

            Self {
                name,
                app,
                icon,
                items,
                min_width,
                _pool,
                status_item,
                _objc_instance: objc_instance,
                menu: std::ptr::null_mut(),
            }
        }
    }

    pub fn get_name(&self) -> Arc<str> {
        self.name.clone()
    }

    pub fn get_icon(&self) -> Option<TrayIcon> {
        self.icon.clone()
    }

    pub fn set_name(&mut self, name: Option<Arc<str>>) {
        if let Some(name) = name {
            self.name = name;
        }
    }

    pub fn set_icon(&mut self, icon: Option<TrayIcon>) {
        self.icon = icon;
    }

    extern "C" fn status_item_clicked(status_item: &Object, _sel: Sel, _sender: id) {
        unsafe {
            let menu: id = msg_send![status_item, menu];
            if !menu.is_null() {
                let button: id = msg_send![status_item, button];
                let frame: NSRect = msg_send![button, frame];
                let point = NSPoint {
                    x: frame.origin.x,
                    y: frame.origin.y,
                };
                let _: () = msg_send![menu, popUpMenuPositioningItem:nil
                    atLocation:point
                    inView:nil];
            }
        }
    }

    fn generate_menu(items: &[MenuEntry<I>], min_width: Option<f64>) -> id {
        unsafe {
            let menu: id = msg_send![class!(NSMenu), alloc];
            let menu: id = msg_send![menu, initWithTitle: NSString::alloc(nil).init_str("")];

            if let Some(width) = min_width {
                let _: () = msg_send![menu, setMinimumWidth:width];
            }

            for item in items {
                match item {
                    MenuEntry::Separator => {
                        let item: id = msg_send![class!(NSMenuItem), separatorItem];
                        let _: () = msg_send![menu, addItem: item];
                    }
                    MenuEntry::Label(_id, icon, label) => {
                        let title_ns = NSString::alloc(nil).init_str(&label);
                        let menu_item: id = msg_send![class!(NSMenuItem), alloc];
                        let menu_item: id = msg_send![
                            menu_item,
                            initWithTitle: title_ns
                            action: sel!(call)
                            keyEquivalent: NSString::alloc(nil).init_str("")
                        ];
                        if let Some(TrayIcon {
                            data_raw: data,
                            width,
                            height,
                            is_template,
                            ..
                        }) = icon.as_ref()
                        {
                            let data: id = msg_send![class!(NSData),
                                dataWithBytes:data.as_ptr()
                                length:data.len()];

                            let icon: id = msg_send![class!(NSImage), alloc];
                            let icon: id = msg_send![icon, initWithData:data];
                            let _: () =
                                msg_send![icon, setTemplate: if *is_template { YES } else { NO }];
                            let icon_width: f64 =
                                MENU_ICON_HEIGHT * (*width as f64 / *height as f64);
                            let _: () =
                                msg_send![icon, setSize:NSSize::new(icon_width, MENU_ICON_HEIGHT)];
                            let _: () = msg_send![menu_item, setImage: icon];
                        }
                        let _: () = msg_send![menu, addItem: menu_item];
                        let _: () = msg_send![menu_item, setEnabled: NO];
                    }
                    MenuEntry::LabelCheck(_id, checked, icon, label) => {
                        let title_ns = NSString::alloc(nil).init_str(&label);
                        let menu_item: id = msg_send![class!(NSMenuItem), alloc];
                        let menu_item: id = msg_send![
                            menu_item,
                            initWithTitle: title_ns
                            action: sel!(call)
                            keyEquivalent: NSString::alloc(nil).init_str("")
                        ];
                        if let Some(TrayIcon {
                            data_raw: data,
                            width,
                            height,
                            is_template,
                            ..
                        }) = icon.as_ref()
                        {
                            let data: id = msg_send![class!(NSData),
                                dataWithBytes:data.as_ptr()
                                length:data.len()];

                            let icon: id = msg_send![class!(NSImage), alloc];
                            let icon: id = msg_send![icon, initWithData:data];
                            let _: () =
                                msg_send![icon, setTemplate: if *is_template { YES } else { NO }];
                            let icon_width: f64 =
                                MENU_ICON_HEIGHT * (*width as f64 / *height as f64);
                            let _: () =
                                msg_send![icon, setSize:NSSize::new(icon_width, MENU_ICON_HEIGHT)];
                            let _: () = msg_send![menu_item, setImage: icon];
                        }
                        let _: () = msg_send![menu, addItem: menu_item];
                        let _: () = msg_send![menu_item, setEnabled: NO];
                        let _: () = msg_send![menu_item, setState: if *checked { 1 } else { 0 }];
                    }
                    MenuEntry::Item(_, item) => {
                        let title_ns = NSString::alloc(nil).init_str(&item.label);
                        let menu_item = NSMenuItem::alloc(nil).initWithTitle_action_keyEquivalent_(
                            title_ns,
                            sel!(call),
                            NSString::alloc(nil).init_str(""),
                        );
                        if let Some(TrayIcon {
                            data_raw: data,
                            width,
                            height,
                            is_template,
                            ..
                        }) = item.icon.as_ref()
                        {
                            let data: id = msg_send![class!(NSData),
                                dataWithBytes:data.as_ptr()
                                length:data.len()];

                            let icon: id = msg_send![class!(NSImage), alloc];
                            let icon: id = msg_send![icon, initWithData:data];
                            let _: () =
                                msg_send![icon, setTemplate: if *is_template { YES } else { NO }];
                            let icon_width: f64 =
                                MENU_ICON_HEIGHT * (*width as f64 / *height as f64);
                            let _: () =
                                msg_send![icon, setSize:NSSize::new(icon_width, MENU_ICON_HEIGHT)];
                            let _: () = msg_send![menu_item, setImage: icon];
                        }
                        if item.enabled {
                            let cb_obj = Callback::from(item.action.clone());
                            let _: () = msg_send![menu_item, setTarget: cb_obj];
                        }
                        let _: () = msg_send![menu, addItem: menu_item];
                        let _: () =
                            msg_send![menu_item, setEnabled: if item.enabled { YES } else { NO }];
                        let _: () =
                            msg_send![menu_item, setState: if item.checked { 1 } else { 0 }];
                    }
                    MenuEntry::SubMenu(_id, enabled, icon, label, items) => {
                        let submenu = Self::generate_menu(&items, None);
                        if submenu.is_null() {
                            continue;
                        }
                        let title_ns = NSString::alloc(nil).init_str(&label);
                        let menu_item: id = msg_send![class!(NSMenuItem), alloc];
                        let menu_item: id = msg_send![
                            menu_item,
                            initWithTitle: title_ns
                            action: sel!(submenuAction:)
                            keyEquivalent: NSString::alloc(nil).init_str("")
                        ];
                        if let Some(TrayIcon {
                            data_raw: data,
                            width,
                            height,
                            is_template,
                            ..
                        }) = icon.as_ref()
                        {
                            let data: id = msg_send![class!(NSData),
                                dataWithBytes:data.as_ptr()
                                length:data.len()];

                            let icon: id = msg_send![class!(NSImage), alloc];
                            let icon: id = msg_send![icon, initWithData:data];
                            let _: () =
                                msg_send![icon, setTemplate: if *is_template { YES } else { NO }];
                            let icon_width: f64 =
                                MENU_ICON_HEIGHT * (*width as f64 / *height as f64);
                            let _: () =
                                msg_send![icon, setSize:NSSize::new(icon_width, MENU_ICON_HEIGHT)];
                            let _: () = msg_send![menu_item, setImage: icon];
                        }
                        let _: () = msg_send![menu_item, setSubmenu: submenu];
                        let _: () = msg_send![menu, addItem: menu_item];
                        let _: () =
                            msg_send![menu_item, setEnabled: if *enabled { YES } else { NO }];
                    }
                }
            }
            menu
        }
    }

    fn build_menu(&mut self) {
        unsafe {
            let button: id = msg_send![self.status_item, button];
            if button.is_null() {
                return;
            }

            if let Some(TrayIcon {
                data_raw: data,
                width,
                height,
                is_template,
                ..
            }) = self.icon.as_ref()
            {
                let data: id = msg_send![class!(NSData),
                    dataWithBytes:data.as_ptr()
                    length:data.len()];

                let icon: id = msg_send![class!(NSImage), alloc];
                let icon: id = msg_send![icon, initWithData:data];
                let _: () = msg_send![icon, setTemplate:YES];
                let icon_width: f64 = TRAY_ICON_HEIGHT * (*width as f64 / *height as f64);
                let _: () = msg_send![icon, setSize:NSSize::new(icon_width, TRAY_ICON_HEIGHT)];
                let _: () = msg_send![button, setImage:icon];
            }

            // let _: () = msg_send![self.menu, release];
            let new_menu = {
                let items = self.items.lock().unwrap();
                Self::generate_menu(&items, self.min_width)
            };
            if new_menu.is_null() {
                return;
            }
            self.menu = new_menu;

            let _: () = msg_send![self.status_item, setMenu:new_menu];

            let _: () = msg_send![button, setTarget:self.status_item];
            let _: () = msg_send![button, setAction:sel!(statusItemClicked:)];
        }
    }

    pub fn start(&mut self) {
        unsafe {
            self.build_menu();

            // Really Run
            let _: () = msg_send![self.app, activateIgnoringOtherApps:YES];
            let _: () = msg_send![self.app, run];
        }
    }

    pub fn update(&mut self) {
        self.build_menu();
    }

    pub fn close(&self) {
        unsafe {
            let _: () = msg_send![self.app, terminate:nil];
        }
    }
}
