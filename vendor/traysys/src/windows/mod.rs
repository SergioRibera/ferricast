use std::ops::AddAssign;
use std::os::windows::ffi::OsStrExt;
use std::ptr::null_mut;
use std::sync::{Arc, Mutex};
use windows_sys::Win32::Foundation::{FALSE, HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::HBITMAP;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Shell::{
    Shell_NotifyIconW, NIF_ICON, NIF_MESSAGE, NIF_TIP, NIM_ADD, NIM_DELETE, NOTIFYICONDATAW,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreatePopupMenu, CreateWindowExW, DefWindowProcW, DestroyMenu, DispatchMessageW,
    GetCursorPos, GetMenuItemCount, GetMessageW, GetWindowLongPtrW, PostQuitMessage,
    RegisterClassW, SetForegroundWindow, SetMenuItemBitmaps, SetWindowLongPtrW, TrackPopupMenu,
    TranslateMessage, CW_USEDEFAULT, GWLP_USERDATA, HMENU, MF_BYPOSITION, MF_CHECKED, MF_DISABLED,
    MF_GRAYED, MF_POPUP, MF_SEPARATOR, MF_STRING, MF_UNCHECKED, MSG, TPM_NONOTIFY, TPM_RETURNCMD,
    TPM_RIGHTBUTTON, WM_COMMAND, WM_DESTROY, WM_RBUTTONDOWN, WNDCLASSW, WS_EX_LAYERED,
    WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TRANSPARENT, WS_OVERLAPPED,
};

mod icon;

use icon::*;

use crate::{MenuEntry, TrayIcon, ID};

#[derive(Clone)]
pub struct Tray<I: ID> {
    name: Arc<str>,
    items: Arc<Mutex<Vec<MenuEntry<I>>>>,
    icon: Option<TrayIcon>,
    hwnd: HWND,
    notify_id: u32,
}

impl<I: ID> Tray<I> {
    pub fn new(
        name: Arc<str>,
        items: Arc<Mutex<Vec<MenuEntry<I>>>>,
        icon: Option<TrayIcon>,
    ) -> Self {
        let notify_id = 0;

        unsafe {
            let class_name = format!("{}_tray", &name.as_ref());
            let class_name: Vec<_> = class_name.encode_utf16().chain(Some(0)).collect();
            let hinstance = get_instance_handle() as *mut _;

            let wnd_class = WNDCLASSW {
                lpfnWndProc: Some(tray_proc::<I>),
                lpszClassName: class_name.as_ptr(),
                hInstance: hinstance,
                ..std::mem::zeroed()
            };

            RegisterClassW(&wnd_class);

            let hwnd = CreateWindowExW(
                WS_EX_NOACTIVATE | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW,
                class_name.as_ptr(),
                std::ptr::null(),
                WS_OVERLAPPED,
                CW_USEDEFAULT,
                0,
                CW_USEDEFAULT,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinstance,
                std::ptr::null_mut(),
            );

            Self {
                name,
                items,
                icon: icon.clone(),
                hwnd,
                notify_id,
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

    fn create_menu(&self) -> HMENU {
        let menu = unsafe { CreatePopupMenu() };
        let items = self.items.lock().unwrap();
        let mut entry_id = 0;

        for entry in items.iter() {
            self.append_menu_entry(menu, entry, &mut entry_id);
        }

        menu
    }

    fn append_menu_entry(&self, menu: HMENU, entry: &MenuEntry<I>, entry_id: &mut u32) {
        match entry {
            MenuEntry::Item(_id, item) => {
                let label_wide: Vec<u16> = item.label.encode_utf16().chain(Some(0)).collect();

                let mut flags = MF_STRING;
                if !item.enabled {
                    flags |= MF_GRAYED | MF_DISABLED;
                }
                if item.checked {
                    flags |= MF_CHECKED;
                } else {
                    flags |= MF_UNCHECKED;
                }
                unsafe {
                    // Agregamos el ítem con su command ID.
                    AppendMenuW(menu, flags, *entry_id as usize, label_wide.as_ptr());
                }
                // Obtener la posición real del nuevo ítem.
                let pos = unsafe { GetMenuItemCount(menu) - 1 };
                unsafe {
                    if let Some(tray_icon) = item.icon.as_ref() {
                        if let Some(hbmp_icon) = Into::<Option<HBITMAP>>::into(tray_icon.clone()) {
                            SetMenuItemBitmaps(
                                menu,
                                pos as u32,
                                MF_BYPOSITION,
                                hbmp_icon,
                                hbmp_icon,
                            );
                        }
                    }
                }
            }
            MenuEntry::LabelCheck(_id, checked, icon, label) => {
                let label_wide: Vec<u16> = label.encode_utf16().chain(Some(0)).collect();
                let mut flags = MF_STRING | MF_DISABLED | MF_GRAYED;
                if *checked {
                    flags |= MF_CHECKED;
                } else {
                    flags |= MF_UNCHECKED;
                }
                unsafe {
                    AppendMenuW(menu, flags, *entry_id as usize, label_wide.as_ptr());
                }
                let pos = unsafe { GetMenuItemCount(menu) - 1 };
                unsafe {
                    if let Some(tray_icon) = icon.as_ref() {
                        if let Some(hbmp_icon) = Into::<Option<HBITMAP>>::into(tray_icon.clone()) {
                            SetMenuItemBitmaps(
                                menu,
                                pos as u32,
                                MF_BYPOSITION,
                                hbmp_icon,
                                hbmp_icon,
                            );
                        }
                    }
                }
            }
            MenuEntry::Label(_id, icon, label) => {
                let label_wide: Vec<u16> = label.encode_utf16().chain(Some(0)).collect();
                let flags = MF_STRING | MF_DISABLED | MF_GRAYED;
                unsafe {
                    AppendMenuW(menu, flags, *entry_id as usize, label_wide.as_ptr());
                }
                let pos = unsafe { GetMenuItemCount(menu) - 1 };
                unsafe {
                    if let Some(tray_icon) = icon.as_ref() {
                        if let Some(hbmp_icon) = Into::<Option<HBITMAP>>::into(tray_icon.clone()) {
                            SetMenuItemBitmaps(
                                menu,
                                pos as u32,
                                MF_BYPOSITION,
                                hbmp_icon,
                                hbmp_icon,
                            );
                        }
                    }
                }
            }
            MenuEntry::Separator => unsafe {
                AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null_mut());
            },
            MenuEntry::SubMenu(_id, enabled, icon, label, submenu) => {
                let submenu_handle = unsafe { CreatePopupMenu() };
                for subentry in submenu {
                    self.append_menu_entry(submenu_handle, subentry, entry_id);
                }

                let label_wide: Vec<u16> = label.encode_utf16().chain(Some(0)).collect();
                let mut flags = MF_POPUP;
                if !*enabled {
                    flags |= MF_GRAYED | MF_DISABLED;
                }
                unsafe {
                    AppendMenuW(menu, flags, submenu_handle as usize, label_wide.as_ptr());
                }
                let pos = unsafe { GetMenuItemCount(menu) - 1 };
                unsafe {
                    if let Some(tray_icon) = icon.as_ref() {
                        if let Some(hbmp_icon) = Into::<Option<HBITMAP>>::into(tray_icon.clone()) {
                            SetMenuItemBitmaps(
                                menu,
                                pos as u32,
                                MF_BYPOSITION,
                                hbmp_icon,
                                hbmp_icon,
                            );
                        }
                    }
                }
            }
        }
        entry_id.add_assign(1);
    }

    pub fn start(self) {
        _ = unsafe { GetModuleHandleW(null_mut()) };
        let icon_handle = self
            .icon
            .clone()
            .map(|icon| Into::<SafeHICON>::into(icon))
            .unwrap();
        let mut nid = NOTIFYICONDATAW {
            hWnd: self.hwnd,
            uID: self.notify_id,
            uFlags: NIF_ICON | NIF_MESSAGE | NIF_TIP,
            uCallbackMessage: 6002,
            hIcon: icon_handle.0 as _,
            szTip: [0; 128],
            ..unsafe { std::mem::zeroed() }
        };
        let tip = std::ffi::OsString::from(self.name.to_string())
            .encode_wide()
            .collect::<Vec<_>>();
        unsafe {
            std::ptr::copy_nonoverlapping(tip.as_ptr(), nid.szTip.as_mut_ptr(), tip.len());
        }

        let hwnd = self.hwnd;
        let tray_ptr = Box::into_raw(Box::new(self));
        unsafe {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, tray_ptr as isize);
            Shell_NotifyIconW(NIM_ADD, &mut nid);
            let mut message: MSG = std::mem::zeroed();
            while GetMessageW(&mut message, null_mut(), 0, 0) > 0 {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
    }

    pub fn close(&self) {
        internal_quit::<I>(self.hwnd);
    }

    fn _handle_callback(step: &mut usize, items: &[MenuEntry<I>], id: usize) {
        for entry in items {
            match entry {
                MenuEntry::Item(_id, item) if *step == id => {
                    (item.action)();
                    break;
                }
                MenuEntry::SubMenu(_id, _enabled, _icon, _label, submenu) => {
                    Self::_handle_callback(step, submenu, id);
                }
                _ => {}
            }
            step.add_assign(1);
        }
    }

    fn handle_menu_command(&self, id: u32) {
        let items = self.items.lock().unwrap();
        let mut step = 0;
        Self::_handle_callback(&mut step, &items, id as usize);
    }
}

// taken from winit's code base
// https://github.com/rust-windowing/winit/blob/ee88e38f13fbc86a7aafae1d17ad3cd4a1e761df/src/platform_impl/windows/util.rs#L138
pub fn get_instance_handle() -> windows_sys::Win32::Foundation::HMODULE {
    // Gets the instance handle by taking the address of the
    // pseudo-variable created by the microsoft linker:
    // https://devblogs.microsoft.com/oldnewthing/20041025-00/?p=37483

    // This is preferred over GetModuleHandle(NULL) because it also works in DLLs:
    // https://stackoverflow.com/questions/21718027/getmodulehandlenull-vs-hinstance

    extern "C" {
        static __ImageBase: windows_sys::Win32::System::SystemServices::IMAGE_DOS_HEADER;
    }

    unsafe { &__ImageBase as *const _ as _ }
}

unsafe extern "system" fn tray_proc<I: ID>(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        6002 => match lparam as u32 {
            WM_RBUTTONDOWN => {
                let tray_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut Tray<I>;
                let tray = &*tray_ptr;

                let mut cursor_pos = std::mem::zeroed();
                GetCursorPos(&mut cursor_pos);

                let menu = tray.create_menu();
                SetForegroundWindow(hwnd);
                let cmd = TrackPopupMenu(
                    menu,
                    TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON,
                    cursor_pos.x,
                    cursor_pos.y,
                    0,
                    hwnd,
                    null_mut(),
                );
                tray.handle_menu_command(cmd as u32);
                DestroyMenu(menu);
                return 0;
            }
            _ => DefWindowProcW(hwnd, msg, wparam, lparam),
        },
        WM_COMMAND => {
            let tray_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut Tray<I>;
            let tray = &*tray_ptr;
            let cmd = wparam as u32;
            tray.handle_menu_command(cmd);
            0
        }
        WM_DESTROY => {
            internal_quit::<I>(hwnd);
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

fn internal_quit<I: ID>(hwnd: HWND) {
    unsafe {
        let tray_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut Tray<I>;

        _ = Box::from_raw(tray_ptr);

        let mut nid = NOTIFYICONDATAW {
            uFlags: NIF_ICON,
            hWnd: hwnd,
            ..std::mem::zeroed()
        };

        if Shell_NotifyIconW(NIM_DELETE, &mut nid as _) == FALSE {
            eprintln!("Error removing system tray icon");
        }

        PostQuitMessage(0);
    }
}
