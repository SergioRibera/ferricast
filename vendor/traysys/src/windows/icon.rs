use std::ptr::null_mut;

use windows_sys::Win32::Graphics::Gdi::{
    CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, SelectObject, BITMAPINFO,
    BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS,
};
use windows_sys::Win32::Graphics::Gdi::{GetDC, ReleaseDC, HBITMAP, HDC};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateIcon, DestroyIcon, DrawIconEx, DI_NORMAL, HICON,
};

use crate::TrayIcon;

const ICON_SIZE: i32 = 16;
pub const PIXEL_SIZE: usize = 4;

pub struct SafeHICON(pub HICON);

impl Drop for SafeHICON {
    fn drop(&mut self) {
        // Release inner HICON when no longer needed
        if !self.0.is_null() {
            unsafe {
                DestroyIcon(self.0);
            }
        }
    }
}

impl From<TrayIcon> for SafeHICON {
    fn from(tray_icon: TrayIcon) -> Self {
        let rgba = tray_icon.data;
        let pixel_count = rgba.len() / PIXEL_SIZE;
        let mut and_mask = Vec::with_capacity(pixel_count);

        // Convert RGBA to BGRA
        let bgra_data = rgba
            .chunks_exact(PIXEL_SIZE)
            .flat_map(|v| {
                and_mask.push(!v[3]);
                [v[2], v[1], v[0], v[3]]
            })
            .collect::<Vec<_>>();

        let handle = unsafe {
            CreateIcon(
                null_mut(),              // HINSTANCE
                tray_icon.width as i32,  // Width
                tray_icon.height as i32, // Height
                1,                       // The number of planes in the XOR bitmask of the icon
                (PIXEL_SIZE * 8) as u8, // The number of bits-per-pixel in the XOR bitmask of the icon
                and_mask.as_ptr(),      // AND bitmask data
                bgra_data.as_ptr(),     // Image data
            )
        };

        SafeHICON(handle)
    }
}

impl From<TrayIcon> for Option<HBITMAP> {
    fn from(tray_icon: TrayIcon) -> Self {
        unsafe {
            // --- 1. Crear el HICON usando los datos del TrayIcon ---
            let rgba = tray_icon.data;
            let pixel_count = rgba.len() / PIXEL_SIZE;
            // La máscara para el ícono (para CreateIcon, se espera un buffer de bytes; 0 = opaco, 0xFF = transparente)
            let mut and_mask = Vec::with_capacity(pixel_count);
            // Se convierte de RGBA a BGRA (simplemente se reordena) y se llena la máscara:
            // Si el canal alfa es menor a 128, marcamos el píxel como transparente (0xFF);
            // en caso contrario, opaco (0).
            let bgra_data = rgba
                .chunks_exact(PIXEL_SIZE)
                .flat_map(|v| {
                    let threshold = 128;
                    let mask_val = if v[3] < threshold { 0xFF } else { 0 };
                    and_mask.push(mask_val);
                    [v[2], v[1], v[0], v[3]]
                })
                .collect::<Vec<_>>();

            // Crear el HICON; se usa CreateIcon pasando la máscara y la imagen (en BGRA)
            let hicon = CreateIcon(
                null_mut(),              // HINSTANCE (no se requiere)
                tray_icon.width as i32,  // ancho original
                tray_icon.height as i32, // alto original
                1,                       // número de planos
                (PIXEL_SIZE * 8) as u8,  // bits por píxel
                and_mask.as_ptr(),       // datos de la máscara
                bgra_data.as_ptr(),      // datos de la imagen en BGRA
            );
            if hicon.is_null() {
                return None;
            }

            // --- 2. Convertir el HICON en un HBITMAP con el tamaño ICON_SIZE, aplicando la transparencia ---

            // Obtener el DC de la pantalla
            let hdc_screen: HDC = GetDC(null_mut());
            if hdc_screen.is_null() {
                DestroyIcon(hicon);
                return None;
            }

            // Crear un DIBSection de 32 bpp (con alpha) del tamaño deseado
            let mut bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: ICON_SIZE,
                    biHeight: -ICON_SIZE, // negativo para top–down (la primera fila es la superior)
                    biPlanes: 1,
                    biBitCount: 32, // 32 bits: B, G, R, A
                    biCompression: BI_RGB,
                    biSizeImage: 0,
                    biXPelsPerMeter: 0,
                    biYPelsPerMeter: 0,
                    biClrUsed: 0,
                    biClrImportant: 0,
                },
                bmiColors: [std::mem::zeroed()],
            };
            let mut bits_ptr = null_mut();
            let hbitmap_final = CreateDIBSection(
                hdc_screen,
                &mut bmi,
                DIB_RGB_COLORS,
                &mut bits_ptr,
                null_mut(),
                0,
            );
            if hbitmap_final.is_null() || bits_ptr.is_null() {
                ReleaseDC(null_mut(), hdc_screen);
                DestroyIcon(hicon);
                return None;
            }

            // Crear un DC de memoria compatible y seleccionar el DIBSection en éste.
            let hdc_mem = CreateCompatibleDC(hdc_screen);
            if hdc_mem.is_null() {
                DeleteObject(hbitmap_final as _);
                ReleaseDC(null_mut(), hdc_screen);
                DestroyIcon(hicon);
                return None;
            }
            let old_obj = SelectObject(hdc_mem, hbitmap_final as *mut _);

            // Limpiar la superficie del bitmap (llenar con 0, lo que en un DIB de 32bpp es totalmente transparente)
            std::ptr::write_bytes(bits_ptr, 0, (ICON_SIZE * ICON_SIZE * 4) as usize);

            // Dibujar el HICON en el DC de memoria con DrawIconEx; DI_NORMAL respeta la máscara
            let draw_ok = DrawIconEx(
                hdc_mem,   // DC destino
                0,         // x
                0,         // y
                hicon,     // el HICON a dibujar
                ICON_SIZE, // ancho a dibujar
                ICON_SIZE, // alto a dibujar
                0,
                null_mut(),
                DI_NORMAL, // bandera para que se aplique la máscara
            );

            // Restaurar y limpiar el DC de memoria
            SelectObject(hdc_mem, old_obj);
            DeleteDC(hdc_mem);
            ReleaseDC(null_mut(), hdc_screen);
            DestroyIcon(hicon);

            if draw_ok == 0 {
                // Si falla el dibujo, limpiar y retornar None.
                DeleteObject(hbitmap_final as _);
                return None;
            }

            // En este punto, hbitmap_final es un HBITMAP de 32bpp que contiene el dibujo
            // del ícono con su transparencia aplicada (las áreas "transparentes" deberían tener 0 en el canal alfa).
            Some(hbitmap_final)
        }
    }
}
