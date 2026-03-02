use crate::TrayIcon;
use cocoa::base::{id, nil};
use cocoa::foundation::NSString;
use objc::{class, msg_send, sel, sel_impl};

pub fn create_sf_symbol_icon(symbol_name: &str, point_size: f64) -> Option<TrayIcon> {
    unsafe {
        let ns_name = NSString::alloc(nil).init_str(symbol_name);
        let image: id = msg_send![class!(NSImage), imageWithSystemSymbolName:ns_name accessibilityDescription:nil];

        if image.is_null() {
            return None;
        }

        let config: id = msg_send![
            class!(NSImageSymbolConfiguration),
            configurationWithPointSize:point_size
            weight:0.0
            scale:1i64
        ];

        let configured_image: id = msg_send![image, imageWithSymbolConfiguration:config];
        let final_image = if !configured_image.is_null() {
            configured_image
        } else {
            image
        };

        let size: (f64, f64) = msg_send![final_image, size];
        let width = size.0.ceil() as u32;
        let height = size.1.ceil() as u32;

        if width == 0 || height == 0 {
            return None;
        }

        let tiff_data: id = msg_send![final_image, TIFFRepresentation];
        if tiff_data.is_null() {
            return None;
        }

        let bitmap: id = msg_send![class!(NSBitmapImageRep), imageRepWithData:tiff_data];
        if bitmap.is_null() {
            return None;
        }

        let bmp_width: isize = msg_send![bitmap, pixelsWide];
        let bmp_height: isize = msg_send![bitmap, pixelsHigh];
        let bits_per_pixel: isize = msg_send![bitmap, bitsPerPixel];

        if bmp_width <= 0 || bmp_height <= 0 || bits_per_pixel != 32 {
            return None;
        }

        let bitmap_data: *mut u8 = msg_send![bitmap, bitmapData];
        if bitmap_data.is_null() {
            return None;
        }

        let bytes_per_row: isize = msg_send![bitmap, bytesPerRow];
        let data_len = (bmp_height * bytes_per_row) as usize;
        let data = std::slice::from_raw_parts(bitmap_data, data_len).to_vec();

        let tiff_length: usize = msg_send![tiff_data, length];
        let tiff_bytes: *const u8 = msg_send![tiff_data, bytes];
        let data_raw = if !tiff_bytes.is_null() && tiff_length > 0 {
            std::slice::from_raw_parts(tiff_bytes, tiff_length).to_vec()
        } else {
            data.clone()
        };

        Some(TrayIcon {
            data: data.into(),
            data_raw: data_raw.into(),
            width: bmp_width as u32,
            height: bmp_height as u32,
            is_template: true,
        })
    }
}
