use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct TrayIcon {
    pub data: Arc<[u8]>,
    pub data_raw: Arc<[u8]>,
    pub width: u32,
    pub height: u32,
    pub is_template: bool,
}

impl TrayIcon {
    /// Creates a static icon from a byte slice.
    pub fn from_static<Data: Into<Vec<u8>>>(width: u32, height: u32, data: Data) -> Self {
        let data = data.into();
        let data: Arc<[u8]> = data.into();

        TrayIcon {
            width,
            height,
            data_raw: data.clone().into(),
            data,
            is_template: true,
        }
    }

    /// Loads an icon dynamically from a bytes.
    #[cfg(feature = "png")]
    pub fn from_png<Data: AsRef<[u8]>>(data: Data) -> crate::error::Result<Self> {
        use image::GenericImageView;
        let data = data.as_ref();
        let data_raw: Arc<[u8]> = data.into();

        let img = image::load_from_memory_with_format(data, image::ImageFormat::Png)?;
        let (width, height) = img.dimensions();

        let data = img.into_rgba8().into_vec().into();

        Ok(Self {
            data,
            width,
            height,
            data_raw,
            is_template: true,
        })
    }

    /// Loads an icon dynamically from a file path at runtime.
    #[cfg(feature = "png")]
    pub fn png_from_path<P: AsRef<std::path::Path>>(path: P) -> crate::error::Result<Self> {
        use image::GenericImageView;

        let input_file = std::fs::File::open(path)?;
        let buff = std::io::BufReader::new(input_file);
        let data_raw = buff.buffer().into();
        let img = image::load_from_memory_with_format(buff.buffer(), image::ImageFormat::Png)?;
        let (width, height) = img.dimensions();

        let data = img.into_rgba8().into_vec().into();

        Ok(Self {
            data,
            width,
            height,
            data_raw,
            is_template: true,
        })
    }

    pub fn with_template(mut self, is_template: bool) -> Self {
        self.is_template = is_template;
        self
    }

    #[cfg(feature = "svg")]
    pub fn from_svg<Data: AsRef<[u8]>>(svg_data: Data, size: u32) -> crate::error::Result<Self> {
        let svg_data = svg_data.as_ref();
        let data_raw: Arc<[u8]> = svg_data.into();

        let opt = usvg::Options::default();
        let tree = usvg::Tree::from_data(svg_data, &opt)?;

        let pixmap_size = tree.size().to_int_size();
        let scale = (size as f32) / pixmap_size.width().max(pixmap_size.height()) as f32;

        let width = (pixmap_size.width() as f32 * scale).ceil() as u32;
        let height = (pixmap_size.height() as f32 * scale).ceil() as u32;

        let mut pixmap = tiny_skia::Pixmap::new(width, height)
            .ok_or_else(|| crate::error::Tray::IconLoad("Failed to create pixmap".into()))?;

        let transform = tiny_skia::Transform::from_scale(scale, scale);
        resvg::render(&tree, transform, &mut pixmap.as_mut());

        let data: Arc<[u8]> = pixmap.data().to_vec().into();

        Ok(Self {
            data,
            width,
            height,
            data_raw,
            is_template: true,
        })
    }

    #[cfg(feature = "svg")]
    pub fn svg_from_path<P: AsRef<std::path::Path>>(
        path: P,
        size: u32,
    ) -> crate::error::Result<Self> {
        let svg_data = std::fs::read(path)?;
        Self::from_svg(svg_data, size)
    }
}
