use thiserror::Error;

pub type Result<T> = std::result::Result<T, Tray>;

#[derive(Error, Debug)]
pub enum Tray {
    #[error("{0}")]
    Custom(&'static str),

    #[error("{0}")]
    IconLoad(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "png")]
    #[cfg_attr(feature = "png", error("PNG error: {0}"))]
    PNG(#[from] image::ImageError),

    #[cfg(feature = "svg")]
    #[cfg_attr(feature = "svg", error("SVG error: {0}"))]
    SVG(#[from] usvg::Error),
}
