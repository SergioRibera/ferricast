//! The few header fields a decoder that is handed bare packets must spell
//! out itself: VideoToolbox wants a format description (bit depth, frame
//! size, the VP9 `vpcC` / AV1 `av1C` records) before it decodes anything.

pub(crate) mod av1;
pub(crate) mod bits;
pub(crate) mod h26x;
pub(crate) mod vp9;
