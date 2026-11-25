//! The render node the VA-API decoders open: the configuration's GPU
//! preference, else the first one that opens.

use std::path::Path;
use std::rc::Rc;

use cros_libva::Display;
use ferricast_codec::GpuInput as _;
use ferricast_core::{BackendId, DecoderConfig, FerricastError, GpuAffinity, Result};
use tracing::debug;

/// Tried in order when the preference names no render node.
const RENDER_NODES: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
    "/dev/dri/renderD130",
];

/// The render node `config`'s GPU preference puts VA-API decode on (any
/// GPU: NVIDIA's VA-API driver decodes too); `None` to take the first that
/// opens.
///
/// # Errors
/// [`FerricastError::Codec`] (`BackendUnavailable`) when the preference
/// pins a GPU that is missing.
pub(crate) fn render_node(config: &DecoderConfig) -> Result<Option<String>> {
    config
        .gpu_for(BackendId::Vaapi, GpuAffinity::Any)
        .map(|gpu| gpu.map(|gpu| gpu.render_node))
        .map_err(FerricastError::Codec)
}

/// Opens `node`, or without one the first render node that opens.
pub(crate) fn open_display(node: Option<&str>) -> Option<Rc<Display>> {
    if let Some(node) = node {
        return match Display::open_drm_display(Path::new(node)) {
            Ok(display) => {
                debug!(node, "opened VA-API DRM display");
                Some(display)
            }
            Err(e) => {
                debug!(node, error = %e, "preferred render node does not open");
                None
            }
        };
    }
    RENDER_NODES
        .iter()
        .filter(|node| Path::new(node).exists())
        .find_map(|node| match Display::open_drm_display(Path::new(node)) {
            Ok(display) => {
                debug!(node, "opened VA-API DRM display");
                Some(display)
            }
            Err(e) => {
                debug!(node, error = %e, "render node open failed");
                None
            }
        })
        .or_else(Display::open)
}
