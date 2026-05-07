//! xdg-desktop-portal ScreenCast handshake.
//!
//! Hands back a [`PortalStream`] containing the global PipeWire `node_id`
//! and size hints. Unlike the `OpenPipeWireRemote` flow, we do NOT take
//! the portal's private fd — once the portal has authorized the source
//! the node is also visible on the user's regular PipeWire daemon, so
//! we can `Context::connect(None)` from `stream.rs`. Avoiding the fd
//! sidesteps the variations in how each portal implementation
//! (gnome / wlroots / kwin) handles the private socket — which is
//! what wlx-capture does and what was needed to make capture work
//! reliably on Niri+portal-gnome.

use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
use ashpd::desktop::PersistMode;
use ferricast_core::{CaptureConfig, CaptureSource, FerricastError, Result};
use tracing::{debug, info};

/// Result of the portal handshake — everything `stream.rs` needs.
pub(super) struct PortalStream {
    /// Global node id of the screencast source on the user's PW graph.
    pub node_id: u32,
    /// Hint of the source size, when the portal reports it.
    pub size_hint: Option<(u32, u32)>,
    /// Hint of the source position on the virtual desktop. Currently
    /// unused; kept here so the future window-cropping path can read it
    /// without re-doing the portal handshake.
    #[allow(dead_code)]
    pub position_hint: Option<(i32, i32)>,
}

pub(super) async fn open_session(
    source: &CaptureSource,
    config: &CaptureConfig,
) -> Result<PortalStream> {
    let proxy = Screencast::new().await.map_err(portal_err)?;
    debug!("portal proxy created");

    let session = proxy.create_session().await.map_err(portal_err)?;
    debug!("portal session created");

    let source_type = match source {
        CaptureSource::FullScreen { .. } => SourceType::Monitor,
        CaptureSource::Window { .. } => SourceType::Window,
    };

    let cursor_mode = if config.show_cursor {
        CursorMode::Embedded
    } else {
        CursorMode::Hidden
    };

    proxy
        .select_sources(
            &session,
            cursor_mode,
            source_type.into(),
            false,
            None,
            PersistMode::DoNot,
        )
        .await
        .map_err(portal_err)?;

    info!(?source_type, ?cursor_mode, "portal sources selected, starting cast");

    let response = proxy
        .start(&session, None)
        .await
        .map_err(portal_err)?
        .response()
        .map_err(portal_err)?;

    let stream = response
        .streams()
        .first()
        .ok_or_else(|| FerricastError::Capture("portal returned no streams".into()))?;

    let portal_stream = PortalStream {
        node_id: stream.pipe_wire_node_id(),
        size_hint: stream.size().map(|(w, h)| (w as u32, h as u32)),
        position_hint: stream.position(),
    };

    info!(
        node_id = portal_stream.node_id,
        size = ?portal_stream.size_hint,
        "portal handshake complete"
    );

    // ashpd uses a process-wide dbus connection internally, so dropping
    // `proxy` / `session` here only drops the Rust handles — the
    // Mutter-side session keeps running as long as the dbus connection
    // stays open (which is forever, in practice).
    Ok(portal_stream)
}

fn portal_err(e: impl std::fmt::Display) -> FerricastError {
    FerricastError::Capture(format!("xdg-portal: {e}"))
}
