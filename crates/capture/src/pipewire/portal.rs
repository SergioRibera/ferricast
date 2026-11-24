//! xdg-desktop-portal `ScreenCast` handshake.
//!
//! Hands back a [`PortalStream`] containing the global `PipeWire` `node_id`
//! and size hints. Unlike the `OpenPipeWireRemote` flow, we do NOT take
//! the portal's private fd — once the portal has authorized the source
//! the node is also visible on the user's regular `PipeWire` daemon, so
//! we can `Context::connect(None)` from `stream.rs`. Avoiding the fd
//! sidesteps the variations in how each portal implementation
//! (gnome / wlroots / kwin) handles the private socket — which is
//! what wlx-capture does and what was needed to make capture work
//! reliably on Niri+portal-gnome.

use std::sync::Arc;

use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
use ashpd::desktop::{PersistMode, Session};
use ferricast_core::{CaptureConfig, CaptureSource, FerricastError, Result};
use futures_util::StreamExt;
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
    #[allow(
        dead_code,
        reason = "kept for restoring a stream without re-doing the portal handshake"
    )]
    pub position_hint: Option<(i32, i32)>,
}

/// Live xdg-desktop-portal `ScreenCast` session. The compositor keeps
/// the cast (and its "sharing" indicator) alive until the session is
/// closed, so the owner must call [`PortalSession::close`] on stop;
/// dropping it only drops the D-Bus proxy.
#[derive(Clone)]
pub(super) struct PortalSession(Arc<Session<'static, Screencast<'static>>>);

impl PortalSession {
    /// Close the portal session, ending the compositor-side cast.
    pub(super) async fn close(&self) {
        match self.0.close().await {
            Ok(()) => debug!("portal session closed"),
            // Already closed by the compositor / user: nothing to do.
            Err(e) => debug!(error = %e, "portal session close failed"),
        }
    }

    /// Resolve once the portal emits `Closed` (user hit "Stop
    /// sharing", compositor revoked the cast). Returns immediately
    /// with an error if the signal can't be subscribed to.
    pub(super) async fn closed(&self) -> Result<()> {
        let mut signal = std::pin::pin!(self.0.receive_closed().await.map_err(portal_err)?);
        signal.next().await;
        Ok(())
    }
}

pub(super) async fn open_session(
    source: &CaptureSource,
    config: &CaptureConfig,
) -> Result<(PortalStream, PortalSession)> {
    let proxy = Screencast::new().await.map_err(portal_err)?;
    debug!("portal proxy created");

    let session = PortalSession(Arc::new(proxy.create_session().await.map_err(portal_err)?));
    debug!("portal session created");

    // Close the session on any handshake failure so a half-set-up
    // cast doesn't linger on the compositor side.
    match negotiate(&proxy, &session, source, config).await {
        Ok(stream) => Ok((stream, session)),
        Err(e) => {
            session.close().await;
            Err(e)
        }
    }
}

async fn negotiate(
    proxy: &Screencast<'static>,
    session: &PortalSession,
    source: &CaptureSource,
    config: &CaptureConfig,
) -> Result<PortalStream> {
    let session = &*session.0;

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
            session,
            cursor_mode,
            source_type.into(),
            false,
            None,
            PersistMode::DoNot,
        )
        .await
        .map_err(portal_err)?;

    info!(
        ?source_type,
        ?cursor_mode,
        "portal sources selected, starting cast"
    );

    let response = proxy
        .start(session, None)
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
        size_hint: stream
            .size()
            .map(|(w, h)| (w.cast_unsigned(), h.cast_unsigned())),
        position_hint: stream.position(),
    };

    info!(
        node_id = portal_stream.node_id,
        size = ?portal_stream.size_hint,
        "portal handshake complete"
    );

    Ok(portal_stream)
}

fn portal_err(e: impl std::fmt::Display) -> FerricastError {
    FerricastError::Capture(format!("xdg-portal: {e}"))
}
