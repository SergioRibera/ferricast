//! Compile-only smoke tests. The goal is to catch API mismatches
//! with zbus 5 without having to spin up a real session bus — every
//! pattern used by the daemon and the CLI client is reproduced here
//! against the same proxy macro that ships in the crate.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use ferricast_dbus::{
    ActiveStreamDto, DeviceDto, ManagerProxy, MonitorInfoDto, SourceDto, WindowInfoDto, BUS_NAME,
    INTERFACE, INTROSPECTION_XML, OBJECT_PATH,
};
use tokio::sync::Mutex;
use zbus::object_server::{InterfaceRef, SignalEmitter};
use zbus::zvariant::OwnedValue;

#[test]
fn constants_present() {
    assert!(BUS_NAME.starts_with("rs."));
    assert!(OBJECT_PATH.starts_with('/'));
    assert!(INTERFACE.ends_with(".Manager1"));
    assert!(INTROSPECTION_XML.contains("<interface name=\"rs.sergioribera.ferricast.Manager1\">"));
}

#[test]
fn source_helpers() {
    assert_eq!(SourceDto::auto().kind, "");
    assert_eq!(SourceDto::screen().kind, "screen");
    assert_eq!(SourceDto::window().kind, "window");

    let m = SourceDto::monitor("HDMI-1");
    assert_eq!(m.kind, "monitor");
    assert!(m.args.contains_key("id"));

    let w = SourceDto::window_by_id("12345");
    assert_eq!(w.kind, "window");
    assert!(w.args.contains_key("id"));

    let w = SourceDto::window_by_title("Firefox");
    assert_eq!(w.kind, "window");
    assert!(w.args.contains_key("title"));
}

/// Replica of the daemon-side interface object. We don't need the
/// real `StreamManager` here — only the zbus surface — so we stub
/// it with an empty struct and confirm every method/signal compiles
/// against zbus 5.
struct Stub;

#[zbus::interface(name = "rs.sergioribera.ferricast.Manager1")]
impl Stub {
    async fn list_devices(&self) -> zbus::fdo::Result<Vec<DeviceDto>> {
        Ok(Vec::new())
    }
    async fn list_active_streams(&self) -> zbus::fdo::Result<Vec<ActiveStreamDto>> {
        Ok(Vec::new())
    }
    async fn start_stream(&self, _id: String, _src: SourceDto) -> zbus::fdo::Result<()> {
        Ok(())
    }
    async fn stop_stream(&self, _id: String) -> zbus::fdo::Result<()> {
        Ok(())
    }

    async fn list_monitors(&self) -> zbus::fdo::Result<Vec<MonitorInfoDto>> {
        Ok(Vec::new())
    }
    async fn list_windows(&self) -> zbus::fdo::Result<Vec<WindowInfoDto>> {
        Ok(Vec::new())
    }
    async fn get_monitor_thumbnail(
        &self,
        _id: String,
        _max_w: u32,
        _max_h: u32,
    ) -> zbus::fdo::Result<Vec<u8>> {
        Ok(Vec::new())
    }
    async fn get_window_thumbnail(
        &self,
        _id: String,
        _max_w: u32,
        _max_h: u32,
    ) -> zbus::fdo::Result<Vec<u8>> {
        Ok(Vec::new())
    }

    #[zbus(property)]
    async fn protocols(&self) -> Vec<String> {
        Vec::new()
    }
    #[zbus(property)]
    async fn enumeration_capabilities(&self) -> Vec<String> {
        Vec::new()
    }

    #[zbus(signal)]
    async fn device_added(emitter: &SignalEmitter<'_>, device: DeviceDto) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn device_removed(emitter: &SignalEmitter<'_>, device_id: String) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn stream_started(
        emitter: &SignalEmitter<'_>,
        device_id: String,
        device_name: String,
    ) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn stream_stopped(emitter: &SignalEmitter<'_>, device_id: String) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn stream_reconnecting(
        emitter: &SignalEmitter<'_>,
        device_id: String,
        attempt: u32,
        reason: String,
    ) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn stream_error(
        emitter: &SignalEmitter<'_>,
        device_id: String,
        message: String,
    ) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn discovery_error(
        emitter: &SignalEmitter<'_>,
        protocol: String,
        message: String,
    ) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn monitors_changed(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;
    #[zbus(signal)]
    async fn windows_changed(emitter: &SignalEmitter<'_>) -> zbus::Result<()>;
}

/// Type-level confirmation: the patterns the daemon uses to wire up
/// the bus and to emit signals from a background task all compile.
/// Never invoked at runtime — no bus is started — but rustc still
/// type-checks the whole thing.
async fn _daemon_shape() -> zbus::Result<()> {
    let _shared = Arc::new(Mutex::new(()));
    let conn = zbus::connection::Builder::session()?
        .name(BUS_NAME)?
        .serve_at(OBJECT_PATH, Stub)?
        .build()
        .await?;

    let iface_ref: InterfaceRef<Stub> = conn
        .object_server()
        .interface::<_, Stub>(OBJECT_PATH)
        .await?;
    let emitter = iface_ref.signal_emitter();

    let dto = DeviceDto {
        id: "id".into(),
        name: "n".into(),
        protocol: "p".into(),
        model: String::new(),
        host: String::new(),
        capabilities: HashMap::<String, OwnedValue>::new(),
    };
    Stub::device_added(emitter, dto).await?;
    Stub::device_removed(emitter, "id".into()).await?;
    Stub::stream_started(emitter, "id".into(), "name".into()).await?;
    Stub::stream_stopped(emitter, "id".into()).await?;
    Stub::stream_reconnecting(emitter, "id".into(), 1, "why".into()).await?;
    Stub::stream_error(emitter, "id".into(), "msg".into()).await?;
    Stub::discovery_error(emitter, "proto".into(), "msg".into()).await?;
    Stub::monitors_changed(emitter).await?;
    Stub::windows_changed(emitter).await?;
    Ok(())
}

/// Same for the client side — every call the CLI makes goes through
/// here so a breaking change in `ManagerProxy` lights up at
/// `cargo test --no-run`, not at runtime.
async fn _client_shape() -> zbus::Result<()> {
    use futures_util::stream::StreamExt;
    let conn = zbus::Connection::session().await?;
    let proxy = ManagerProxy::new(&conn).await?;

    let _: Vec<DeviceDto> = proxy.list_devices().await?;
    let _: Vec<ActiveStreamDto> = proxy.list_active_streams().await?;
    proxy.start_stream("x", SourceDto::screen()).await?;
    proxy.stop_stream("x").await?;
    let _: Vec<String> = proxy.protocols().await?;
    let _: Vec<MonitorInfoDto> = proxy.list_monitors().await?;
    let _: Vec<WindowInfoDto> = proxy.list_windows().await?;
    let _: Vec<String> = proxy.enumeration_capabilities().await?;
    let _: Vec<u8> = proxy.get_monitor_thumbnail("HDMI-1", 320, 180).await?;
    let _: Vec<u8> = proxy.get_window_thumbnail("12345", 320, 180).await?;

    let mut added = proxy.receive_device_added().await?;
    if let Some(sig) = added.next().await {
        let args = sig.args()?;
        let _: &DeviceDto = args.device();
    }
    let mut removed = proxy.receive_device_removed().await?;
    if let Some(sig) = removed.next().await {
        let args = sig.args()?;
        let _id: &str = args.device_id();
    }
    // The argument-less change signals don't carry args, so we just
    // confirm the receiver future compiles and yields the right
    // signal message type.
    let mut mc = proxy.receive_monitors_changed().await?;
    let _ = mc.next().await;
    let mut wc = proxy.receive_windows_changed().await?;
    let _ = wc.next().await;
    Ok(())
}
