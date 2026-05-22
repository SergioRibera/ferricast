//! D-Bus client implementations of the `list`, `stream`, `stop`
//! and `introspect` subcommands.
//!
//! All of them talk to the daemon over the session bus via the
//! strongly-typed [`ManagerProxy`] from `ferricast-dbus`. None of
//! them touch capture, encoders or protocol code — that's all behind
//! the daemon.

use ferricast_dbus::{ManagerProxy, SourceDto, BUS_NAME};
use futures_util::stream::StreamExt;

use crate::cli::SourceKind;

/// Map `zbus::Error` for the "no daemon owns the bus name" case into
/// a friendly message. Everything else passes through unchanged.
fn rewrap(err: zbus::Error) -> anyhow::Error {
    use zbus::fdo::Error as FdoError;
    if let zbus::Error::FDO(boxed) = &err {
        if let FdoError::ServiceUnknown(_) | FdoError::NameHasNoOwner(_) = boxed.as_ref() {
            return anyhow::anyhow!(
                "no ferricast daemon is running on the session bus \
                 (looked for `{BUS_NAME}`). Start one with \
                 `ferricast-gui --background` and try again."
            );
        }
    }
    anyhow::Error::new(err)
}

pub(crate) async fn proxy() -> anyhow::Result<ManagerProxy<'static>> {
    let conn = zbus::Connection::session().await.map_err(rewrap)?;
    ManagerProxy::new(&conn).await.map_err(rewrap)
}

pub async fn list(watch: bool) -> anyhow::Result<()> {
    let p = proxy().await?;

    for d in p.list_devices().await.map_err(rewrap)? {
        print_device_line(&d.id, &d.name, &d.protocol, &d.model, &d.host);
    }

    if !watch {
        return Ok(());
    }

    let mut added = p.receive_device_added().await.map_err(rewrap)?;
    let mut removed = p.receive_device_removed().await.map_err(rewrap)?;
    println!("# watching for changes (Ctrl-C to stop)");
    loop {
        tokio::select! {
            ev = added.next() => {
                if let Some(sig) = ev {
                    if let Ok(args) = sig.args() {
                        let d = args.device();
                        print!("+ ");
                        print_device_line(&d.id, &d.name, &d.protocol, &d.model, &d.host);
                    }
                } else { break; }
            }
            ev = removed.next() => {
                if let Some(sig) = ev {
                    if let Ok(args) = sig.args() {
                        println!("- {}", args.device_id());
                    }
                } else { break; }
            }
        }
    }
    Ok(())
}

fn print_device_line(id: &str, name: &str, protocol: &str, model: &str, host: &str) {
    let model_field = if model.is_empty() { "-".into() } else { model.to_string() };
    let host_field = if host.is_empty() { "-".into() } else { host.to_string() };
    println!("{id}\t{name}\t{protocol}\t{model_field}\t{host_field}");
}

pub async fn stream(device: String, source: Option<SourceKind>) -> anyhow::Result<()> {
    let src = match source {
        None => SourceDto::auto(),
        Some(SourceKind::Screen) => SourceDto::screen(),
        Some(SourceKind::Window) => SourceDto::window(),
    };
    let p = proxy().await?;
    p.start_stream(&device, src).await.map_err(rewrap)?;
    println!("ok: streaming requested for {device}");
    Ok(())
}

pub async fn stop(device: String) -> anyhow::Result<()> {
    let p = proxy().await?;
    p.stop_stream(&device).await.map_err(rewrap)?;
    println!("ok: stop requested for {device}");
    Ok(())
}

/// Print the embedded introspection XML to stdout.
pub fn introspect() {
    print!("{}", ferricast_dbus::INTROSPECTION_XML);
}

pub async fn thumb(
    kind: crate::cli::ThumbKind,
    id: String,
    max_w: u32,
    max_h: u32,
    output: Option<std::path::PathBuf>,
) -> anyhow::Result<()> {
    use std::io::Write;
    let p = proxy().await?;
    let bytes = match kind {
        crate::cli::ThumbKind::Monitor => p.get_monitor_thumbnail(&id, max_w, max_h).await,
        crate::cli::ThumbKind::Window => p.get_window_thumbnail(&id, max_w, max_h).await,
    }
    .map_err(rewrap)?;
    if bytes.is_empty() {
        anyhow::bail!(
            "daemon returned an empty thumbnail — likely missing protocol (e.g. \
             ext-image-copy-capture for window thumbnails on this compositor)"
        );
    }
    match output {
        Some(path) => {
            std::fs::write(&path, &bytes)?;
            eprintln!("{} bytes → {}", bytes.len(), path.display());
        }
        None => {
            std::io::stdout().write_all(&bytes)?;
        }
    }
    Ok(())
}

pub async fn monitors(watch: bool) -> anyhow::Result<()> {
    let p = proxy().await?;
    print_monitors(&p).await?;
    if !watch {
        return Ok(());
    }
    let mut stream = p.receive_monitors_changed().await.map_err(rewrap)?;
    eprintln!("# watching MonitorsChanged (Ctrl-C to stop)");
    while stream.next().await.is_some() {
        println!("---");
        print_monitors(&p).await?;
    }
    Ok(())
}

async fn print_monitors(p: &ManagerProxy<'_>) -> anyhow::Result<()> {
    let mons = p.list_monitors().await.map_err(rewrap)?;
    for m in mons {
        let make_model = match (m.make.as_str(), m.model.as_str()) {
            ("", "") => "-".to_string(),
            (mk, "") => mk.to_string(),
            ("", md) => md.to_string(),
            (mk, md) => format!("{mk} {md}"),
        };
        let refresh = if m.refresh_mhz > 0 {
            format!("{:.3}Hz", m.refresh_mhz as f64 / 1000.0)
        } else {
            "-".into()
        };
        let primary = if m.primary { " *" } else { "" };
        println!(
            "{id}\t{name}\t{make_model}\t{w}x{h}+{x},{y}@{scale:.2}\t{refresh}{primary}",
            id = m.id,
            name = m.name,
            w = m.width,
            h = m.height,
            x = m.x,
            y = m.y,
            scale = m.scale,
        );
    }
    Ok(())
}

pub async fn windows(watch: bool) -> anyhow::Result<()> {
    let p = proxy().await?;
    print_windows(&p).await?;
    if !watch {
        return Ok(());
    }
    let mut stream = p.receive_windows_changed().await.map_err(rewrap)?;
    eprintln!("# watching WindowsChanged (Ctrl-C to stop)");
    while stream.next().await.is_some() {
        println!("---");
        print_windows(&p).await?;
    }
    Ok(())
}

async fn print_windows(p: &ManagerProxy<'_>) -> anyhow::Result<()> {
    let ws = p.list_windows().await.map_err(rewrap)?;
    for w in ws {
        let app = if w.app_id.is_empty() { "-".into() } else { w.app_id };
        let pid = if w.pid == 0 { "-".into() } else { w.pid.to_string() };
        let geo = if w.has_geometry {
            format!("{}x{}+{},{}", w.width, w.height, w.x, w.y)
        } else {
            "-".into()
        };
        let mon = if w.on_monitor.is_empty() { "-".into() } else { w.on_monitor };
        println!(
            "{id}\t{app}\t{pid}\t{geo}\t{mon}\t{title}",
            id = w.id,
            title = w.title,
        );
    }
    Ok(())
}
