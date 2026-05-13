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

async fn proxy() -> anyhow::Result<ManagerProxy<'static>> {
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
