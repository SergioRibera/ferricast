//! Which source and backend a ferricast capture asks the plugin for.

use ferricast_core::{CaptureSource, FerricastError, Result, WindowIdentifier};
use istmo_screen_capture::{
    AutoPreference, AutoSelect, BackendId, BackendSelection, PickerOptions, Source, SourceCapture,
    SourceKind, SourceSelector,
};
use tracing::warn;

/// The environment variable that pins the capture backend
/// (`wlr-screencopy`, `ext-image-copy`, `x11`, `portal`; `wayland-direct`
/// and `pipewire` are the older names of the Wayland and portal paths).
pub const BACKEND_ENV_VAR: &str = "FERRICAST_CAPTURE";

/// What to open: a source on a backend, or a picker of one backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Target {
    /// Captured without asking the user.
    Direct {
        source: SourceSelector,
        backend: BackendSelection,
    },
    /// The user picks a source of `kind` in `backend`'s dialog.
    Picker {
        kind: SourceKind,
        backend: BackendId,
    },
}

impl Target {
    /// The target for `source`, resolved against `listed` (the sources of
    /// its kind the plugin lists here), with the backend `pinned` by
    /// [`BACKEND_ENV_VAR`] when set.
    ///
    /// # Errors
    /// A monitor id that names no listed monitor.
    pub(crate) fn resolve(
        source: &CaptureSource,
        listed: &[Source],
        pinned: Option<BackendSelection>,
    ) -> Result<Self> {
        let backend = pinned.unwrap_or_else(|| {
            BackendSelection::Auto(AutoSelect {
                prefer: AutoPreference::ZeroCopy,
                exclude: Vec::new(),
            })
        });
        if backend == BackendSelection::Fixed(BackendId::LinuxPortal) {
            let kind = match source {
                CaptureSource::FullScreen { .. } => SourceKind::Monitor,
                CaptureSource::Window { .. } => SourceKind::Window,
            };
            return Ok(Self::portal(kind));
        }
        match source {
            CaptureSource::FullScreen { monitor: None } => Ok(Self::Direct {
                source: SourceSelector::PrimaryMonitor,
                backend,
            }),
            CaptureSource::FullScreen { monitor: Some(id) } => {
                let monitor = listed
                    .iter()
                    .filter(|s| s.kind == SourceKind::Monitor)
                    .find(|s| names(s, id))
                    .ok_or_else(|| {
                        let known: Vec<&str> = listed
                            .iter()
                            .filter(|s| s.kind == SourceKind::Monitor)
                            .map(|s| s.connector.as_deref().unwrap_or(&s.id.value))
                            .collect();
                        FerricastError::Capture(format!(
                            "capture: no monitor {id:?} (monitors here: {known:?})"
                        ))
                    })?;
                Ok(Self::Direct {
                    source: SourceSelector::Id(monitor.id.clone()),
                    backend,
                })
            }
            CaptureSource::Window { identifier: None } => Ok(Self::portal(SourceKind::Window)),
            CaptureSource::Window {
                identifier: Some(identifier),
            } => {
                let window = listed
                    .iter()
                    .filter(|s| s.kind == SourceKind::Window)
                    .find(|s| identifies(identifier, s))
                    .ok_or_else(|| {
                        FerricastError::Capture(format!("capture: no window {identifier:?} here"))
                    })?;
                Ok(match &window.capture {
                    SourceCapture::ById => Self::Direct {
                        source: SourceSelector::Id(window.id.clone()),
                        backend,
                    },
                    // Listed from the compositor's window list only: the
                    // picker of a backend that can capture it.
                    SourceCapture::PickerOnly(backends) => Self::Picker {
                        kind: SourceKind::Window,
                        backend: backends.first().copied().unwrap_or(BackendId::LinuxPortal),
                    },
                })
            }
        }
    }

    /// The xdg-desktop-portal dialog for a `kind`.
    pub(crate) const fn portal(kind: SourceKind) -> Self {
        Self::Picker {
            kind,
            backend: BackendId::LinuxPortal,
        }
    }

    /// The selector and backend this target opens with.
    pub(crate) fn selection(&self) -> (SourceSelector, BackendSelection) {
        match self {
            Self::Direct { source, backend } => (source.clone(), backend.clone()),
            Self::Picker { kind, backend } => (
                SourceSelector::Picker(PickerOptions {
                    kinds: vec![*kind],
                    multiple: false,
                }),
                BackendSelection::Fixed(*backend),
            ),
        }
    }

    pub(crate) const fn is_picker(&self) -> bool {
        matches!(self, Self::Picker { .. })
    }
}

/// Whether `identifier` names the window `source`. A source id names it
/// exactly; a title names the first window with that title.
fn identifies(identifier: &WindowIdentifier, source: &Source) -> bool {
    match identifier {
        WindowIdentifier::Source(id) => source.id.value == *id,
        // X11 lists windows as `window:0x…`.
        WindowIdentifier::Id(xid) => {
            source.id.value == xid.to_string() || source.id.value == format!("window:{xid:#x}")
        }
        WindowIdentifier::Title(title) => source.name == *title,
    }
}

/// Whether `id` (a ferricast monitor id: the connector, the plugin's id
/// or the name) names `source`.
fn names(source: &Source, id: &str) -> bool {
    source.id.value == id || source.connector.as_deref() == Some(id) || source.name == id
}

/// The backend [`BACKEND_ENV_VAR`] pins, if any.
pub(crate) fn pinned_backend() -> Option<BackendSelection> {
    let value = std::env::var(BACKEND_ENV_VAR).ok()?;
    let fixed = |id| Some(BackendSelection::Fixed(id));
    match value.as_str() {
        "" => None,
        "wayland-direct" | "wayland" => Some(BackendSelection::Ordered(vec![
            BackendId::LinuxExtImageCopy,
            BackendId::LinuxWlrScreencopy,
        ])),
        "wlr-screencopy" => fixed(BackendId::LinuxWlrScreencopy),
        "ext-image-copy" => fixed(BackendId::LinuxExtImageCopy),
        "x11" => fixed(BackendId::LinuxX11),
        "pipewire" | "portal" => fixed(BackendId::LinuxPortal),
        other => {
            warn!(
                "{BACKEND_ENV_VAR}={other:?} names no capture backend \
                 (wlr-screencopy, ext-image-copy, x11, portal); choosing automatically"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use istmo_screen_capture::SourceId;

    use super::*;

    fn monitor(value: &str, connector: &str) -> Source {
        let mut source = Source::new(
            SourceId::new(BackendId::LinuxWlrScreencopy, value),
            SourceKind::Monitor,
            "Some Panel",
        );
        source.connector = Some(connector.into());
        source
    }

    fn window(value: &str, title: &str, capture: SourceCapture) -> Source {
        let mut source = Source::new(
            SourceId::new(BackendId::LinuxWlrScreencopy, value),
            SourceKind::Window,
            title,
        );
        source.capture = capture;
        source
    }

    fn auto() -> BackendSelection {
        BackendSelection::Auto(AutoSelect {
            prefer: AutoPreference::ZeroCopy,
            exclude: Vec::new(),
        })
    }

    #[test]
    fn a_monitor_is_found_by_connector_or_id() {
        let listed = [monitor("17", "DP-3"), monitor("18", "HDMI-A-1")];
        for id in ["HDMI-A-1", "18"] {
            let target = Target::resolve(
                &CaptureSource::FullScreen {
                    monitor: Some(id.into()),
                },
                &listed,
                None,
            )
            .unwrap();
            assert_eq!(
                target,
                Target::Direct {
                    source: SourceSelector::Id(listed[1].id.clone()),
                    backend: auto(),
                }
            );
        }
    }

    #[test]
    fn an_unknown_monitor_is_an_error_naming_the_known_ones() {
        let error = Target::resolve(
            &CaptureSource::FullScreen {
                monitor: Some("DP-9".into()),
            },
            &[monitor("17", "DP-3")],
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("DP-3"), "{error}");
    }

    #[test]
    fn no_monitor_is_the_primary_one_without_a_dialog() {
        let target =
            Target::resolve(&CaptureSource::FullScreen { monitor: None }, &[], None).unwrap();
        assert!(!target.is_picker());
        assert_eq!(target.selection().0, SourceSelector::PrimaryMonitor);
    }

    #[test]
    fn windows_listed_only_go_through_their_picker() {
        let listed = [
            window(
                "a",
                "Editor",
                SourceCapture::PickerOnly(vec![BackendId::LinuxPortal]),
            ),
            window("42", "Term", SourceCapture::ById),
        ];
        let by_title = Target::resolve(
            &CaptureSource::Window {
                identifier: Some(WindowIdentifier::Title("Editor".into())),
            },
            &listed,
            None,
        )
        .unwrap();
        assert_eq!(by_title, Target::portal(SourceKind::Window));
        let by_id = Target::resolve(
            &CaptureSource::Window {
                identifier: Some(WindowIdentifier::Id(42)),
            },
            &listed,
            None,
        )
        .unwrap();
        assert_eq!(
            by_id.selection().0,
            SourceSelector::Id(listed[1].id.clone())
        );
        let unnamed =
            Target::resolve(&CaptureSource::Window { identifier: None }, &listed, None).unwrap();
        assert_eq!(unnamed, Target::portal(SourceKind::Window));
    }

    fn by_identifier(identifier: WindowIdentifier, listed: &[Source]) -> Result<Target> {
        Target::resolve(
            &CaptureSource::Window {
                identifier: Some(identifier),
            },
            listed,
            None,
        )
    }

    #[test]
    fn a_window_is_found_by_its_listed_id_among_same_title_windows() {
        let listed = [
            window("toplevel:1", "Terminal", SourceCapture::ById),
            window("toplevel:2", "Terminal", SourceCapture::ById),
        ];
        let target = by_identifier(WindowIdentifier::Source("toplevel:2".into()), &listed).unwrap();
        assert_eq!(
            target.selection().0,
            SourceSelector::Id(listed[1].id.clone())
        );
        // A title alone is the explicit fallback: the first match.
        let target = by_identifier(WindowIdentifier::Title("Terminal".into()), &listed).unwrap();
        assert_eq!(
            target.selection().0,
            SourceSelector::Id(listed[0].id.clone())
        );
    }

    #[test]
    fn an_xid_names_an_x11_window() {
        let listed = [window("window:0x2a00007", "xterm", SourceCapture::ById)];
        let target = by_identifier(WindowIdentifier::Id(0x2a0_0007), &listed).unwrap();
        assert_eq!(
            target.selection().0,
            SourceSelector::Id(listed[0].id.clone())
        );
    }

    #[test]
    fn an_unknown_window_is_an_error_not_a_dialog() {
        let listed = [window("toplevel:1", "Terminal", SourceCapture::ById)];
        assert!(by_identifier(WindowIdentifier::Source("toplevel:9".into()), &listed).is_err());
        assert!(by_identifier(WindowIdentifier::Title("Gone".into()), &listed).is_err());
    }

    #[test]
    fn a_pinned_portal_always_asks() {
        let target = Target::resolve(
            &CaptureSource::FullScreen {
                monitor: Some("DP-3".into()),
            },
            &[monitor("17", "DP-3")],
            Some(BackendSelection::Fixed(BackendId::LinuxPortal)),
        )
        .unwrap();
        assert_eq!(target, Target::portal(SourceKind::Monitor));
    }
}
