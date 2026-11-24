//! How a codec picks its backend.
//!
//! [`BackendSelection::Auto`] lets the codec probe every compiled
//! backend in preference order and keep the first that opens; the other
//! variants pin the choice. Users override `Auto` without recompiling
//! through the job's environment variable ([`CodecTarget::env_var`]),
//! which takes the same text [`BackendSelection`] parses.

use std::fmt;
use std::str::FromStr;

use super::backend::{BackendId, CodecTarget};
use super::error::CodecError;

/// Which kinds of backend automatic selection may pick, and in what order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum AccelerationPolicy {
    /// Hardware backends first, software ones as the fallback.
    #[default]
    PreferHardware,
    /// Software backends first (reproducible output, no driver quirks),
    /// hardware ones as the fallback.
    PreferSoftware,
    /// Hardware backends only; fails rather than burn the CPU.
    HardwareOnly,
    /// Software backends only.
    SoftwareOnly,
}

impl fmt::Display for AccelerationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PreferHardware => "prefer hardware",
            Self::PreferSoftware => "prefer software",
            Self::HardwareOnly => "hardware only",
            Self::SoftwareOnly => "software only",
        })
    }
}

/// What a codec does when the backend it opened fails on the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum RuntimeFallback {
    /// Report the failure; never switch backends once one is open.
    Disabled,
    /// Move to the next candidate backend. A decoder that fails before
    /// its first picture replays up to `packets` buffered packets into
    /// the next backend so no picture is lost; past that window, or
    /// once a picture came out, only a failure the backend reports as
    /// [`CodecError::UnsupportedStream`] still hands the stream over.
    NextBackend { packets: u16 },
}

impl RuntimeFallback {
    /// Packets a decoder keeps for replay by default: enough to reach
    /// the first picture of any sane stream (one GOP prefix), small
    /// enough not to matter in memory.
    pub const DEFAULT_REPLAY_PACKETS: u16 = 64;
}

impl Default for RuntimeFallback {
    fn default() -> Self {
        Self::NextBackend {
            packets: Self::DEFAULT_REPLAY_PACKETS,
        }
    }
}

/// Automatic selection: probe compiled backends in preference order
/// and keep the first that opens with the configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct AutoSelect {
    pub acceleration: AccelerationPolicy,
    pub runtime_fallback: RuntimeFallback,
    /// Backends never to pick, e.g. one known to misbehave on a device.
    pub exclude: Vec<BackendId>,
}

impl AutoSelect {
    #[must_use]
    pub fn with_acceleration(mut self, acceleration: AccelerationPolicy) -> Self {
        self.acceleration = acceleration;
        self
    }

    #[must_use]
    pub fn with_runtime_fallback(mut self, fallback: RuntimeFallback) -> Self {
        self.runtime_fallback = fallback;
        self
    }

    #[must_use]
    pub fn excluding(mut self, backend: BackendId) -> Self {
        if !self.exclude.contains(&backend) {
            self.exclude.push(backend);
        }
        self
    }
}

/// How a codec picks its backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum BackendSelection {
    /// Let the codec choose (see [`AutoSelect`]). The job's environment
    /// variable, when set, replaces this choice.
    Auto(AutoSelect),
    /// Exactly this backend: opening fails with an error naming why
    /// instead of falling back, and a runtime failure is reported.
    Fixed(BackendId),
    /// These backends, in this order.
    Ordered {
        backends: Vec<BackendId>,
        runtime_fallback: RuntimeFallback,
    },
}

impl Default for BackendSelection {
    fn default() -> Self {
        Self::Auto(AutoSelect::default())
    }
}

impl From<BackendId> for BackendSelection {
    fn from(backend: BackendId) -> Self {
        Self::Fixed(backend)
    }
}

impl From<AutoSelect> for BackendSelection {
    fn from(auto: AutoSelect) -> Self {
        Self::Auto(auto)
    }
}

impl BackendSelection {
    /// What the codec for `target` actually uses: `self`, unless it is
    /// [`Self::Auto`] and the job's environment variable (or one of its
    /// legacy names) is set, which then wins.
    ///
    /// # Errors
    /// [`CodecError::InvalidSelection`] when the variable is set to
    /// text that is not a selection.
    pub fn resolve_env(&self, target: CodecTarget) -> Result<Self, CodecError> {
        if !matches!(self, Self::Auto(_)) {
            return Ok(self.clone());
        }
        let canonical = target.env_var();
        let names =
            std::iter::once(canonical.as_str()).chain(target.legacy_env_vars().iter().copied());
        for name in names {
            let Ok(value) = std::env::var(name) else {
                continue;
            };
            if value.trim().is_empty() {
                continue;
            }
            return value.parse().map_err(|error: InvalidSelectionText| {
                CodecError::InvalidSelection {
                    origin: format!("the environment variable {name}"),
                    value,
                    reason: error.to_string(),
                }
            });
        }
        Ok(self.clone())
    }

    /// The runtime fallback this selection asks for.
    #[must_use]
    pub fn runtime_fallback(&self) -> RuntimeFallback {
        match self {
            Self::Auto(auto) => auto.runtime_fallback,
            Self::Fixed(_) => RuntimeFallback::Disabled,
            Self::Ordered {
                runtime_fallback, ..
            } => *runtime_fallback,
        }
    }
}

/// Why text is not a [`BackendSelection`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "{reason}; expected `auto`, `hardware`, `software`, `prefer-hardware`, `prefer-software`, a \
     backend name or a comma-separated list of backend names ({})",
    BackendId::all_names()
)]
pub struct InvalidSelectionText {
    reason: String,
}

impl FromStr for BackendSelection {
    type Err = InvalidSelectionText;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let policy = |acceleration| {
            Ok(Self::Auto(
                AutoSelect::default().with_acceleration(acceleration),
            ))
        };
        match text.trim().to_ascii_lowercase().as_str() {
            "" => {
                return Err(InvalidSelectionText {
                    reason: "the selection is empty".into(),
                });
            }
            "auto" => return Ok(Self::default()),
            "hardware" | "hardware-only" => return policy(AccelerationPolicy::HardwareOnly),
            "software" | "software-only" => return policy(AccelerationPolicy::SoftwareOnly),
            "prefer-hardware" => return policy(AccelerationPolicy::PreferHardware),
            "prefer-software" => return policy(AccelerationPolicy::PreferSoftware),
            _ => {}
        }
        let backends = text
            .split(',')
            .map(str::parse::<BackendId>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| InvalidSelectionText {
                reason: format!("{:?} is not a backend name", error.value),
            })?;
        Ok(match backends.as_slice() {
            [single] => Self::Fixed(*single),
            _ => Self::Ordered {
                backends,
                runtime_fallback: RuntimeFallback::default(),
            },
        })
    }
}

impl fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto(auto) => {
                write!(f, "automatic ({})", auto.acceleration)?;
                if !auto.exclude.is_empty() {
                    let names: Vec<_> = auto.exclude.iter().map(|b| b.name()).collect();
                    write!(f, " excluding {}", names.join(", "))?;
                }
                Ok(())
            }
            Self::Fixed(backend) => write!(f, "{backend} only"),
            Self::Ordered { backends, .. } => {
                let names: Vec<_> = backends.iter().map(|b| b.name()).collect();
                write!(f, "{} in this order", names.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Codec;

    #[test]
    fn selections_parse_from_text() {
        assert_eq!("auto".parse(), Ok(BackendSelection::default()));
        assert_eq!(
            "Software".parse(),
            Ok(BackendSelection::Auto(
                AutoSelect::default().with_acceleration(AccelerationPolicy::SoftwareOnly)
            ))
        );
        assert_eq!(
            "nvdec".parse(),
            Ok(BackendSelection::Fixed(BackendId::Nvdec))
        );
        assert_eq!(
            "vaapi, openh264".parse(),
            Ok(BackendSelection::Ordered {
                backends: vec![BackendId::Vaapi, BackendId::OpenH264],
                runtime_fallback: RuntimeFallback::default(),
            })
        );
        let error = "vaapi,ffmpeg".parse::<BackendSelection>().unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("\"ffmpeg\" is not a backend name"),
            "{error}"
        );
        assert!("  ".parse::<BackendSelection>().is_err());
    }

    #[test]
    fn the_environment_replaces_automatic_selection() {
        let target = CodecTarget::decode(crate::AudioCodec::Pcm);
        let var = target.env_var();
        // SAFETY: no other test reads or writes FERRICAST_PCM_DECODE_BACKEND,
        // and the process spawns no thread that reads the environment.
        unsafe { std::env::set_var(&var, "pcm") };
        let resolved = BackendSelection::default().resolve_env(target);
        // SAFETY: as above.
        unsafe { std::env::set_var(&var, "ffmpeg") };
        let invalid = BackendSelection::default().resolve_env(target);
        // SAFETY: as above.
        unsafe { std::env::remove_var(&var) };
        assert_eq!(resolved, Ok(BackendSelection::Fixed(BackendId::Pcm)));
        assert!(matches!(
            invalid,
            Err(CodecError::InvalidSelection { origin, value, .. })
                if origin.contains("FERRICAST_PCM_DECODE_BACKEND") && value == "ffmpeg"
        ));
    }

    #[test]
    fn a_pinned_selection_ignores_the_environment() {
        let fixed = BackendSelection::Fixed(BackendId::X264);
        assert_eq!(
            fixed.resolve_env(CodecTarget::encode(Codec::H264)),
            Ok(fixed.clone())
        );
        assert_eq!(fixed.runtime_fallback(), RuntimeFallback::Disabled);
    }
}
