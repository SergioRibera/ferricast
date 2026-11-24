//! The backends that can do one job, and how a [`BackendSelection`]
//! turns them into the ordered list a codec tries.

use std::fmt;

use ferricast_core::{
    Acceleration, AccelerationPolicy, BackendAttempt, BackendId, BackendSelection, CodecError,
    CodecResult, CodecTarget, RuntimeFallback,
};
use tracing::{debug, info};

/// Opens a backend session for a configuration.
pub type OpenFn<C, S> = fn(&C) -> CodecResult<Box<S>>;

/// Checks, without a configuration, whether a backend can run here.
pub type ProbeFn = fn() -> CodecResult<()>;

/// One compiled backend that can do a registry's job.
pub struct Candidate<C, S: ?Sized> {
    backend: BackendId,
    acceleration: Acceleration,
    open: OpenFn<C, S>,
    probe: Option<ProbeFn>,
    automatic: bool,
}

impl<C, S: ?Sized> Candidate<C, S> {
    /// A backend automatic selection may pick.
    #[must_use]
    pub fn new(backend: BackendId, acceleration: Acceleration, open: OpenFn<C, S>) -> Self {
        Self {
            backend,
            acceleration,
            open,
            probe: None,
            automatic: true,
        }
    }

    /// Adds a cheap availability check, used by capability reports.
    #[must_use]
    pub fn with_probe(mut self, probe: ProbeFn) -> Self {
        self.probe = Some(probe);
        self
    }

    /// Keeps the backend out of automatic selection: only a selection
    /// that names it opens it (a backend still being validated).
    #[must_use]
    pub fn opt_in(mut self) -> Self {
        self.automatic = false;
        self
    }

    #[must_use]
    /// The backend this candidate opens.
    pub fn backend(&self) -> BackendId {
        self.backend
    }

    #[must_use]
    /// Whether it runs on media hardware.
    pub fn acceleration(&self) -> Acceleration {
        self.acceleration
    }

    /// Whether automatic selection considers this backend.
    #[must_use]
    pub fn is_automatic(&self) -> bool {
        self.automatic
    }

    /// Opens a session.
    ///
    /// # Errors
    /// Whatever the backend reports.
    pub fn open(&self, config: &C) -> CodecResult<Box<S>> {
        (self.open)(config)
    }
}

impl<C, S: ?Sized> fmt::Debug for Candidate<C, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Candidate")
            .field("backend", &self.backend)
            .field("acceleration", &self.acceleration)
            .field("automatic", &self.automatic)
            .finish_non_exhaustive()
    }
}

/// A backend that can do the job in some build but not in this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Missing {
    /// The backend left out.
    pub backend: BackendId,
    /// `crate/feature` that compiles it in.
    pub feature: &'static str,
    /// Why it is absent even when the feature is on, if it is a
    /// platform matter (e.g. "only on Windows").
    pub platform: Option<&'static str>,
}

/// Every backend for one job, in the order automatic selection prefers
/// them (fastest first; within a kind, the most robust first).
pub struct Registry<C, S: ?Sized> {
    target: CodecTarget,
    candidates: Vec<Candidate<C, S>>,
    missing: Vec<Missing>,
}

impl<C, S: ?Sized> fmt::Debug for Registry<C, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registry")
            .field("target", &self.target)
            .field("candidates", &self.candidates)
            .field("missing", &self.missing)
            .finish()
    }
}

impl<C, S: ?Sized> Registry<C, S> {
    #[must_use]
    /// An empty registry for `target`; add backends with [`Self::candidate`].
    pub fn new(target: CodecTarget) -> Self {
        Self {
            target,
            candidates: Vec::new(),
            missing: Vec::new(),
        }
    }

    /// Appends a compiled backend (lower preference than the ones before).
    #[must_use]
    pub fn candidate(mut self, candidate: Candidate<C, S>) -> Self {
        self.candidates.push(candidate);
        self
    }

    /// Records a backend this build leaves out because `feature` is off.
    #[must_use]
    pub fn missing(mut self, backend: BackendId, feature: &'static str) -> Self {
        self.missing.push(Missing {
            backend,
            feature,
            platform: None,
        });
        self
    }

    /// Records a backend that only exists on another platform.
    #[must_use]
    pub fn elsewhere(
        mut self,
        backend: BackendId,
        feature: &'static str,
        platform: &'static str,
    ) -> Self {
        self.missing.push(Missing {
            backend,
            feature,
            platform: Some(platform),
        });
        self
    }

    #[must_use]
    /// The job this registry's backends do.
    pub fn target(&self) -> CodecTarget {
        self.target
    }

    /// The compiled backends, in automatic preference order.
    #[must_use]
    pub fn candidates(&self) -> &[Candidate<C, S>] {
        &self.candidates
    }

    /// Backends that could do the job in another build.
    #[must_use]
    pub fn missing_backends(&self) -> &[Missing] {
        &self.missing
    }

    /// The candidates `selection` (after its environment override)
    /// asks for, in the order to try them.
    ///
    /// # Errors
    /// - [`CodecError::InvalidSelection`]: the environment override does
    ///   not parse.
    /// - [`CodecError::NoBackendCompiled`]: nothing in this build does
    ///   the job.
    /// - [`CodecError::BackendNotCompiled`] /
    ///   [`CodecError::BackendCannotHandle`] /
    ///   [`CodecError::BackendUnavailable`]: a named backend is not in
    ///   this build, never does this job, or is for another platform.
    /// - [`CodecError::NoBackendMatchesPolicy`]: automatic selection's
    ///   policy filtered everything out.
    pub fn plan(&self, selection: &BackendSelection) -> CodecResult<Plan> {
        let selection = selection.resolve_env(self.target)?;
        if self.candidates.is_empty() {
            return Err(CodecError::NoBackendCompiled {
                target: self.target,
                features: self.missing_features(),
            });
        }
        let order = match &selection {
            BackendSelection::Auto(auto) => {
                let allowed = |candidate: &&Candidate<C, S>| {
                    candidate.automatic && !auto.exclude.contains(&candidate.backend)
                };
                let kind = |wanted: Acceleration| {
                    self.indices(move |candidate| candidate.acceleration == wanted)
                };
                let filtered: Vec<usize> = match auto.acceleration {
                    AccelerationPolicy::PreferHardware => {
                        let mut order = kind(Acceleration::Hardware);
                        order.extend(kind(Acceleration::Software));
                        order
                    }
                    AccelerationPolicy::PreferSoftware => {
                        let mut order = kind(Acceleration::Software);
                        order.extend(kind(Acceleration::Hardware));
                        order
                    }
                    AccelerationPolicy::HardwareOnly => kind(Acceleration::Hardware),
                    AccelerationPolicy::SoftwareOnly => kind(Acceleration::Software),
                }
                .into_iter()
                .filter(|&index| allowed(&&self.candidates[index]))
                .collect();
                if filtered.is_empty() {
                    return Err(CodecError::NoBackendMatchesPolicy {
                        target: self.target,
                        policy: selection.to_string(),
                    });
                }
                filtered
            }
            BackendSelection::Fixed(backend) => self.named(*backend)?,
            BackendSelection::Ordered { backends, .. } => {
                let mut order = Vec::new();
                for backend in backends {
                    for index in self.named(*backend)? {
                        if !order.contains(&index) {
                            order.push(index);
                        }
                    }
                }
                order
            }
        };
        Ok(Plan {
            order,
            runtime_fallback: selection.runtime_fallback(),
            fixed: matches!(selection, BackendSelection::Fixed(_)),
        })
    }

    /// Opens the first candidate of `plan`, starting at `from`, that
    /// accepts `config`.
    ///
    /// # Errors
    /// A [`Plan`] for a single named backend returns that backend's own
    /// error; otherwise [`CodecError::NoBackendOpened`] lists why each
    /// candidate was rejected.
    pub fn open(&self, plan: &Plan, from: usize, config: &C) -> CodecResult<Opened<S>> {
        let mut attempts = Vec::new();
        for (position, &index) in plan.order.iter().enumerate().skip(from) {
            let candidate = &self.candidates[index];
            match candidate.open(config) {
                Ok(session) => {
                    info!(
                        target_job = %self.target,
                        backend = %candidate.backend,
                        acceleration = %candidate.acceleration,
                        "codec backend opened"
                    );
                    return Ok(Opened {
                        session,
                        position,
                        backend: candidate.backend,
                        acceleration: candidate.acceleration,
                    });
                }
                Err(error) => {
                    debug!(
                        target_job = %self.target,
                        backend = %candidate.backend,
                        %error,
                        "codec backend rejected the configuration"
                    );
                    if plan.fixed && plan.order.len() == 1 {
                        return Err(error);
                    }
                    attempts.push(BackendAttempt {
                        backend: candidate.backend,
                        reason: error.to_string(),
                    });
                }
            }
        }
        Err(CodecError::NoBackendOpened {
            target: self.target,
            attempts,
        })
    }

    /// Every compiled backend with whether it can run here (probing the
    /// ones that have a probe), plus the ones this build leaves out.
    #[must_use]
    pub fn report(&self) -> Vec<BackendReport> {
        let compiled = self.candidates.iter().map(|candidate| BackendReport {
            backend: candidate.backend,
            target: self.target,
            acceleration: Some(candidate.acceleration),
            automatic: candidate.automatic,
            availability: match candidate.probe {
                Some(probe) => match probe() {
                    Ok(()) => Availability::Available,
                    Err(error) => Availability::Unavailable {
                        reason: error.to_string(),
                    },
                },
                None => Availability::Unprobed,
            },
        });
        let missing = self.missing.iter().map(|missing| BackendReport {
            backend: missing.backend,
            target: self.target,
            acceleration: None,
            automatic: false,
            availability: Availability::NotCompiled {
                feature: missing.feature.to_owned(),
                platform: missing.platform.map(str::to_owned),
            },
        });
        compiled.chain(missing).collect()
    }

    fn indices(&self, keep: impl Fn(&Candidate<C, S>) -> bool) -> Vec<usize> {
        (0..self.candidates.len())
            .filter(|&index| keep(&self.candidates[index]))
            .collect()
    }

    /// Indices of the candidates for `backend`, or why there are none.
    fn named(&self, backend: BackendId) -> CodecResult<Vec<usize>> {
        let found = self.indices(|candidate| candidate.backend == backend);
        if !found.is_empty() {
            return Ok(found);
        }
        match self
            .missing
            .iter()
            .find(|missing| missing.backend == backend)
        {
            Some(Missing {
                platform: Some(platform),
                ..
            }) => Err(CodecError::BackendUnavailable {
                backend,
                reason: format!("it is {platform}"),
            }),
            Some(missing) => Err(CodecError::BackendNotCompiled {
                backend,
                target: self.target,
                feature: missing.feature.to_owned(),
            }),
            None => Err(CodecError::BackendCannotHandle {
                backend,
                target: self.target,
            }),
        }
    }

    fn missing_features(&self) -> String {
        let features: Vec<_> = self
            .missing
            .iter()
            .filter(|missing| missing.platform.is_none())
            .map(|missing| missing.feature)
            .collect();
        if features.is_empty() {
            "none exists for this platform".to_owned()
        } else {
            features.join(", ")
        }
    }
}

/// The candidates a selection resolved to, in the order to try them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    order: Vec<usize>,
    runtime_fallback: RuntimeFallback,
    fixed: bool,
}

impl Plan {
    /// What to do when the open backend fails on the stream.
    #[must_use]
    pub fn runtime_fallback(&self) -> RuntimeFallback {
        self.runtime_fallback
    }

    /// Whether a candidate follows `position`.
    #[must_use]
    pub fn has_after(&self, position: usize) -> bool {
        position + 1 < self.order.len()
    }
}

/// A session the registry opened.
pub struct Opened<S: ?Sized> {
    /// The open session.
    pub session: Box<S>,
    /// Where in the [`Plan`] it sits, to resume after it on fallback.
    pub position: usize,
    /// The backend that opened.
    pub backend: BackendId,
    /// Whether it runs on media hardware.
    pub acceleration: Acceleration,
}

impl<S: ?Sized> fmt::Debug for Opened<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Opened")
            .field("position", &self.position)
            .field("backend", &self.backend)
            .field("acceleration", &self.acceleration)
            .finish_non_exhaustive()
    }
}

/// Whether a backend can run here.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "istmo", istmo_macros::message(crate = "::istmo_core"))]
pub enum Availability {
    /// Its probe found the driver / library / device it needs.
    Available,
    /// Compiled in, but its probe failed.
    Unavailable {
        /// Why the probe failed.
        reason: String,
    },
    /// Compiled in; it has no configuration-free probe, so only opening
    /// it with a configuration tells.
    Unprobed,
    /// Not in this build.
    NotCompiled {
        /// The `crate/feature` that compiles it in.
        feature: String,
        /// The platform it is limited to, when that is why it is absent.
        platform: Option<String>,
    },
}

/// One backend's standing for a job, as a capability report lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "istmo", istmo_macros::message(crate = "::istmo_core"))]
pub struct BackendReport {
    /// The backend reported on.
    pub backend: BackendId,
    /// The job it was reported for.
    pub target: CodecTarget,
    /// `None` for a backend that is not compiled in.
    pub acceleration: Option<Acceleration>,
    /// Whether automatic selection considers it.
    pub automatic: bool,
    /// Whether it can run here.
    pub availability: Availability,
}

#[cfg(test)]
mod tests {
    use ferricast_core::{AutoSelect, Codec, Operation};

    use super::*;

    type TestRegistry = Registry<u32, str>;

    #[expect(
        clippy::unnecessary_wraps,
        clippy::trivially_copy_pass_by_ref,
        reason = "an `OpenFn`"
    )]
    fn opens(config: &u32) -> CodecResult<Box<str>> {
        Ok(format!("session {config}").into_boxed_str())
    }

    #[expect(clippy::trivially_copy_pass_by_ref, reason = "an `OpenFn`")]
    fn refuses(_: &u32) -> CodecResult<Box<str>> {
        Err(CodecError::backend(
            BackendId::Vaapi,
            Operation::Open,
            "no render node",
        ))
    }

    fn registry() -> TestRegistry {
        Registry::new(CodecTarget::decode(Codec::H264))
            .candidate(Candidate::new(
                BackendId::Nvdec,
                Acceleration::Hardware,
                refuses,
            ))
            .candidate(Candidate::new(
                BackendId::Vaapi,
                Acceleration::Hardware,
                refuses,
            ))
            .candidate(Candidate::new(
                BackendId::OpenH264,
                Acceleration::Software,
                opens,
            ))
            .candidate(Candidate::new(BackendId::Rav1d, Acceleration::Software, opens).opt_in())
            .missing(
                BackendId::MediaFoundation,
                "ferricast-decoder/media-foundation",
            )
            .elsewhere(
                BackendId::VideoToolbox,
                "ferricast-decoder/videotoolbox",
                "only on Apple platforms",
            )
    }

    fn backends(registry: &TestRegistry, plan: &Plan) -> Vec<BackendId> {
        plan.order
            .iter()
            .map(|&index| registry.candidates[index].backend)
            .collect()
    }

    #[test]
    fn automatic_selection_orders_by_policy_and_skips_opt_in_backends() {
        let registry = registry();
        let plan = registry.plan(&BackendSelection::default()).unwrap();
        assert_eq!(
            backends(&registry, &plan),
            [BackendId::Nvdec, BackendId::Vaapi, BackendId::OpenH264]
        );
        let software_first = AutoSelect::default()
            .with_acceleration(AccelerationPolicy::PreferSoftware)
            .excluding(BackendId::Vaapi);
        let plan = registry.plan(&software_first.into()).unwrap();
        assert_eq!(
            backends(&registry, &plan),
            [BackendId::OpenH264, BackendId::Nvdec]
        );
    }

    #[test]
    fn opening_falls_through_and_reports_every_rejection() {
        let registry = registry();
        let plan = registry.plan(&BackendSelection::default()).unwrap();
        let opened = registry.open(&plan, 0, &7).unwrap();
        assert_eq!((opened.backend, opened.position), (BackendId::OpenH264, 2));
        assert_eq!(&*opened.session, "session 7");

        let hardware = AutoSelect::default().with_acceleration(AccelerationPolicy::HardwareOnly);
        let plan = registry.plan(&hardware.into()).unwrap();
        let Err(CodecError::NoBackendOpened { attempts, .. }) = registry.open(&plan, 0, &7) else {
            panic!("hardware-only selection must fail");
        };
        assert_eq!(
            attempts.iter().map(|a| a.backend).collect::<Vec<_>>(),
            [BackendId::Nvdec, BackendId::Vaapi]
        );
    }

    #[test]
    fn a_named_backend_fails_with_its_own_reason() {
        let registry = registry();
        let plan = registry.plan(&BackendId::Vaapi.into()).unwrap();
        assert_eq!(
            registry.open(&plan, 0, &1).unwrap_err(),
            CodecError::backend(BackendId::Vaapi, Operation::Open, "no render node")
        );
        assert!(matches!(
            registry.plan(&BackendId::MediaFoundation.into()),
            Err(CodecError::BackendNotCompiled { feature, .. }) if feature == "ferricast-decoder/media-foundation"
        ));
        assert!(matches!(
            registry.plan(&BackendId::VideoToolbox.into()),
            Err(CodecError::BackendUnavailable { .. })
        ));
        assert!(matches!(
            registry.plan(&BackendId::X264.into()),
            Err(CodecError::BackendCannotHandle { .. })
        ));
        let opt_in = registry.plan(&BackendId::Rav1d.into()).unwrap();
        assert_eq!(
            registry.open(&opt_in, 0, &3).unwrap().backend,
            BackendId::Rav1d
        );
    }

    #[test]
    fn an_empty_build_names_the_features_to_enable() {
        let empty = TestRegistry::new(CodecTarget::decode(Codec::H264))
            .missing(BackendId::OpenH264, "ferricast-decoder/openh264-decode");
        assert_eq!(
            empty.plan(&BackendSelection::default()),
            Err(CodecError::NoBackendCompiled {
                target: CodecTarget::decode(Codec::H264),
                features: "ferricast-decoder/openh264-decode".into(),
            })
        );
    }

    #[test]
    fn reports_cover_compiled_and_missing_backends() {
        let report = registry().report();
        assert_eq!(report.len(), 6);
        assert_eq!(report[0].availability, Availability::Unprobed);
        assert!(matches!(
            &report[5].availability,
            Availability::NotCompiled {
                platform: Some(_),
                ..
            }
        ));
    }
}
