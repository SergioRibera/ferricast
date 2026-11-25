//! The backends that can do one job, and how a [`BackendSelection`]
//! turns them into the ordered list a codec tries.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ferricast_core::{
    Acceleration, AccelerationPolicy, BackendAttempt, BackendId, BackendSelection, CodecError,
    CodecResult, CodecTarget, DevNumber, GpuAffinity, GpuPlacement, RuntimeFallback,
};
use tracing::{debug, info};

use crate::cache::{self, SelectionCache, SelectionInput, SelectionKey};

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
    gpu: Option<GpuAffinity>,
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
            gpu: None,
        }
    }

    /// Marks a backend that runs on a DRM GPU: automatic selection orders
    /// it by the configuration's [`ferricast_core::GpuPreference`], and
    /// drops it when the preference pins a GPU it cannot run on.
    #[must_use]
    pub fn on_gpu(mut self, affinity: GpuAffinity) -> Self {
        self.gpu = Some(affinity);
        self
    }

    /// The GPUs it runs on; `None` for a backend not tied to a DRM GPU
    /// (software, platform frameworks).
    #[must_use]
    pub fn gpu_affinity(&self) -> Option<GpuAffinity> {
        self.gpu
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
    /// `None`: the process-wide cache ([`cache::global`]).
    cache: Option<Arc<dyn SelectionCache>>,
}

impl<C, S: ?Sized> fmt::Debug for Registry<C, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registry")
            .field("target", &self.target)
            .field("candidates", &self.candidates)
            .field("missing", &self.missing)
            .field("cache", &self.cache)
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
            cache: None,
        }
    }

    /// Remembers automatic selection's winners in `cache` instead of the
    /// process-wide one (see [`crate::cache`]).
    #[must_use]
    pub fn with_cache(mut self, cache: Arc<dyn SelectionCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Uses `cache` from now on, as [`Self::with_cache`] does.
    pub fn set_cache(&mut self, cache: Arc<dyn SelectionCache>) {
        self.cache = Some(cache);
    }

    /// The cache automatic selection remembers its winners in.
    #[must_use]
    pub fn cache(&self) -> Arc<dyn SelectionCache> {
        self.cache.clone().unwrap_or_else(cache::global)
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
        self.plan_on(selection, &GpuPlacement::default())
    }

    /// [`Self::plan`] for the GPUs `gpus` ranks: automatic selection tries
    /// the GPU-bound hardware backends ([`Candidate::on_gpu`]) that run on
    /// the first ranked GPU first, then those of the next GPU, then the
    /// other hardware backends; a placement pinned to one GPU drops the
    /// GPU-bound backends that cannot run on it. `Fixed` and `Ordered`
    /// selections keep their order (their backends still open on the
    /// preferred GPU). Without GPUs (no sysfs) the order is unchanged.
    ///
    /// # Errors
    /// As [`Self::plan`].
    pub fn plan_on(&self, selection: &BackendSelection, gpus: &GpuPlacement) -> CodecResult<Plan> {
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
                let filtered = self.by_gpu(filtered, gpus);
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
            auto: match &selection {
                BackendSelection::Auto(auto) => Some(auto.acceleration),
                BackendSelection::Fixed(_) | BackendSelection::Ordered { .. } => None,
            },
            gpu: gpus.key(),
        })
    }

    /// `order` (an automatic plan) with its GPU-bound hardware backends
    /// grouped by the first ranked GPU each runs on, best GPU first, ahead
    /// of the other hardware backends; software keeps its place relative
    /// to hardware (the acceleration policy decided it).
    fn by_gpu(&self, order: Vec<usize>, gpus: &GpuPlacement) -> Vec<usize> {
        if gpus.gpus().is_empty() && !gpus.is_pinned() {
            return order;
        }
        let rank = |index: usize| -> Option<Option<usize>> {
            let candidate = &self.candidates[index];
            let affinity = candidate.gpu?;
            Some(gpus.gpus().iter().position(|gpu| affinity.accepts(gpu)))
        };
        let hardware =
            |index: &usize| self.candidates[*index].acceleration == Acceleration::Hardware;
        let mut gpu_bound: Vec<(usize, usize)> = Vec::new();
        let mut unplaced = Vec::new();
        let mut others = Vec::new();
        for &index in order.iter().filter(|index| hardware(index)) {
            match rank(index) {
                Some(Some(position)) => gpu_bound.push((position, index)),
                // On no ranked GPU: kept last, unless one GPU is pinned.
                Some(None) if gpus.is_pinned() => {
                    debug!(
                        target_job = %self.target,
                        backend = %self.candidates[index].backend,
                        "backend cannot run on the pinned GPU; left out"
                    );
                }
                Some(None) => unplaced.push(index),
                None => others.push(index),
            }
        }
        // Stable: within one GPU the registry's order stays.
        gpu_bound.sort_by_key(|(position, _)| *position);
        let hardware_order = gpu_bound
            .into_iter()
            .map(|(_, index)| index)
            .chain(unplaced)
            .chain(others);
        // The policy keeps each kind together: the hardware block goes
        // where the first hardware backend was.
        let first_hardware = order.iter().position(hardware).unwrap_or(order.len());
        let software = order.iter().copied().filter(|index| !hardware(index));
        let (before, after): (Vec<usize>, Vec<usize>) = software
            .partition(|&index| order.iter().position(|&i| i == index) < Some(first_hardware));
        before
            .into_iter()
            .chain(hardware_order)
            .chain(after)
            .collect()
    }

    /// Opens the first candidate of `plan`, starting at `from`, that
    /// accepts `config`.
    ///
    /// An automatic plan opened from its start tries the backend its
    /// cache remembers for this job and configuration first (see
    /// [`crate::cache`]), then the rest in order; whichever opens is
    /// remembered. A candidate with a probe is only opened when its
    /// (once per process) probe passes.
    ///
    /// # Errors
    /// A [`Plan`] for a single named backend returns that backend's own
    /// error; otherwise [`CodecError::NoBackendOpened`] lists why each
    /// candidate was rejected and how long it took.
    pub fn open(&self, plan: &Plan, from: usize, config: &C) -> CodecResult<Opened<S>>
    where
        C: SelectionInput,
    {
        let memory = self.memory(plan, config);
        let remembered = memory
            .as_ref()
            .filter(|_| from == 0)
            .and_then(|(cache, key)| cache.winner(key))
            .and_then(|winner| {
                plan.order
                    .iter()
                    .position(|&index| self.candidates[index].backend == winner)
            });
        let order = remembered
            .into_iter()
            .chain((from..plan.order.len()).filter(|&position| Some(position) != remembered));
        let mut attempts = Vec::new();
        for position in order {
            let candidate = &self.candidates[plan.order[position]];
            let started = Instant::now();
            let result = self.try_open(candidate, config);
            let elapsed = started.elapsed();
            match result {
                Ok(session) => {
                    info!(
                        target_job = %self.target,
                        backend = %candidate.backend,
                        acceleration = %candidate.acceleration,
                        elapsed_ms = elapsed.as_secs_f64() * 1e3,
                        remembered = remembered == Some(position),
                        rejected = attempts.len(),
                        "codec backend opened"
                    );
                    if let Some((cache, key)) = &memory {
                        cache.record(key, candidate.backend);
                    }
                    return Ok(Opened {
                        session,
                        position,
                        backend: candidate.backend,
                        acceleration: candidate.acceleration,
                        elapsed,
                        attempts,
                    });
                }
                Err(error) => {
                    debug!(
                        target_job = %self.target,
                        backend = %candidate.backend,
                        elapsed_ms = elapsed.as_secs_f64() * 1e3,
                        %error,
                        "codec backend rejected the configuration"
                    );
                    if let Some((cache, key)) = &memory {
                        cache.evict(key, candidate.backend);
                    }
                    if plan.fixed && plan.order.len() == 1 {
                        return Err(error);
                    }
                    attempts.push(BackendAttempt {
                        backend: candidate.backend,
                        reason: error.to_string(),
                        elapsed_us: micros(elapsed),
                    });
                }
            }
        }
        Err(CodecError::NoBackendOpened {
            target: self.target,
            attempts,
        })
    }

    /// The backend `plan` would most likely open, without opening any: the
    /// first whose (once per process) probe passes, or that has none. For
    /// decisions made before a session exists (what capture should hand
    /// the encoder); a configuration the backend refuses still moves on.
    #[must_use]
    pub fn first_available(&self, plan: &Plan) -> Option<BackendId> {
        plan.order
            .iter()
            .map(|&index| &self.candidates[index])
            .find(|candidate| {
                candidate.probe.is_none_or(|probe| {
                    cache::probe_once(candidate.backend, self.target, probe).is_ok()
                })
            })
            .map(Candidate::backend)
    }

    /// `backend`, which `plan` opened for `config`, failed on the
    /// stream: automatic selection stops trying it first.
    pub fn demote(&self, plan: &Plan, config: &C, backend: BackendId)
    where
        C: SelectionInput,
    {
        if let Some((cache, key)) = self.memory(plan, config) {
            cache.evict(&key, backend);
        }
    }

    /// The cache and key an automatic plan remembers its winner under.
    fn memory(&self, plan: &Plan, config: &C) -> Option<(Arc<dyn SelectionCache>, SelectionKey)>
    where
        C: SelectionInput,
    {
        let policy = plan.auto?;
        let key = SelectionKey {
            target: self.target,
            policy,
            traits: config.selection_traits(),
            gpu: plan.gpu,
        };
        Some((self.cache(), key))
    }

    /// Probes (once per process) then opens `candidate`. A failed open
    /// forgets the probe verdict, so the next selection probes again.
    fn try_open(&self, candidate: &Candidate<C, S>, config: &C) -> CodecResult<Box<S>> {
        // A probe initialises the device (CUDA, VA-API) just as opening
        // does, so running one here would pay that start-up twice; only a
        // verdict already known to be negative skips the open.
        if candidate.probe.is_some()
            && let Some(error) = cache::known_unavailable(candidate.backend, self.target)
        {
            return Err(error);
        }
        candidate
            .open(config)
            .inspect_err(|_| cache::forget_probe(candidate.backend, self.target))
    }

    /// Every compiled backend with whether it can run here (probing the
    /// ones that have a probe, once per process), plus the ones this
    /// build leaves out.
    #[must_use]
    pub fn report(&self) -> Vec<BackendReport> {
        let compiled = self.candidates.iter().map(|candidate| BackendReport {
            backend: candidate.backend,
            target: self.target,
            acceleration: Some(candidate.acceleration),
            automatic: candidate.automatic,
            availability: match candidate.probe {
                Some(probe) => match cache::probe_once(candidate.backend, self.target, probe) {
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
    /// The acceleration policy of an automatic selection, the one kind
    /// of plan that remembers its winner.
    auto: Option<AccelerationPolicy>,
    /// The most preferred GPU, part of the remembered winner's key.
    gpu: Option<DevNumber>,
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
    /// How long opening it took.
    pub elapsed: Duration,
    /// The candidates rejected before it, with why and how long each took.
    pub attempts: Vec<BackendAttempt>,
}

impl<S: ?Sized> fmt::Debug for Opened<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Opened")
            .field("position", &self.position)
            .field("backend", &self.backend)
            .field("acceleration", &self.acceleration)
            .field("elapsed", &self.elapsed)
            .field("attempts", &self.attempts)
            .finish_non_exhaustive()
    }
}

/// `elapsed` in whole microseconds, saturating.
fn micros(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX)
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

    /// A test configuration is a picture height.
    impl SelectionInput for u32 {
        fn selection_traits(&self) -> cache::ConfigTraits {
            cache::ConfigTraits {
                size: cache::SizeClass::of(*self, *self),
                ..cache::ConfigTraits::AUDIO
            }
        }
    }

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

    /// Remembers nothing, so tests do not see each other's winners.
    fn registry() -> TestRegistry {
        Registry::new(CodecTarget::decode(Codec::H264))
            .with_cache(Arc::new(cache::NoCache))
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

    static COUNTED_REFUSALS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    #[expect(clippy::trivially_copy_pass_by_ref, reason = "an `OpenFn`")]
    fn counted_refusal(config: &u32) -> CodecResult<Box<str>> {
        COUNTED_REFUSALS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        refuses(config)
    }

    #[test]
    fn a_warm_cache_opens_the_remembered_winner_first() {
        let memory: Arc<dyn SelectionCache> = Arc::new(cache::MemoryCache::default());
        let registry = TestRegistry::new(CodecTarget::encode(Codec::H264))
            .with_cache(Arc::clone(&memory))
            .candidate(Candidate::new(
                BackendId::Nvenc,
                Acceleration::Hardware,
                counted_refusal,
            ))
            .candidate(Candidate::new(
                BackendId::X264,
                Acceleration::Software,
                opens,
            ));
        let plan = registry.plan(&BackendSelection::default()).unwrap();
        let cold = registry.open(&plan, 0, &1080).unwrap();
        assert_eq!(cold.backend, BackendId::X264);
        assert_eq!(cold.attempts.len(), 1, "the cold open tried nvenc first");
        let refusals = COUNTED_REFUSALS.load(std::sync::atomic::Ordering::SeqCst);

        let warm = registry.open(&plan, 0, &1080).unwrap();
        assert_eq!((warm.backend, warm.position), (BackendId::X264, 1));
        assert!(warm.attempts.is_empty(), "the warm open skipped nvenc");
        assert_eq!(
            COUNTED_REFUSALS.load(std::sync::atomic::Ordering::SeqCst),
            refusals
        );

        // Another size class is another key: the full order again.
        let other = registry.open(&plan, 0, &4000).unwrap();
        assert_eq!(other.attempts.len(), 1);

        // A pinned selection never consults the cache.
        let fixed = registry.plan(&BackendId::Nvenc.into()).unwrap();
        assert!(registry.open(&fixed, 0, &1080).is_err());

        // A runtime failure of the winner makes it lose its place.
        registry.demote(&plan, &1080, BackendId::X264);
        assert_eq!(registry.open(&plan, 0, &1080).unwrap().attempts.len(), 1);
    }

    #[test]
    fn opening_never_probes_but_honours_a_failed_probe_verdict() {
        static PROBED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        // A job no other test uses: probe verdicts are process-wide.
        let registry = TestRegistry::new(CodecTarget::encode(Codec::Vp8))
            .with_cache(Arc::new(cache::NoCache))
            .candidate(
                Candidate::new(BackendId::Nvenc, Acceleration::Hardware, refuses).with_probe(
                    || {
                        PROBED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Err(CodecError::BackendUnavailable {
                            backend: BackendId::Nvenc,
                            reason: "no NVIDIA driver".into(),
                        })
                    },
                ),
            )
            .candidate(Candidate::new(
                BackendId::X264,
                Acceleration::Software,
                opens,
            ));
        let plan = registry.plan(&BackendSelection::default()).unwrap();
        // Opening is the test: no probe runs, nvenc is tried and refuses.
        let opened = registry.open(&plan, 0, &720).unwrap();
        assert!(!opened.attempts[0].reason.contains("no NVIDIA driver"));
        assert_eq!(PROBED.load(std::sync::atomic::Ordering::SeqCst), 0);
        // A capability report probes once; its negative verdict then skips
        // the open.
        assert_eq!(registry.report().len(), 2);
        for _ in 0..3 {
            let opened = registry.open(&plan, 0, &720).unwrap();
            assert!(opened.attempts[0].reason.contains("no NVIDIA driver"));
        }
        assert_eq!(registry.report().len(), 2);
        assert_eq!(PROBED.load(std::sync::atomic::Ordering::SeqCst), 1);
        cache::forget_probe(BackendId::Nvenc, CodecTarget::encode(Codec::Vp8));
    }
}

#[cfg(test)]
mod gpu_order_tests {
    use ferricast_core::{
        AutoSelect, Codec, DevNumber, Gpu, GpuKind, GpuPreference, GpuVendor, Gpus,
    };

    use super::*;

    #[expect(clippy::trivially_copy_pass_by_ref, reason = "an `OpenFn`")]
    fn opens(_: &u32) -> CodecResult<Box<str>> {
        Ok("session".into())
    }

    /// NVENC, VA-API, a platform encoder, x264: the facades' order.
    fn registry() -> Registry<u32, str> {
        Registry::new(CodecTarget::encode(Codec::H264))
            .with_cache(Arc::new(cache::NoCache))
            .candidate(
                Candidate::new(BackendId::Nvenc, Acceleration::Hardware, opens)
                    .on_gpu(GpuAffinity::Nvidia),
            )
            .candidate(
                Candidate::new(BackendId::Vaapi, Acceleration::Hardware, opens)
                    .on_gpu(GpuAffinity::NotNvidia),
            )
            .candidate(Candidate::new(
                BackendId::MediaFoundation,
                Acceleration::Hardware,
                opens,
            ))
            .candidate(Candidate::new(
                BackendId::X264,
                Acceleration::Software,
                opens,
            ))
    }

    fn gpu(minor: u32, vendor: GpuVendor, pci: &str) -> Gpu {
        Gpu {
            render_node: format!("/dev/dri/renderD{minor}"),
            render_dev: DevNumber { major: 226, minor },
            primary_dev: Some(DevNumber {
                major: 226,
                minor: minor - 128,
            }),
            pci: pci.parse().ok(),
            vendor,
            device_id: 0,
            driver: None,
            boot_vga: false,
            kind: GpuKind::Unknown,
        }
    }

    fn hybrid() -> Gpus {
        Gpus::classify(vec![
            gpu(128, GpuVendor::INTEL, "0000:00:02.0"),
            gpu(129, GpuVendor::NVIDIA, "0000:01:00.0"),
        ])
    }

    fn order(
        preference: &str,
        source: Option<u32>,
        selection: &BackendSelection,
    ) -> Vec<BackendId> {
        let registry = registry();
        let source =
            source.map(|minor| ferricast_core::GpuId::Dev(DevNumber { major: 226, minor }));
        let placement = GpuPlacement::resolve_in(
            &preference.parse::<GpuPreference>().unwrap(),
            source.as_ref(),
            &hybrid(),
        );
        let plan = registry.plan_on(selection, &placement).unwrap();
        plan.order
            .iter()
            .map(|&index| registry.candidates[index].backend)
            .collect()
    }

    #[test]
    fn hardware_backends_follow_the_preferred_gpu() {
        use BackendId::{MediaFoundation as Mf, Nvenc, Vaapi, X264};
        let auto = BackendSelection::default();
        // Frames rendered on the iGPU: VA-API there first (no copy).
        assert_eq!(order("auto", Some(128), &auto), [Vaapi, Nvenc, Mf, X264]);
        assert_eq!(order("auto", Some(129), &auto), [Nvenc, Vaapi, Mf, X264]);
        // No producer: discrete first.
        assert_eq!(order("auto", None, &auto), [Nvenc, Vaapi, Mf, X264]);
        assert_eq!(
            order("integrated", Some(129), &auto),
            [Vaapi, Nvenc, Mf, X264]
        );
        assert_eq!(
            order("discrete", Some(128), &auto),
            [Nvenc, Vaapi, Mf, X264]
        );
        // Pins drop GPU-bound backends that cannot run there.
        assert_eq!(order("renderD128", None, &auto), [Vaapi, Mf, X264]);
        assert_eq!(order("0000:01:00.0", None, &auto), [Nvenc, Mf, X264]);
        assert_eq!(
            order("same-as-capture", Some(128), &auto),
            [Vaapi, Mf, X264]
        );
        assert_eq!(order("renderD200", None, &auto), [Mf, X264]);
        // Software first keeps software first.
        let software_first = BackendSelection::from(
            AutoSelect::default().with_acceleration(AccelerationPolicy::PreferSoftware),
        );
        assert_eq!(
            order("integrated", None, &software_first),
            [X264, Vaapi, Nvenc, Mf]
        );
        // Fixed and ordered selections are not reordered.
        assert_eq!(order("renderD128", None, &BackendId::Nvenc.into()), [Nvenc]);
        let ordered = BackendSelection::Ordered {
            backends: vec![Nvenc, Vaapi],
            runtime_fallback: RuntimeFallback::Disabled,
        };
        assert_eq!(order("integrated", None, &ordered), [Nvenc, Vaapi]);
    }

    #[test]
    fn without_gpus_the_order_is_unchanged() {
        let registry = registry();
        let placement =
            GpuPlacement::resolve_in(&GpuPreference::Integrated, None, &Gpus::default());
        let plan = registry
            .plan_on(&BackendSelection::default(), &placement)
            .unwrap();
        assert_eq!(plan.order, [0, 1, 2, 3]);
        assert_eq!(plan.gpu, None);
    }
}
