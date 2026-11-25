//! What automatic selection remembers between sessions.
//!
//! Opening a backend can cost tens or hundreds of milliseconds (a GPU
//! context, a driver enumeration, a hardware session that then refuses
//! the configuration), and automatic selection may try several before
//! one opens. Two memories make the next session cheaper:
//!
//! - **Probes**: the configuration-free availability check of a
//!   backend ([`crate::Candidate::with_probe`]) runs once per process
//!   and job; a later session reuses its verdict. A backend whose open
//!   then fails is probed again next time ([`reset_probes`] forgets
//!   every verdict, e.g. after a device was plugged in).
//! - **Winners**: a [`SelectionCache`] remembers which backend opened
//!   for a [`SelectionKey`] (the job, the acceleration policy and the
//!   parts of the configuration that change which backend can take it),
//!   and automatic selection tries that backend first. It never
//!   overrides a [`ferricast_core::BackendSelection::Fixed`] or
//!   [`ferricast_core::BackendSelection::Ordered`] selection, the
//!   policy or the exclusions: the remembered backend is only moved to
//!   the front when the selection would try it anyway. When it fails it
//!   is forgotten and the usual order continues.
//!
//! The cache a registry uses is, in order: the one given with
//! [`crate::Registry::with_cache`] (or a facade's / `CodecService`'s
//! builder), else the process-wide one ([`global`], replaceable with
//! [`install`]), which the `FERRICAST_CODEC_CACHE` environment variable
//! picks on first use:
//!
//! | Value | Cache |
//! |---|---|
//! | unset, empty, `memory` | [`MemoryCache`]: this process only |
//! | `off`, `none` | [`NoCache`]: always the full order |
//! | anything else | [`FileCache`] at that path: survives restarts |

use std::collections::HashMap;
use std::fmt;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, RwLock};

use ferricast_core::{
    AccelerationPolicy, AudioDecoderConfig, AudioEncoderConfig, BackendId, BitDepth, CodecError,
    CodecTarget, DecoderConfig, DevNumber, EncoderConfig, FrameMemory,
};
use tracing::{debug, warn};

/// The environment variable that picks the process-wide cache.
pub const CACHE_ENV_VAR: &str = "FERRICAST_CODEC_CACHE";

/// Picture size, coarsely: backends accept or refuse by size class (a
/// hardware encoder's maximum, a decoder level), not by exact pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SizeClass {
    /// An audio job: no picture.
    Audio,
    /// Up to 640×480 pixels.
    Sd,
    /// Up to 1280×720 pixels.
    Hd,
    /// Up to 1920×1080 pixels.
    FullHd,
    /// Up to 3840×2160 pixels.
    Uhd,
    /// More than 3840×2160 pixels.
    Larger,
}

impl SizeClass {
    /// The class of a `width`×`height` picture.
    #[must_use]
    pub fn of(width: u32, height: u32) -> Self {
        let pixels = u64::from(width) * u64::from(height);
        match pixels {
            0..=307_200 => Self::Sd,
            307_201..=921_600 => Self::Hd,
            921_601..=2_073_600 => Self::FullHd,
            2_073_601..=8_294_400 => Self::Uhd,
            _ => Self::Larger,
        }
    }

    fn token(self) -> &'static str {
        match self {
            Self::Audio => "audio",
            Self::Sd => "sd",
            Self::Hd => "720p",
            Self::FullHd => "1080p",
            Self::Uhd => "2160p",
            Self::Larger => "larger",
        }
    }
}

/// The parts of a codec configuration that decide which backends can
/// take it, as far as remembering a winner goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfigTraits {
    /// The picture size class.
    pub size: SizeClass,
    /// Bits per sample of the coded stream or the requested output.
    pub bit_depth: BitDepth,
    /// Where decoded pictures are asked for.
    pub memory: FrameMemory,
}

impl ConfigTraits {
    /// An audio job's traits.
    pub const AUDIO: Self = Self {
        size: SizeClass::Audio,
        bit_depth: BitDepth::Eight,
        memory: FrameMemory::Cpu,
    };
}

/// A codec configuration a [`SelectionKey`] can be built from.
pub trait SelectionInput {
    /// The traits of this configuration that matter for selection.
    fn selection_traits(&self) -> ConfigTraits;
}

impl SelectionInput for EncoderConfig {
    fn selection_traits(&self) -> ConfigTraits {
        ConfigTraits {
            size: SizeClass::of(self.width, self.height),
            bit_depth: self.bit_depth,
            memory: FrameMemory::Cpu,
        }
    }
}

impl SelectionInput for DecoderConfig {
    fn selection_traits(&self) -> ConfigTraits {
        ConfigTraits {
            size: SizeClass::of(self.width, self.height),
            bit_depth: self.pixel_format.bit_depth(),
            memory: self.memory,
        }
    }
}

impl SelectionInput for AudioEncoderConfig {
    fn selection_traits(&self) -> ConfigTraits {
        ConfigTraits::AUDIO
    }
}

impl SelectionInput for AudioDecoderConfig {
    fn selection_traits(&self) -> ConfigTraits {
        ConfigTraits::AUDIO
    }
}

/// What a remembered winner is filed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SelectionKey {
    /// The job.
    pub target: CodecTarget,
    /// The automatic selection's policy: another policy may prefer
    /// another backend.
    pub policy: AccelerationPolicy,
    /// The configuration's traits.
    pub traits: ConfigTraits,
    /// The most preferred GPU (`GpuPlacement::key`): a winner on one GPU
    /// says nothing about another. `None` without GPUs.
    pub gpu: Option<DevNumber>,
}

impl SelectionKey {
    /// A stable one-word text form, the key [`FileCache`] stores.
    #[must_use]
    pub fn token(&self) -> String {
        // `FERRICAST_H264_DECODE_BACKEND` → `h264_decode`: already a
        // stable name for every job.
        let job = self.target.env_var();
        let job = job
            .strip_prefix("FERRICAST_")
            .and_then(|rest| rest.strip_suffix("_BACKEND"))
            .unwrap_or(&job)
            .to_ascii_lowercase();
        let policy = match self.policy {
            AccelerationPolicy::PreferHardware => "prefer-hardware",
            AccelerationPolicy::PreferSoftware => "prefer-software",
            AccelerationPolicy::HardwareOnly => "hardware-only",
            AccelerationPolicy::SoftwareOnly => "software-only",
        };
        let depth = match self.traits.bit_depth {
            BitDepth::Eight => "8bit",
            BitDepth::Ten => "10bit",
        };
        let memory = match self.traits.memory {
            FrameMemory::Cpu => "cpu",
            FrameMemory::Native => "native",
        };
        // No GPU segment without a GPU: such keys read as before.
        let gpu = self
            .gpu
            .map(|dev| format!("/gpu-{}-{}", dev.major, dev.minor))
            .unwrap_or_default();
        format!(
            "{job}/{policy}/{}/{depth}/{memory}{gpu}",
            self.traits.size.token()
        )
    }
}

impl fmt::Display for SelectionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.token())
    }
}

/// Remembers the backend automatic selection opened, per key.
///
/// Implementations must never fail the codec: a store that cannot be
/// read or written behaves as empty and logs why.
pub trait SelectionCache: Send + Sync + fmt::Debug {
    /// The backend that last opened for `key`.
    fn winner(&self, key: &SelectionKey) -> Option<BackendId>;
    /// `backend` opened for `key`.
    fn record(&self, key: &SelectionKey, backend: BackendId);
    /// `backend` failed for `key`; forget it if it is the winner.
    fn evict(&self, key: &SelectionKey, backend: BackendId);
}

/// Remembers nothing: automatic selection always walks its full order.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCache;

impl SelectionCache for NoCache {
    fn winner(&self, _: &SelectionKey) -> Option<BackendId> {
        None
    }
    fn record(&self, _: &SelectionKey, _: BackendId) {}
    fn evict(&self, _: &SelectionKey, _: BackendId) {}
}

/// Remembers winners for the life of the process.
#[derive(Debug, Default)]
pub struct MemoryCache {
    winners: Mutex<HashMap<SelectionKey, BackendId>>,
}

impl SelectionCache for MemoryCache {
    fn winner(&self, key: &SelectionKey) -> Option<BackendId> {
        lock(&self.winners).get(key).copied()
    }

    fn record(&self, key: &SelectionKey, backend: BackendId) {
        lock(&self.winners).insert(*key, backend);
    }

    fn evict(&self, key: &SelectionKey, backend: BackendId) {
        let mut winners = lock(&self.winners);
        if winners.get(key) == Some(&backend) {
            winners.remove(key);
        }
    }
}

/// Remembers winners in a file, so they survive restarts.
///
/// The file is plain text: a version header, then one `key backend`
/// line per winner (see [`SelectionKey::token`]). It is read once when
/// the cache opens and rewritten whole (to a temporary file, then
/// renamed over the old one) whenever a winner changes. A missing,
/// unreadable, foreign-version or malformed file is treated as empty and
/// logged; a write that fails is logged and the cache keeps working in
/// memory.
#[derive(Debug)]
pub struct FileCache {
    path: PathBuf,
    winners: Mutex<HashMap<String, BackendId>>,
}

impl FileCache {
    /// The header line of the current format.
    pub const HEADER: &'static str = "# ferricast-codec selection cache v1";

    /// Opens the cache stored at `path`, reading what it holds.
    #[must_use]
    pub fn open(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let winners = match std::fs::read_to_string(&path) {
            Ok(text) => Self::parse(&path, &text),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => HashMap::new(),
            Err(error) => {
                warn!(path = %path.display(), %error, "selection cache unreadable; starting empty");
                HashMap::new()
            }
        };
        Self {
            path,
            winners: Mutex::new(winners),
        }
    }

    /// Where the cache is stored.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn parse(path: &Path, text: &str) -> HashMap<String, BackendId> {
        let mut lines = text.lines();
        if lines.next().map(str::trim) != Some(Self::HEADER) {
            warn!(path = %path.display(), "selection cache has an unknown format; ignoring it");
            return HashMap::new();
        }
        let mut winners = HashMap::new();
        for line in lines.map(str::trim).filter(|line| !line.is_empty()) {
            let parsed = line
                .split_once(' ')
                .and_then(|(key, backend)| Some((key, backend.trim().parse().ok()?)));
            match parsed {
                Some((key, backend)) => {
                    winners.insert(key.to_owned(), backend);
                }
                None => {
                    warn!(path = %path.display(), line, "selection cache line ignored");
                }
            }
        }
        winners
    }

    fn persist(&self, winners: &HashMap<String, BackendId>) {
        let mut entries: Vec<_> = winners.iter().collect();
        entries.sort();
        let mut text = format!("{}\n", Self::HEADER);
        for (key, backend) in entries {
            text.push_str(key);
            text.push(' ');
            text.push_str(backend.name());
            text.push('\n');
        }
        if let Err(error) = self.write_atomically(text.as_bytes()) {
            warn!(path = %self.path.display(), %error, "selection cache not saved");
        }
    }

    fn write_atomically(&self, bytes: &[u8]) -> std::io::Result<()> {
        if let Some(dir) = self.path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut temporary = self.path.clone().into_os_string();
        temporary.push(format!(".{}.tmp", std::process::id()));
        let temporary = PathBuf::from(temporary);
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, &self.path).inspect_err(|_| {
            std::fs::remove_file(&temporary).ok();
        })
    }
}

impl SelectionCache for FileCache {
    fn winner(&self, key: &SelectionKey) -> Option<BackendId> {
        lock(&self.winners).get(&key.token()).copied()
    }

    fn record(&self, key: &SelectionKey, backend: BackendId) {
        let mut winners = lock(&self.winners);
        if winners.insert(key.token(), backend) != Some(backend) {
            self.persist(&winners);
        }
    }

    fn evict(&self, key: &SelectionKey, backend: BackendId) {
        let mut winners = lock(&self.winners);
        let token = key.token();
        if winners.get(&token) == Some(&backend) {
            winners.remove(&token);
            self.persist(&winners);
        }
    }
}

/// The cache `FERRICAST_CODEC_CACHE` asks for (see the module docs).
#[must_use]
pub fn from_env() -> Arc<dyn SelectionCache> {
    match std::env::var_os(CACHE_ENV_VAR) {
        None => Arc::new(MemoryCache::default()),
        Some(value) => match value.to_str().map(str::trim) {
            Some("" | "memory") => Arc::new(MemoryCache::default()),
            Some("off" | "none") => Arc::new(NoCache),
            _ => {
                debug!(path = ?value, "selection cache stored in a file");
                Arc::new(FileCache::open(value))
            }
        },
    }
}

static GLOBAL: LazyLock<RwLock<Arc<dyn SelectionCache>>> =
    LazyLock::new(|| RwLock::new(from_env()));

/// The process-wide cache: the one [`install`]ed, else the one
/// `FERRICAST_CODEC_CACHE` picked on first use.
#[must_use]
pub fn global() -> Arc<dyn SelectionCache> {
    Arc::clone(&GLOBAL.read().unwrap_or_else(PoisonError::into_inner))
}

/// Replaces the process-wide cache for every registry without its own.
pub fn install(cache: Arc<dyn SelectionCache>) {
    *GLOBAL.write().unwrap_or_else(PoisonError::into_inner) = cache;
}

/// A probe verdict per backend and job.
type ProbeVerdicts = HashMap<(BackendId, CodecTarget), Result<(), CodecError>>;

/// Probe verdicts for the life of the process.
static PROBES: LazyLock<Mutex<ProbeVerdicts>> = LazyLock::new(Mutex::default);

/// The verdict of `probe` for `backend` on `target`, running it only
/// the first time.
pub(crate) fn probe_once(
    backend: BackendId,
    target: CodecTarget,
    probe: impl FnOnce() -> Result<(), CodecError>,
) -> Result<(), CodecError> {
    if let Some(verdict) = lock(&PROBES).get(&(backend, target)) {
        return verdict.clone();
    }
    // Probed outside the lock: a slow driver query must not stall other
    // jobs' probes. Two racing first probes both run; either verdict is right.
    let verdict = probe();
    // A busy backend (a full GPU, a driver that failed to start this time)
    // is not unavailable: the next selection probes it again.
    if !matches!(verdict, Err(CodecError::BackendBusy { .. })) {
        lock(&PROBES).insert((backend, target), verdict.clone());
    }
    verdict
}

/// The memoized verdict of a probe of `backend` on `target` that failed,
/// without running one: opening is the definitive test, so the open path
/// only skips a backend a capability report already found missing.
pub(crate) fn known_unavailable(backend: BackendId, target: CodecTarget) -> Option<CodecError> {
    match lock(&PROBES).get(&(backend, target)) {
        Some(Err(error)) => Some(error.clone()),
        Some(Ok(())) | None => None,
    }
}

/// Forgets the probe verdict of `backend` on `target`.
pub(crate) fn forget_probe(backend: BackendId, target: CodecTarget) {
    lock(&PROBES).remove(&(backend, target));
}

/// Forgets every probe verdict, so the next selection probes again
/// (after a GPU was plugged in or a driver installed).
pub fn reset_probes() {
    lock(&PROBES).clear();
}

/// The maps here hold plain values a panic cannot leave half-written.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use ferricast_core::Codec;

    use super::*;

    fn key(size: SizeClass) -> SelectionKey {
        SelectionKey {
            target: CodecTarget::decode(Codec::H264),
            policy: AccelerationPolicy::PreferHardware,
            traits: ConfigTraits {
                size,
                bit_depth: BitDepth::Eight,
                memory: FrameMemory::Cpu,
            },
            gpu: None,
        }
    }

    #[test]
    fn each_gpu_remembers_its_own_winner() {
        let cache = MemoryCache::default();
        let on = |minor| SelectionKey {
            gpu: Some(DevNumber { major: 226, minor }),
            ..key(SizeClass::FullHd)
        };
        cache.record(&on(128), BackendId::Vaapi);
        cache.record(&on(129), BackendId::Nvdec);
        assert_eq!(cache.winner(&on(128)), Some(BackendId::Vaapi));
        assert_eq!(cache.winner(&on(129)), Some(BackendId::Nvdec));
        assert_eq!(cache.winner(&key(SizeClass::FullHd)), None);
        assert_eq!(
            on(129).token(),
            "h264_decode/prefer-hardware/1080p/8bit/cpu/gpu-226-129"
        );
    }

    #[test]
    fn sizes_fall_into_classes() {
        assert_eq!(SizeClass::of(640, 480), SizeClass::Sd);
        assert_eq!(SizeClass::of(1280, 720), SizeClass::Hd);
        assert_eq!(SizeClass::of(1920, 1080), SizeClass::FullHd);
        assert_eq!(SizeClass::of(3840, 2160), SizeClass::Uhd);
        assert_eq!(SizeClass::of(7680, 4320), SizeClass::Larger);
        assert_eq!(
            key(SizeClass::FullHd).token(),
            "h264_decode/prefer-hardware/1080p/8bit/cpu"
        );
    }

    #[test]
    fn eviction_only_forgets_the_failing_winner() {
        let cache = MemoryCache::default();
        let key = key(SizeClass::FullHd);
        cache.record(&key, BackendId::Vaapi);
        cache.evict(&key, BackendId::Nvdec);
        assert_eq!(cache.winner(&key), Some(BackendId::Vaapi));
        cache.evict(&key, BackendId::Vaapi);
        assert_eq!(cache.winner(&key), None);
    }

    #[test]
    fn a_file_cache_survives_reopening_and_ignores_garbage() {
        let dir = std::env::temp_dir().join(format!("ferricast-cache-{}", std::process::id()));
        let path = dir.join("nested").join("selection");
        let cache = FileCache::open(&path);
        cache.record(&key(SizeClass::FullHd), BackendId::Vaapi);
        cache.record(&key(SizeClass::Uhd), BackendId::OpenH264);
        let reopened = FileCache::open(&path);
        assert_eq!(
            reopened.winner(&key(SizeClass::FullHd)),
            Some(BackendId::Vaapi)
        );
        reopened.evict(&key(SizeClass::Uhd), BackendId::OpenH264);
        assert_eq!(FileCache::open(&path).winner(&key(SizeClass::Uhd)), None);

        std::fs::write(&path, "not a cache\nh264_decode/x vaapi\n").unwrap();
        assert_eq!(FileCache::open(&path).winner(&key(SizeClass::FullHd)), None);
        let header = FileCache::HEADER;
        std::fs::write(
            &path,
            format!("{header}\nh264_decode/prefer-hardware/1080p/8bit/cpu ffmpeg\nbroken\n"),
        )
        .unwrap();
        assert_eq!(FileCache::open(&path).winner(&key(SizeClass::FullHd)), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn probes_run_once_until_forgotten() {
        // A job no backend does, so no other test shares the verdict.
        let target = CodecTarget::encode(Codec::Av1);
        let mut runs = 0;
        for _ in 0..3 {
            probe_once(BackendId::Pcm, target, || {
                runs += 1;
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(runs, 1);
        forget_probe(BackendId::Pcm, target);
        probe_once(BackendId::Pcm, target, || {
            runs += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(runs, 2);
    }

    #[test]
    fn a_busy_backend_is_probed_again() {
        // A job no backend does, so no other test shares the verdict.
        let target = CodecTarget::decode(Codec::Av1);
        let mut runs = 0;
        for _ in 0..3 {
            let verdict = probe_once(BackendId::Pcm, target, || {
                runs += 1;
                Err(CodecError::BackendBusy {
                    backend: BackendId::Pcm,
                    reason: "session limit".into(),
                })
            });
            assert!(matches!(verdict, Err(CodecError::BackendBusy { .. })));
            assert!(known_unavailable(BackendId::Pcm, target).is_none());
        }
        assert_eq!(runs, 3);
    }
}
