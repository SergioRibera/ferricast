//! Which GPU the hardware codecs run on.
//!
//! A machine can have several GPUs: a hybrid laptop's integrated GPU
//! (which usually drives the panel and renders the desktop) and a
//! discrete one, or a desktop with two cards. [`GpuPreference`] is the
//! user's choice; [`Gpus::scan`] lists the GPUs from sysfs (Linux and
//! Android; elsewhere the list is empty), and [`GpuPlacement`] turns a
//! preference into the GPUs in the order the hardware backends should
//! use them.
//!
//! Integrated or discrete is not something the kernel reports, so
//! [`GpuKind`] is a heuristic (see [`Gpus::classify`]); a GPU it cannot
//! place is [`GpuKind::Unknown`], and [`GpuPreference::Device`] names a
//! GPU exactly when the heuristic is wrong.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// The environment variable that overrides [`GpuPreference::Auto`]
/// without rebuilding; it takes the text [`GpuPreference`] parses.
pub const GPU_ENV_VAR: &str = "FERRICAST_GPU";

const NVIDIA: u16 = 0x10de;
const AMD: u16 = 0x1002;
const INTEL: u16 = 0x8086;

/// A PCI address, `domain:bus:device.function` (`0000:01:00.0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct PciAddress {
    /// The PCI domain (segment).
    pub domain: u32,
    /// The bus number.
    pub bus: u8,
    /// The device number (0-31).
    pub device: u8,
    /// The function number (0-7).
    pub function: u8,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

impl FromStr for PciAddress {
    type Err = InvalidGpuText;

    /// `0000:01:00.0`, or `01:00.0` (domain 0), hexadecimal, any case.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = |reason: &str| InvalidGpuText {
            reason: format!("`{text}` is not a PCI address: {reason}"),
        };
        let (rest, function) = text
            .rsplit_once('.')
            .ok_or_else(|| invalid("no `.function`"))?;
        let parts: Vec<&str> = rest.split(':').collect();
        let (domain, bus, device) = match parts.as_slice() {
            [domain, bus, device] => (*domain, *bus, *device),
            [bus, device] => ("0", *bus, *device),
            _ => return Err(invalid("expected `domain:bus:device.function`")),
        };
        let hex = |part: &str, what: &str| {
            u32::from_str_radix(part, 16).map_err(|_| invalid(&format!("bad {what} `{part}`")))
        };
        let narrow = |value: u32, max: u32, what: &str| {
            u8::try_from(value)
                .ok()
                .filter(|&v| u32::from(v) <= max)
                .ok_or_else(|| invalid(&format!("{what} {value:#x} out of range")))
        };
        Ok(Self {
            domain: hex(domain, "domain")?,
            bus: narrow(hex(bus, "bus")?, 0xff, "bus")?,
            device: narrow(hex(device, "device")?, 0x1f, "device")?,
            function: narrow(hex(function, "function")?, 7, "function")?,
        })
    }
}

/// A character device number, `major:minor` (a DRM node's `dev_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct DevNumber {
    /// The major number (226 for DRM).
    pub major: u32,
    /// The minor number (128 and up for render nodes).
    pub minor: u32,
}

impl DevNumber {
    /// From a Linux `dev_t` (glibc's `gnu_dev_major` / `gnu_dev_minor`
    /// encoding, which the kernel's `new_encode_dev` matches).
    #[must_use]
    pub const fn from_dev(dev: u64) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "each half is masked to 32 bits"
        )]
        Self {
            major: (((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0x0000_0fff)) as u32,
            minor: (((dev >> 12) & 0xffff_ff00) | (dev & 0x0000_00ff)) as u32,
        }
    }

    /// The Linux `dev_t` (glibc's `gnu_dev_makedev`).
    #[must_use]
    pub const fn dev(self) -> u64 {
        let (major, minor) = (self.major as u64, self.minor as u64);
        ((major & 0xffff_f000) << 32)
            | ((major & 0x0000_0fff) << 8)
            | ((minor & 0xffff_ff00) << 12)
            | (minor & 0x0000_00ff)
    }
}

impl fmt::Display for DevNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.major, self.minor)
    }
}

impl FromStr for DevNumber {
    type Err = InvalidGpuText;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid = || InvalidGpuText {
            reason: format!("`{text}` is not a device number (`major:minor`)"),
        };
        let (major, minor) = text.trim().split_once(':').ok_or_else(invalid)?;
        Ok(Self {
            major: major.parse().map_err(|_| invalid())?,
            minor: minor.parse().map_err(|_| invalid())?,
        })
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl From<crate::DrmDevice> for DevNumber {
    fn from(device: crate::DrmDevice) -> Self {
        Self::from_dev(device.dev())
    }
}

/// One GPU, named the way a user or a frame names it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum GpuId {
    /// A DRM node path: `/dev/dri/renderD129` (a `card*` primary node
    /// names its GPU too). `renderD129` alone means `/dev/dri/renderD129`.
    Node(String),
    /// The GPU's PCI address (`0000:01:00.0`): stable across reboots,
    /// unlike render node numbers.
    Pci(PciAddress),
    /// A DRM node's device number, what a captured frame's producer is
    /// (`GpuFrame::producer`).
    Dev(DevNumber),
}

impl fmt::Display for GpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node(path) => f.write_str(path),
            Self::Pci(address) => address.fmt(f),
            Self::Dev(dev) => dev.fmt(f),
        }
    }
}

impl FromStr for GpuId {
    type Err = InvalidGpuText;

    /// A path (`/dev/dri/renderD129`), a node name (`renderD129`,
    /// `card1`), a PCI address (`0000:01:00.0`, `01:00.0`) or a device
    /// number (`226:129`).
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        if text.starts_with('/') {
            return Ok(Self::Node(text.to_owned()));
        }
        if text.starts_with("renderD") || text.starts_with("card") {
            return Ok(Self::Node(format!("/dev/dri/{text}")));
        }
        if text.contains('.') {
            return text.parse().map(Self::Pci);
        }
        if text.contains(':') {
            return text.parse().map(Self::Dev);
        }
        Err(InvalidGpuText {
            reason: format!(
                "`{text}` names no GPU: expected a DRM node (`/dev/dri/renderD129`), a PCI \
                 address (`0000:01:00.0`) or a device number (`226:129`)"
            ),
        })
    }
}

/// Which GPU the hardware encoders and decoders run on.
///
/// | Variant | Hardware backends use | When nothing matches |
/// |---|---|---|
/// | `Auto` | the GPU that renders the captured frames (zero-copy), then discrete GPUs, then the rest | any GPU |
/// | `SameAsCapture` | only the GPU that renders the captured frames | without a known producer: as `Auto` |
/// | `Integrated` | integrated GPUs first, then the rest | any GPU |
/// | `Discrete` | discrete GPUs first, then the rest | any GPU |
/// | `Device(id)` | only that GPU | no hardware backend opens (software takes over) |
///
/// Automatic backend selection orders the hardware backends by it (a
/// backend that runs on the first GPU of the order is tried first);
/// each hardware backend opens on the first GPU of the order it can run
/// on. A `Fixed` backend selection keeps its backend but still opens it
/// on the preferred GPU.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum GpuPreference {
    /// The capture's GPU, else a discrete one: no copy between GPUs
    /// when the frames come from a GPU that encodes, the faster GPU
    /// otherwise.
    #[default]
    Auto,
    /// Only the GPU that renders the captured frames.
    SameAsCapture,
    /// Integrated GPUs first (a hybrid laptop's power-saving choice: the
    /// discrete GPU can stay asleep).
    Integrated,
    /// Discrete GPUs first (encode throughput).
    Discrete,
    /// Exactly this GPU.
    Device(GpuId),
}

impl fmt::Display for GpuPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::SameAsCapture => f.write_str("same-as-capture"),
            Self::Integrated => f.write_str("integrated"),
            Self::Discrete => f.write_str("discrete"),
            Self::Device(id) => id.fmt(f),
        }
    }
}

impl FromStr for GpuPreference {
    type Err = InvalidGpuText;

    /// `auto`, `same-as-capture`, `integrated`, `discrete`, or a GPU as
    /// [`GpuId`] parses it.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let trimmed = text.trim();
        match trimmed.to_ascii_lowercase().replace('_', "-").as_str() {
            "" | "auto" => Ok(Self::Auto),
            "same-as-capture" | "capture" => Ok(Self::SameAsCapture),
            "integrated" | "igpu" => Ok(Self::Integrated),
            "discrete" | "dgpu" => Ok(Self::Discrete),
            _ => trimmed
                .parse()
                .map(Self::Device)
                .map_err(|error| InvalidGpuText {
                    reason: format!(
                        "{}; or one of `auto`, `same-as-capture`, `integrated`, `discrete`",
                        error.reason
                    ),
                }),
        }
    }
}

impl GpuPreference {
    /// What the codecs actually use: `self`, unless it is [`Self::Auto`]
    /// and [`GPU_ENV_VAR`] is set, which then wins.
    ///
    /// # Errors
    /// [`GpuError::InvalidEnv`] when the variable holds text that is not a
    /// preference.
    pub fn resolve_env(&self) -> Result<Self, GpuError> {
        if *self != Self::Auto {
            return Ok(self.clone());
        }
        match std::env::var(GPU_ENV_VAR) {
            Ok(value) if !value.trim().is_empty() => {
                value
                    .parse()
                    .map_err(|error: InvalidGpuText| GpuError::InvalidEnv {
                        variable: GPU_ENV_VAR,
                        value,
                        reason: error.reason,
                    })
            }
            _ => Ok(Self::Auto),
        }
    }
}

/// Why text is not a [`GpuPreference`], [`GpuId`] or [`PciAddress`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{reason}")]
pub struct InvalidGpuText {
    reason: String,
}

/// Why a GPU preference cannot be followed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpuError {
    /// The preference names a GPU this machine does not have.
    #[error("the preferred GPU {id} is not on this machine{}", list_hint(.available))]
    NotFound {
        /// The GPU asked for.
        id: GpuId,
        /// The render nodes found, for the message.
        available: Vec<String>,
    },
    /// The preferred GPU exists, but the backend cannot run on it.
    #[error("the preferred GPU {gpu} ({vendor}) cannot run {what}")]
    Unusable {
        /// The GPU's render node.
        gpu: String,
        /// Its vendor.
        vendor: GpuVendor,
        /// What it lacks (`an NVIDIA backend`).
        what: &'static str,
    },
    /// [`GPU_ENV_VAR`] does not parse.
    #[error("the environment variable {variable}={value:?} is not a GPU preference: {reason}")]
    InvalidEnv {
        /// The variable.
        variable: &'static str,
        /// Its value.
        value: String,
        /// Why it does not parse.
        reason: String,
    },
}

fn list_hint(available: &[String]) -> String {
    if available.is_empty() {
        " (no GPU found in sysfs)".to_owned()
    } else {
        format!(" (found: {})", available.join(", "))
    }
}

/// A PCI vendor id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct GpuVendor(pub u16);

impl GpuVendor {
    /// NVIDIA.
    pub const NVIDIA: Self = Self(NVIDIA);
    /// AMD (ATI).
    pub const AMD: Self = Self(AMD);
    /// Intel.
    pub const INTEL: Self = Self(INTEL);

    /// The vendor's name, `None` for one ferricast does not know.
    #[must_use]
    pub const fn name(self) -> Option<&'static str> {
        match self.0 {
            NVIDIA => Some("NVIDIA"),
            AMD => Some("AMD"),
            INTEL => Some("Intel"),
            _ => None,
        }
    }
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => f.write_str(name),
            None => write!(f, "{:#06x}", self.0),
        }
    }
}

/// Whether a GPU is part of the CPU package or a card of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub enum GpuKind {
    /// In the CPU package, sharing system memory.
    Integrated,
    /// A card (or laptop chip) with its own memory.
    Discrete,
    /// The heuristic cannot tell (a lone AMD GPU: APU or card).
    Unknown,
}

impl fmt::Display for GpuKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Integrated => "integrated",
            Self::Discrete => "discrete",
            Self::Unknown => "unknown",
        })
    }
}

/// Which GPUs a hardware backend runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuAffinity {
    /// NVIDIA GPUs only (NVENC, NVDEC: CUDA).
    Nvidia,
    /// Every GPU but NVIDIA's (VA-API encode: the NVIDIA VA-API driver
    /// decodes only).
    NotNvidia,
    /// Any GPU with a DRM render node (VA-API decode).
    Any,
}

impl GpuAffinity {
    /// Whether a backend with this affinity runs on `gpu`.
    #[must_use]
    pub fn accepts(self, gpu: &Gpu) -> bool {
        match self {
            Self::Nvidia => gpu.vendor == GpuVendor::NVIDIA,
            Self::NotNvidia => gpu.vendor != GpuVendor::NVIDIA,
            Self::Any => true,
        }
    }

    fn needs(self) -> &'static str {
        match self {
            Self::Nvidia => "an NVIDIA-only backend",
            Self::NotNvidia => "a backend NVIDIA's driver does not provide",
            Self::Any => "this backend",
        }
    }
}

/// One GPU with a DRM render node, as sysfs describes it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "wire", derive(wincode::SchemaWrite, wincode::SchemaRead))]
pub struct Gpu {
    /// The render node (`/dev/dri/renderD128`).
    pub render_node: String,
    /// The render node's device number.
    pub render_dev: DevNumber,
    /// The primary node's (`/dev/dri/card1`) device number, when sysfs
    /// lists one.
    pub primary_dev: Option<DevNumber>,
    /// The PCI address; `None` for a GPU that is not a PCI device.
    pub pci: Option<PciAddress>,
    /// The PCI vendor.
    pub vendor: GpuVendor,
    /// The PCI device id.
    pub device_id: u16,
    /// The kernel driver (`i915`, `xe`, `amdgpu`, `nvidia`).
    pub driver: Option<String>,
    /// Whether the firmware initialised it as the boot display
    /// (`boot_vga`).
    pub boot_vga: bool,
    /// Integrated or discrete, by [`Gpus::classify`]'s heuristic.
    pub kind: GpuKind,
}

impl Gpu {
    /// Whether `id` names this GPU.
    #[must_use]
    pub fn matches(&self, id: &GpuId) -> bool {
        match id {
            GpuId::Node(path) => {
                *path == self.render_node || node_dev(path).is_some_and(|dev| self.has_node(dev))
            }
            GpuId::Pci(address) => self.pci == Some(*address),
            GpuId::Dev(dev) => self.has_node(*dev),
        }
    }

    fn has_node(&self, dev: DevNumber) -> bool {
        self.render_dev == dev || self.primary_dev == Some(dev)
    }

    /// A one-line description: `NVIDIA 10de:2504 (nvidia, discrete)`.
    #[must_use]
    pub fn label(&self) -> String {
        format!(
            "{} {:04x}:{:04x} ({}, {})",
            self.vendor,
            self.vendor.0,
            self.device_id,
            self.driver.as_deref().unwrap_or("no driver"),
            self.kind
        )
    }
}

/// The device number of the node at `path`, from the file system.
#[cfg(unix)]
fn node_dev(path: &str) -> Option<DevNumber> {
    use std::os::unix::fs::MetadataExt as _;
    std::fs::metadata(path)
        .ok()
        .map(|metadata| DevNumber::from_dev(metadata.rdev()))
}

/// No device nodes off Unix.
#[cfg(not(unix))]
fn node_dev(_path: &str) -> Option<DevNumber> {
    None
}

/// The GPUs of this machine, in render node order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Gpus(Vec<Gpu>);

impl Gpus {
    /// The GPUs sysfs lists (`/sys/class/drm/renderD*`); empty where
    /// there is no sysfs (Windows, Apple) or no DRM GPU.
    #[must_use]
    pub fn scan() -> Self {
        Self::scan_at(Path::new("/sys"), Path::new("/dev"))
    }

    /// [`Self::scan`] against another sysfs and `/dev` root (tests).
    #[must_use]
    pub fn scan_at(sys: &Path, dev: &Path) -> Self {
        let class = sys.join("class/drm");
        let mut nodes: Vec<(u32, PathBuf, String)> = std::fs::read_dir(&class)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                let number = name.strip_prefix("renderD")?.parse().ok()?;
                Some((number, entry.path(), name))
            })
            .collect();
        nodes.sort_by_key(|(number, ..)| *number);
        let gpus = nodes
            .into_iter()
            .filter_map(|(_, path, name)| read_gpu(&path, &dev.join("dri").join(name)))
            .collect();
        Self::classify(gpus)
    }

    /// Sets each GPU's [`GpuKind`]. The kernel has no such attribute, so:
    ///
    /// 1. NVIDIA: discrete (its only integrated GPUs, Tegra, are not PCI).
    /// 2. Intel: integrated on PCI bus 0 (Intel's integrated GPUs sit at
    ///    `0000:00:02.0`), discrete elsewhere (Arc cards sit behind a
    ///    `PCIe` port).
    /// 3. The rest (AMD, others), undecided by vendor:
    ///    - discrete when another GPU is already integrated (a machine has
    ///      one integrated GPU);
    ///    - else, when the only undecided GPU sits next to a discrete one,
    ///      integrated (an AMD APU in a laptop with an NVIDIA or Arc GPU);
    ///    - else, among several undecided GPUs, the boot display
    ///      (`boot_vga`) is integrated and the others discrete (an AMD
    ///      APU with an AMD discrete GPU: firmware boots on the panel's
    ///      GPU);
    ///    - else (a lone AMD GPU) unknown: APU and card look alike in
    ///      sysfs.
    #[must_use]
    pub fn classify(mut gpus: Vec<Gpu>) -> Self {
        for gpu in &mut gpus {
            gpu.kind = match gpu.vendor {
                GpuVendor::NVIDIA => GpuKind::Discrete,
                GpuVendor::INTEL => match gpu.pci {
                    Some(PciAddress { bus: 0, .. }) => GpuKind::Integrated,
                    Some(_) => GpuKind::Discrete,
                    None => GpuKind::Unknown,
                },
                _ => GpuKind::Unknown,
            };
        }
        let count = |kind| gpus.iter().filter(|gpu| gpu.kind == kind).count();
        let undecided = count(GpuKind::Unknown);
        let decide = |gpu: &Gpu| {
            if count(GpuKind::Integrated) > 0 {
                GpuKind::Discrete
            } else if undecided == 1 && count(GpuKind::Discrete) > 0 {
                GpuKind::Integrated
            } else if undecided > 1 {
                if gpu.boot_vga {
                    GpuKind::Integrated
                } else {
                    GpuKind::Discrete
                }
            } else {
                GpuKind::Unknown
            }
        };
        let kinds: Vec<GpuKind> = gpus
            .iter()
            .map(|gpu| match gpu.kind {
                GpuKind::Unknown => decide(gpu),
                known => known,
            })
            .collect();
        for (gpu, kind) in gpus.iter_mut().zip(kinds) {
            gpu.kind = kind;
        }
        Self(gpus)
    }

    /// The GPUs.
    #[must_use]
    pub fn as_slice(&self) -> &[Gpu] {
        &self.0
    }

    /// The GPU `id` names.
    #[must_use]
    pub fn find(&self, id: &GpuId) -> Option<&Gpu> {
        self.0.iter().find(|gpu| gpu.matches(id))
    }
}

impl From<Vec<Gpu>> for Gpus {
    fn from(gpus: Vec<Gpu>) -> Self {
        Self(gpus)
    }
}

impl IntoIterator for Gpus {
    type Item = Gpu;
    type IntoIter = std::vec::IntoIter<Gpu>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// One `/sys/class/drm/renderD*` entry; `None` without a readable
/// device number.
fn read_gpu(class_entry: &Path, render_node: &Path) -> Option<Gpu> {
    let read = |path: PathBuf| std::fs::read_to_string(path).ok();
    let render_dev = read(class_entry.join("dev"))?.parse().ok()?;
    let device = class_entry.join("device");
    let uevent = read(device.join("uevent")).unwrap_or_default();
    let field = |name: &str| {
        uevent
            .lines()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix('='))
            .map(str::trim)
    };
    let hex16 = |text: &str| u16::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok();
    let pci_id = field("PCI_ID").and_then(|id| id.split_once(':'));
    let vendor = read(device.join("vendor"))
        .and_then(|text| hex16(&text))
        .or_else(|| pci_id.and_then(|(vendor, _)| hex16(vendor)))
        .unwrap_or(0);
    let device_id = read(device.join("device"))
        .and_then(|text| hex16(&text))
        .or_else(|| pci_id.and_then(|(_, id)| hex16(id)))
        .unwrap_or(0);
    let primary_dev = std::fs::read_dir(device.join("drm"))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("card")
                .is_some_and(|n| n.chars().all(|c| c.is_ascii_digit()))
        })
        .find_map(|entry| read(entry.path().join("dev"))?.parse().ok());
    Some(Gpu {
        render_node: render_node.to_string_lossy().into_owned(),
        render_dev,
        primary_dev,
        pci: field("PCI_SLOT_NAME").and_then(|slot| slot.parse().ok()),
        vendor: GpuVendor(vendor),
        device_id,
        driver: field("DRIVER").map(str::to_owned),
        boot_vga: read(device.join("boot_vga")).is_some_and(|text| text.trim() == "1"),
        kind: GpuKind::Unknown,
    })
}

/// A [`GpuPreference`] applied to this machine's GPUs: the GPUs hardware
/// backends may use, best first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuPlacement {
    ranked: Vec<Gpu>,
    /// The GPU the preference is limited to, found or not.
    pinned: Option<GpuId>,
    /// Every render node found, for messages.
    available: Vec<String>,
}

impl GpuPlacement {
    /// Applies `preference` (after [`GpuPreference::resolve_env`]) to the
    /// GPUs [`Gpus::scan`] finds; `source` is the GPU the frames come
    /// from, when known.
    ///
    /// # Errors
    /// [`GpuError::InvalidEnv`] from the environment override.
    pub fn resolve(preference: &GpuPreference, source: Option<&GpuId>) -> Result<Self, GpuError> {
        let preference = preference.resolve_env()?;
        Ok(Self::resolve_in(&preference, source, &Gpus::scan()))
    }

    /// Applies `preference` to `gpus`, ignoring the environment.
    #[must_use]
    pub fn resolve_in(preference: &GpuPreference, source: Option<&GpuId>, gpus: &Gpus) -> Self {
        let available = gpus.0.iter().map(|gpu| gpu.render_node.clone()).collect();
        let source_gpu = source.and_then(|id| gpus.find(id));
        let pinned = |id: &GpuId, available: Vec<String>| Self {
            ranked: gpus.find(id).cloned().into_iter().collect(),
            pinned: Some(id.clone()),
            available,
        };
        let by_kind = |kinds: [GpuKind; 3]| -> Vec<Gpu> {
            kinds
                .iter()
                .flat_map(|kind| gpus.0.iter().filter(move |gpu| gpu.kind == *kind))
                .cloned()
                .collect()
        };
        let discrete_first = [GpuKind::Discrete, GpuKind::Unknown, GpuKind::Integrated];
        let ranked = match preference {
            GpuPreference::Device(id) => return pinned(id, available),
            GpuPreference::SameAsCapture => match source {
                Some(id) => return pinned(id, available),
                None => by_kind(discrete_first),
            },
            GpuPreference::Auto => {
                let mut ranked: Vec<Gpu> = source_gpu.cloned().into_iter().collect();
                ranked.extend(
                    by_kind(discrete_first)
                        .into_iter()
                        .filter(|gpu| Some(gpu) != source_gpu),
                );
                ranked
            }
            GpuPreference::Integrated => {
                by_kind([GpuKind::Integrated, GpuKind::Unknown, GpuKind::Discrete])
            }
            GpuPreference::Discrete => by_kind(discrete_first),
        };
        Self {
            ranked,
            pinned: None,
            available,
        }
    }

    /// The GPUs hardware backends may use, best first.
    #[must_use]
    pub fn gpus(&self) -> &[Gpu] {
        &self.ranked
    }

    /// Whether the preference limits the hardware backends to one GPU.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.pinned.is_some()
    }

    /// The GPU a backend with `affinity` opens on: the first it can run
    /// on; `None` when the machine's GPUs are unknown (no sysfs) and
    /// nothing is pinned, or when no listed GPU suits it (the backend
    /// then uses its own default device, which fails if it has none).
    ///
    /// # Errors
    /// [`GpuError::NotFound`] when the pinned GPU does not exist,
    /// [`GpuError::Unusable`] when the backend cannot run on it.
    pub fn target(&self, affinity: GpuAffinity) -> Result<Option<&Gpu>, GpuError> {
        if let Some(id) = &self.pinned {
            let gpu = self.ranked.first().ok_or_else(|| GpuError::NotFound {
                id: id.clone(),
                available: self.available.clone(),
            })?;
            return if affinity.accepts(gpu) {
                Ok(Some(gpu))
            } else {
                Err(GpuError::Unusable {
                    gpu: gpu.render_node.clone(),
                    vendor: gpu.vendor,
                    what: affinity.needs(),
                })
            };
        }
        Ok(self.ranked.iter().find(|gpu| affinity.accepts(gpu)))
    }

    /// What a remembered backend winner is filed under: the most preferred
    /// GPU's render node (`None` without GPUs).
    #[must_use]
    pub fn key(&self) -> Option<DevNumber> {
        self.ranked.first().map(|gpu| gpu.render_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(minor: u32, vendor: GpuVendor, pci: &str, boot_vga: bool) -> Gpu {
        Gpu {
            render_node: format!("/dev/dri/renderD{minor}"),
            render_dev: DevNumber { major: 226, minor },
            primary_dev: Some(DevNumber {
                major: 226,
                minor: minor - 128,
            }),
            pci: Some(pci.parse().unwrap()),
            vendor,
            device_id: 0x1234,
            driver: None,
            boot_vga,
            kind: GpuKind::Unknown,
        }
    }

    fn kinds(gpus: Vec<Gpu>) -> Vec<GpuKind> {
        Gpus::classify(gpus).0.iter().map(|gpu| gpu.kind).collect()
    }

    #[test]
    fn preferences_parse_and_print() {
        let cases = [
            ("auto", GpuPreference::Auto),
            ("", GpuPreference::Auto),
            ("Same_As_Capture", GpuPreference::SameAsCapture),
            ("integrated", GpuPreference::Integrated),
            ("discrete", GpuPreference::Discrete),
            (
                "/dev/dri/renderD129",
                GpuPreference::Device(GpuId::Node("/dev/dri/renderD129".into())),
            ),
            (
                "renderD130",
                GpuPreference::Device(GpuId::Node("/dev/dri/renderD130".into())),
            ),
            (
                "0000:01:00.0",
                GpuPreference::Device(GpuId::Pci(PciAddress {
                    domain: 0,
                    bus: 1,
                    device: 0,
                    function: 0,
                })),
            ),
            (
                "c1:00.1",
                GpuPreference::Device(GpuId::Pci(PciAddress {
                    domain: 0,
                    bus: 0xc1,
                    device: 0,
                    function: 1,
                })),
            ),
            (
                "226:129",
                GpuPreference::Device(GpuId::Dev(DevNumber {
                    major: 226,
                    minor: 129,
                })),
            ),
        ];
        for (text, expected) in cases {
            let parsed: GpuPreference = text.parse().unwrap();
            assert_eq!(parsed, expected, "{text}");
            let again: GpuPreference = parsed.to_string().parse().unwrap();
            assert_eq!(again, expected, "round trip of {text}");
        }
        assert_eq!(
            GpuPreference::Device(GpuId::Pci("c1:00.1".parse().unwrap())).to_string(),
            "0000:c1:00.1"
        );
        for bad in ["fastest", "0000:01:20.0", "01:00.8", "0000:1ff:00.0", "x:1"] {
            assert!(bad.parse::<GpuPreference>().is_err(), "{bad}");
        }
    }

    #[test]
    fn dev_numbers_round_trip_the_linux_encoding() {
        for (major, minor) in [(226, 128), (226, 0), (4095, 255), (5000, 70_000)] {
            let number = DevNumber { major, minor };
            assert_eq!(DevNumber::from_dev(number.dev()), number);
        }
        // 226:128 is 0xe280 in the classic 8/8 bit layout.
        assert_eq!(
            DevNumber {
                major: 226,
                minor: 128
            }
            .dev(),
            0xe280
        );
    }

    #[test]
    fn classification_follows_the_documented_heuristic() {
        let nvidia = || gpu(129, GpuVendor::NVIDIA, "0000:01:00.0", false);
        // Intel laptop with NVIDIA.
        assert_eq!(
            kinds(vec![
                gpu(128, GpuVendor::INTEL, "0000:00:02.0", true),
                nvidia()
            ]),
            [GpuKind::Integrated, GpuKind::Discrete]
        );
        // Intel iGPU + Arc card.
        assert_eq!(
            kinds(vec![
                gpu(128, GpuVendor::INTEL, "0000:00:02.0", true),
                gpu(129, GpuVendor::INTEL, "0000:03:00.0", false)
            ]),
            [GpuKind::Integrated, GpuKind::Discrete]
        );
        // AMD APU + NVIDIA, even when the NVIDIA GPU is the boot display.
        assert_eq!(
            kinds(vec![
                gpu(128, GpuVendor::AMD, "0000:05:00.0", false),
                nvidia()
            ]),
            [GpuKind::Integrated, GpuKind::Discrete]
        );
        // AMD APU + AMD card: the boot display is the APU.
        assert_eq!(
            kinds(vec![
                gpu(128, GpuVendor::AMD, "0000:03:00.0", false),
                gpu(129, GpuVendor::AMD, "0000:c4:00.0", true)
            ]),
            [GpuKind::Discrete, GpuKind::Integrated]
        );
        // Intel iGPU + AMD card.
        assert_eq!(
            kinds(vec![
                gpu(128, GpuVendor::INTEL, "0000:00:02.0", false),
                gpu(129, GpuVendor::AMD, "0000:03:00.0", true)
            ]),
            [GpuKind::Integrated, GpuKind::Discrete]
        );
        // A lone AMD GPU cannot be told apart; a lone NVIDIA is discrete.
        assert_eq!(
            kinds(vec![gpu(128, GpuVendor::AMD, "0000:03:00.0", true)]),
            [GpuKind::Unknown]
        );
        assert_eq!(kinds(vec![nvidia()]), [GpuKind::Discrete]);
    }

    /// A fake sysfs: `/sys/class/drm/renderD*/{dev,device/...}`.
    struct FakeSys(PathBuf);

    impl FakeSys {
        fn new(name: &str) -> Self {
            let root =
                std::env::temp_dir().join(format!("ferricast-gpu-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            Self(root)
        }

        fn add(&self, minor: u32, card: u32, vendor: &str, device: &str, slot: &str, boot: bool) {
            let class = self.0.join(format!("class/drm/renderD{minor}"));
            let device_dir = class.join("device");
            std::fs::create_dir_all(device_dir.join(format!("drm/card{card}"))).unwrap();
            std::fs::write(class.join("dev"), format!("226:{minor}\n")).unwrap();
            std::fs::write(
                device_dir.join(format!("drm/card{card}/dev")),
                format!("226:{card}\n"),
            )
            .unwrap();
            std::fs::write(device_dir.join("vendor"), format!("0x{vendor}\n")).unwrap();
            std::fs::write(device_dir.join("device"), format!("0x{device}\n")).unwrap();
            std::fs::write(
                device_dir.join("uevent"),
                format!(
                    "DRIVER=drv{minor}\nPCI_ID={}:{}\nPCI_SLOT_NAME={slot}\n",
                    vendor.to_uppercase(),
                    device.to_uppercase()
                ),
            )
            .unwrap();
            std::fs::write(
                device_dir.join("boot_vga"),
                if boot { "1\n" } else { "0\n" },
            )
            .unwrap();
        }
    }

    impl Drop for FakeSys {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn scan_reads_a_hybrid_laptop() {
        let sys = FakeSys::new("hybrid");
        sys.add(129, 1, "10de", "28e0", "0000:01:00.0", false);
        sys.add(128, 0, "8086", "a7a0", "0000:00:02.0", true);
        std::fs::create_dir_all(sys.0.join("class/drm/card0-eDP-1")).unwrap();
        let gpus = Gpus::scan_at(&sys.0, Path::new("/dev"));
        let [intel, nvidia] = gpus.as_slice() else {
            panic!("two GPUs expected: {gpus:?}");
        };
        assert_eq!(intel.render_node, "/dev/dri/renderD128");
        assert_eq!(intel.vendor, GpuVendor::INTEL);
        assert_eq!(intel.device_id, 0xa7a0);
        assert_eq!(intel.kind, GpuKind::Integrated);
        assert!(intel.boot_vga);
        assert_eq!(intel.driver.as_deref(), Some("drv128"));
        assert_eq!(
            intel.primary_dev,
            Some(DevNumber {
                major: 226,
                minor: 0
            })
        );
        assert_eq!(nvidia.pci, Some("0000:01:00.0".parse().unwrap()));
        assert_eq!(nvidia.kind, GpuKind::Discrete);
        assert!(nvidia.matches(&GpuId::Dev(DevNumber {
            major: 226,
            minor: 1
        })));
        assert!(nvidia.matches(&"01:00.0".parse().unwrap()));
        assert!(nvidia.matches(&"renderD129".parse().unwrap()));
        assert!(!nvidia.matches(&"renderD128".parse().unwrap()));
        assert_eq!(nvidia.label(), "NVIDIA 10de:28e0 (drv129, discrete)");
    }

    #[test]
    fn scan_without_sysfs_is_empty() {
        let sys = FakeSys::new("empty");
        assert!(
            Gpus::scan_at(&sys.0, Path::new("/dev"))
                .as_slice()
                .is_empty()
        );
    }

    fn hybrid() -> Gpus {
        Gpus::classify(vec![
            gpu(128, GpuVendor::INTEL, "0000:00:02.0", true),
            gpu(129, GpuVendor::NVIDIA, "0000:01:00.0", false),
        ])
    }

    fn order(placement: &GpuPlacement) -> Vec<u32> {
        placement
            .gpus()
            .iter()
            .map(|gpu| gpu.render_dev.minor)
            .collect()
    }

    #[test]
    fn placement_orders_gpus_per_preference_and_producer() {
        let gpus = hybrid();
        let intel_frames = GpuId::Dev(DevNumber {
            major: 226,
            minor: 0,
        });
        let nvidia_frames = GpuId::Dev(DevNumber {
            major: 226,
            minor: 129,
        });
        let place = |preference: &GpuPreference, source: Option<&GpuId>| {
            GpuPlacement::resolve_in(preference, source, &gpus)
        };
        assert_eq!(order(&place(&GpuPreference::Auto, None)), [129, 128]);
        assert_eq!(
            order(&place(&GpuPreference::Auto, Some(&intel_frames))),
            [128, 129]
        );
        assert_eq!(
            order(&place(&GpuPreference::Auto, Some(&nvidia_frames))),
            [129, 128]
        );
        assert_eq!(
            order(&place(&GpuPreference::Integrated, Some(&nvidia_frames))),
            [128, 129]
        );
        assert_eq!(
            order(&place(&GpuPreference::Discrete, Some(&intel_frames))),
            [129, 128]
        );
        let same = place(&GpuPreference::SameAsCapture, Some(&intel_frames));
        assert_eq!(order(&same), [128]);
        assert!(same.is_pinned());
        assert_eq!(
            order(&place(&GpuPreference::SameAsCapture, None)),
            [129, 128]
        );
        let pinned = place(&"0000:00:02.0".parse().unwrap(), Some(&nvidia_frames));
        assert_eq!(order(&pinned), [128]);
        assert_eq!(
            pinned.key(),
            Some(DevNumber {
                major: 226,
                minor: 128
            })
        );
    }

    #[test]
    fn targets_follow_affinity_and_pins() {
        let gpus = hybrid();
        let auto = GpuPlacement::resolve_in(&GpuPreference::Integrated, None, &gpus);
        let minor = |result: Result<Option<&Gpu>, GpuError>| {
            result.unwrap().map(|gpu| gpu.render_dev.minor)
        };
        assert_eq!(minor(auto.target(GpuAffinity::Nvidia)), Some(129));
        assert_eq!(minor(auto.target(GpuAffinity::NotNvidia)), Some(128));
        assert_eq!(minor(auto.target(GpuAffinity::Any)), Some(128));

        let intel = GpuPlacement::resolve_in(&"renderD128".parse().unwrap(), None, &gpus);
        assert!(matches!(
            intel.target(GpuAffinity::Nvidia),
            Err(GpuError::Unusable { .. })
        ));
        assert_eq!(minor(intel.target(GpuAffinity::NotNvidia)), Some(128));

        let missing = GpuPlacement::resolve_in(&"renderD200".parse().unwrap(), None, &gpus);
        let error = missing.target(GpuAffinity::Any).unwrap_err();
        assert!(matches!(error, GpuError::NotFound { .. }));
        assert_eq!(
            error.to_string(),
            "the preferred GPU /dev/dri/renderD200 is not on this machine (found: \
             /dev/dri/renderD128, /dev/dri/renderD129)"
        );
        assert_eq!(missing.key(), None);

        let unknown = GpuPlacement::resolve_in(&GpuPreference::Auto, None, &Gpus::default());
        assert_eq!(minor(unknown.target(GpuAffinity::Nvidia)), None);
    }
}
