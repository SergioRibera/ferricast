//! DRM device nodes: which GPU a DMA-BUF producer renders on.

use std::fmt;
use std::os::unix::fs::{FileTypeExt as _, MetadataExt as _};
use std::path::Path;

use crate::cuda::PciBusId;
use crate::error::{Error, Result};

/// A DRM device node (`/dev/dri/cardN` or `/dev/dri/renderDN`) identified by
/// its device number, as Wayland's `linux-dmabuf` feedback (`main_device`)
/// and PipeWire report it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DrmNode {
    dev: u64,
}

impl DrmNode {
    /// The node with this device number (`dev_t`).
    #[must_use]
    pub const fn from_dev(dev: u64) -> Self {
        Self { dev }
    }

    /// The node at `path`.
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when `path` cannot be inspected,
    /// [`Error::InvalidDmaBuf`] when it is not a character device.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path).map_err(|error| Error::Os {
            call: "stat",
            errno: error.raw_os_error().unwrap_or(0),
        })?;
        if !metadata.file_type().is_char_device() {
            return Err(Error::InvalidDmaBuf {
                reason: format!("{} is not a DRM device node", path.display()),
            });
        }
        Ok(Self::from_dev(metadata.rdev()))
    }

    /// The render node of the GPU at `pci` (`/sys/bus/pci/devices/<pci>/drm/renderD*`).
    ///
    /// # Errors
    ///
    /// [`Error::Os`] when sysfs cannot be read, [`Error::InvalidDmaBuf`]
    /// when the device has no render node.
    pub fn render_node_of(pci: PciBusId) -> Result<Self> {
        let directory = format!("/sys/bus/pci/devices/{pci}/drm");
        let entries = std::fs::read_dir(&directory).map_err(|error| Error::Os {
            call: "opendir",
            errno: error.raw_os_error().unwrap_or(0),
        })?;
        let render = entries
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.file_name())
            .find(|name| name.to_string_lossy().starts_with("renderD"))
            .ok_or_else(|| Error::InvalidDmaBuf {
                reason: format!("the GPU at {pci} has no DRM render node"),
            })?;
        Self::from_path(Path::new("/dev/dri").join(render))
    }

    /// The device number.
    #[must_use]
    pub const fn dev(self) -> u64 {
        self.dev
    }

    /// The GPU's PCI address, from sysfs; `None` for a non-PCI device
    /// (a system-on-chip GPU, a virtual device) or without sysfs.
    #[must_use]
    pub fn pci_bus_id(self) -> Option<PciBusId> {
        let device = std::fs::read_link(format!("{}/device", self.sysfs())).ok()?;
        device.file_name()?.to_str()?.parse().ok()
    }

    /// The kernel's name for the node (`dri/renderD128`), from sysfs.
    #[must_use]
    pub fn name(self) -> Option<String> {
        let uevent = std::fs::read_to_string(format!("{}/uevent", self.sysfs())).ok()?;
        uevent
            .lines()
            .find_map(|line| line.strip_prefix("DEVNAME="))
            .map(str::to_owned)
    }

    fn sysfs(self) -> String {
        format!(
            "/sys/dev/char/{}:{}",
            libc::major(self.dev),
            libc::minor(self.dev)
        )
    }
}

impl fmt::Display for DrmNode {
    /// The kernel name when known, else `major:minor`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => write!(f, "/dev/{name}"),
            None => write!(f, "{}:{}", libc::major(self.dev), libc::minor(self.dev)),
        }
    }
}

impl fmt::Debug for DrmNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DrmNode({}:{})",
            libc::major(self.dev),
            libc::minor(self.dev)
        )
    }
}
