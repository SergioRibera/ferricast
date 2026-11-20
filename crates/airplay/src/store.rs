//! Persistent pairing credentials.
//!
//! * `identity.json` — this sender's long-term Ed25519 key, its HomeKit
//!   pairing identifier (a UUID string) and the MAC-style device id used by
//!   legacy pairing.
//! * `<receiver device id>.json` — the receiver's long-term public key and
//!   identifier learned during pair-setup, so later connections only run
//!   pair-verify.
//! * `<receiver device id>_password.json` — marks a receiver that wants an
//!   access password and, once it accepted one, the password itself so it
//!   is not asked for again. Stored in clear text, protected by the file
//!   mode only.
//!
//! Files live in `$XDG_DATA_HOME/ferricast/airplay/` (default
//! `~/.local/share/ferricast/airplay/`) and are created with mode 0600.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use ed25519_dalek::SigningKey;
use ferricast_core::{FerricastError, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

/// This sender's long-term pairing identity.
#[derive(Clone)]
pub(crate) struct Identity {
    /// HomeKit pairing identifier (UUID string, sent as UTF-8).
    pub pairing_id: String,
    /// MAC-style id (`AA:BB:CC:DD:EE:FF`) used as the legacy SRP username
    /// and the `deviceID` in `SETUP`.
    pub device_id: String,
    pub key: SigningKey,
}

impl Identity {
    fn generate() -> Self {
        let mac: [u8; 6] = rand::random();
        Self {
            pairing_id: uuid::Uuid::new_v4().to_string().to_uppercase(),
            device_id: mac
                .iter()
                .map(|b| format!("{b:02X}"))
                .collect::<Vec<_>>()
                .join(":"),
            key: SigningKey::generate(&mut rand::rngs::OsRng),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct IdentityFile {
    pairing_id: String,
    device_id: String,
    /// Hex Ed25519 seed.
    seed: String,
}

/// Long-term key material of a paired receiver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReceiverKey {
    /// Receiver identifier from HomeKit pair-setup M6 (empty for legacy).
    pub id: String,
    /// Hex Ed25519 public key.
    pub public_key: String,
}

impl ReceiverKey {
    pub fn public_key_bytes(&self) -> Option<[u8; 32]> {
        hex::decode(&self.public_key).ok()?.try_into().ok()
    }
}

#[derive(Serialize, Deserialize)]
struct PasswordFile {
    password: Option<String>,
}

/// Credential storage. Without a usable data directory it keeps everything
/// in memory for the lifetime of the session.
#[derive(Debug, Clone)]
pub(crate) struct CredentialStore {
    dir: Option<PathBuf>,
}

impl Default for CredentialStore {
    fn default() -> Self {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
        Self {
            dir: base.map(|b| b.join("ferricast").join("airplay")),
        }
    }
}

impl CredentialStore {
    #[cfg(test)]
    pub fn at(dir: &Path) -> Self {
        Self {
            dir: Some(dir.to_path_buf()),
        }
    }

    /// Load this sender's identity, creating and saving one on first use.
    pub fn identity(&self) -> Identity {
        if let Some(path) = self.path("identity") {
            if let Some(identity) = Self::read_identity(&path) {
                return identity;
            }
            let identity = Identity::generate();
            let file = IdentityFile {
                pairing_id: identity.pairing_id.clone(),
                device_id: identity.device_id.clone(),
                seed: hex::encode(identity.key.to_bytes()),
            };
            if let Err(e) = Self::write_json(&path, &file) {
                warn!(error = %e, "cannot persist AirPlay identity; pairing will not survive restarts");
            }
            return identity;
        }
        Identity::generate()
    }

    fn read_identity(path: &Path) -> Option<Identity> {
        let text = fs::read_to_string(path).ok()?;
        let file: IdentityFile = serde_json::from_str(&text)
            .inspect_err(
                |e| warn!(path = %path.display(), error = %e, "ignoring corrupt AirPlay identity"),
            )
            .ok()?;
        let seed: [u8; 32] = hex::decode(&file.seed).ok()?.try_into().ok()?;
        Some(Identity {
            pairing_id: file.pairing_id,
            device_id: file.device_id,
            key: SigningKey::from_bytes(&seed),
        })
    }

    /// Stored key of receiver `device_id`, if paired before.
    pub fn receiver(&self, device_id: &str) -> Option<ReceiverKey> {
        let text = fs::read_to_string(self.path(device_id)?).ok()?;
        serde_json::from_str(&text).ok()
    }

    /// Remember receiver `device_id`'s key.
    pub fn save_receiver(&self, device_id: &str, key: &ReceiverKey) -> Result<()> {
        match self.path(device_id) {
            Some(path) => Self::write_json(&path, key),
            None => Ok(()),
        }
    }

    /// Forget receiver `device_id` (e.g. after it rejected pair-verify).
    pub fn forget_receiver(&self, device_id: &str) {
        if let Some(path) = self.path(device_id) {
            let _ = fs::remove_file(path);
        }
    }

    /// What is known about receiver `device_id`'s access password: `None`
    /// when it is not known to want one, `Some(None)` when it wants one that
    /// is not stored.
    pub fn password(&self, device_id: &str) -> Option<Option<String>> {
        let text = fs::read_to_string(self.password_path(device_id)?).ok()?;
        serde_json::from_str::<PasswordFile>(&text)
            .ok()
            .map(|f| f.password)
    }

    /// Record that receiver `device_id` wants an access password, and the
    /// password when it is known to be right.
    pub fn save_password(&self, device_id: &str, password: Option<&str>) -> Result<()> {
        match self.password_path(device_id) {
            Some(path) => Self::write_json(
                &path,
                &PasswordFile {
                    password: password.map(str::to_owned),
                },
            ),
            None => Ok(()),
        }
    }

    fn password_path(&self, device_id: &str) -> Option<PathBuf> {
        self.path(&format!("{device_id}.password"))
    }

    fn path(&self, name: &str) -> Option<PathBuf> {
        // Device ids are MAC addresses; keep only filename-safe characters.
        let safe: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        Some(self.dir.as_ref()?.join(format!("{safe}.json")))
    }

    fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
        let io_err =
            |e: std::io::Error| FerricastError::Other(format!("write {}: {e}", path.display()));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(io_err)?;
        }
        let json = serde_json::to_vec_pretty(value)
            .map_err(|e| FerricastError::Other(format!("serialize credentials: {e}")))?;
        let tmp = path.with_extension("json.tmp");
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            opts.mode(0o600);
        }
        let mut file = opts.open(&tmp).map_err(io_err)?;
        file.write_all(&json).map_err(io_err)?;
        file.sync_all().map_err(io_err)?;
        fs::rename(&tmp, path).map_err(io_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_receiver_keys_persist() {
        let dir =
            std::env::temp_dir().join(format!("ferricast-airplay-store-{}", uuid::Uuid::new_v4()));
        let store = CredentialStore::at(&dir);
        let a = store.identity();
        let b = store.identity();
        assert_eq!(a.pairing_id, b.pairing_id);
        assert_eq!(a.key.to_bytes(), b.key.to_bytes());
        assert_eq!(a.device_id.len(), 17);

        let key = ReceiverKey {
            id: "accessory".into(),
            public_key: hex::encode([3_u8; 32]),
        };
        store.save_receiver("AA:BB:CC:DD:EE:FF", &key).unwrap();
        assert_eq!(store.receiver("AA:BB:CC:DD:EE:FF"), Some(key));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(dir.join("AA_BB_CC_DD_EE_FF.json"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        store.forget_receiver("AA:BB:CC:DD:EE:FF");
        assert_eq!(store.receiver("AA:BB:CC:DD:EE:FF"), None);

        assert_eq!(store.password("AA:BB:CC:DD:EE:FF"), None);
        store.save_password("AA:BB:CC:DD:EE:FF", None).unwrap();
        assert_eq!(store.password("AA:BB:CC:DD:EE:FF"), Some(None));
        store
            .save_password("AA:BB:CC:DD:EE:FF", Some("secret"))
            .unwrap();
        assert_eq!(
            store.password("AA:BB:CC:DD:EE:FF"),
            Some(Some("secret".to_owned()))
        );
        let _ = fs::remove_dir_all(dir);
    }
}
