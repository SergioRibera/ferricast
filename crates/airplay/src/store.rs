//! Persistent pairing credentials.
//!
//! Non-secret data lives in JSON files under `$XDG_DATA_HOME/ferricast/airplay/`
//! (default `~/.local/share/ferricast/airplay/`), created with mode 0600:
//!
//! * `identity.json` — this sender's HomeKit pairing identifier (a UUID
//!   string) and the MAC-style device id used by legacy pairing.
//! * `<receiver device id>.json` — the receiver's long-term public key and
//!   identifier learned during pair-setup, so later connections only run
//!   pair-verify.
//! * `<receiver device id>_password.json` — marks a receiver that wants an
//!   access password.
//!
//! Secrets — the Ed25519 seed of this sender's pairing key and accepted
//! receiver passwords — go to the system keyring ([`secret`]). Files
//! written by earlier versions held them in clear text; reading such a file
//! moves the secret into the keyring and rewrites the file without it.

mod secret;

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ed25519_dalek::SigningKey;
use ferricast_core::{FerricastError, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use self::secret::{SecretBackend, SecretEntry, SecretError, SystemKeyring};

/// Set once the "identity seed kept in a file" warning was logged, so a
/// keyring-less session is told once rather than on every connection.
static SEED_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

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

    fn with_seed(file: &IdentityFile, seed: &[u8]) -> Option<Self> {
        let seed: [u8; 32] = seed.try_into().ok()?;
        Some(Self {
            pairing_id: file.pairing_id.clone(),
            device_id: file.device_id.clone(),
            key: SigningKey::from_bytes(&seed),
        })
    }
}

#[derive(Serialize, Deserialize)]
struct IdentityFile {
    pairing_id: String,
    device_id: String,
    /// Hex Ed25519 seed. Only present when no keyring is available, or in
    /// files written before secrets moved to the keyring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<String>,
}

/// A sender that completed PIN pairing with our receiver; it may skip the
/// PIN on later connections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PairedSender {
    /// Device id the sender gave as its SRP user name.
    pub device_id: String,
    /// Hex Ed25519 public key it pair-verifies with.
    pub public_key: String,
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

#[derive(Default, Serialize, Deserialize)]
struct PasswordFile {
    /// Clear-text password written by earlier versions; migrated to the
    /// keyring on read and never written again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    password: Option<String>,
}

/// What is known about a receiver's access password.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PasswordState {
    /// The receiver is not known to want one.
    NotRequired,
    /// It wants one; `Some` when a password it accepted is stored.
    Required(Option<String>),
}

/// Credential storage. Without a usable data directory it keeps everything
/// in memory for the lifetime of the session.
#[derive(Debug, Clone)]
pub(crate) struct CredentialStore {
    dir: Option<PathBuf>,
    secrets: Arc<dyn SecretBackend>,
}

impl Default for CredentialStore {
    fn default() -> Self {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
        Self {
            dir: base.map(|b| b.join("ferricast").join("airplay")),
            secrets: Arc::new(SystemKeyring::default()),
        }
    }
}

impl CredentialStore {
    #[cfg(test)]
    fn with_backend(dir: &Path, secrets: Arc<dyn SecretBackend>) -> Self {
        Self {
            dir: Some(dir.to_path_buf()),
            secrets,
        }
    }

    /// A store that keeps nothing: identities are generated afresh and no
    /// pairing outlives the session. For tests and throw-away runs.
    pub fn ephemeral() -> Self {
        Self {
            dir: None,
            secrets: Arc::new(secret::NoKeyring),
        }
    }

    /// A store under `dir` instead of the XDG data directory.
    pub fn in_dir(dir: &Path) -> Self {
        Self {
            dir: Some(dir.to_path_buf()),
            secrets: Arc::new(SystemKeyring::default()),
        }
    }

    /// Load this sender's identity, creating and saving one on first use.
    pub async fn identity(&self) -> Identity {
        self.named_identity("identity").await
    }

    /// Load the identity this process advertises as an AirPlay receiver
    /// (its `deviceid` and `pk`), creating and saving one on first use.
    /// Kept apart from the sender identity so pairing with ourselves
    /// never confuses the two.
    pub async fn receiver_identity(&self) -> Identity {
        self.named_identity("receiver-identity").await
    }

    /// Senders that completed PIN pairing with our receiver, by Ed25519
    /// public key (hex).
    pub fn paired_senders(&self) -> Vec<PairedSender> {
        self.path("receiver-paired-senders")
            .and_then(|path| read_json::<Vec<PairedSender>>(&path))
            .unwrap_or_default()
    }

    /// Remember a sender that completed PIN pairing.
    pub fn save_paired_sender(&self, sender: PairedSender) -> Result<()> {
        let Some(path) = self.path("receiver-paired-senders") else {
            return Ok(());
        };
        let mut senders = read_json::<Vec<PairedSender>>(&path).unwrap_or_default();
        senders.retain(|s| s.public_key != sender.public_key);
        senders.push(sender);
        write_json(&path, &senders)
    }

    async fn named_identity(&self, name: &str) -> Identity {
        let Some(path) = self.path(name) else {
            return Identity::generate();
        };
        if let Some(file) = read_json::<IdentityFile>(&path) {
            match file.seed.as_deref() {
                Some(seed_hex) => {
                    if let Some(identity) = hex::decode(seed_hex)
                        .ok()
                        .and_then(|seed| Identity::with_seed(&file, &seed))
                    {
                        self.move_seed_to_keyring(&path, &file, &identity).await;
                        return identity;
                    }
                    warn!(path = %path.display(), "ignoring corrupt AirPlay identity seed");
                }
                None => match self
                    .secrets
                    .load(SecretEntry::IdentitySeed {
                        pairing_id: &file.pairing_id,
                    })
                    .await
                {
                    Ok(Some(seed)) => {
                        if let Some(identity) = Identity::with_seed(&file, &seed) {
                            return identity;
                        }
                        warn!("ignoring corrupt AirPlay identity seed in the keyring");
                    }
                    Ok(None) => {
                        warn!("AirPlay identity seed missing from the keyring; pairing again");
                    }
                    Err(e) => {
                        // Saving a new identity now would orphan every
                        // pairing tied to the one in the keyring, which may
                        // only be locked or briefly unreachable.
                        warn!(error = %e, "cannot read the AirPlay identity seed; using a temporary identity");
                        return Identity::generate();
                    }
                },
            }
        }
        let identity = Identity::generate();
        self.save_identity(&path, &identity).await;
        identity
    }

    /// Move a clear-text seed found in `identity.json` into the keyring and
    /// rewrite the file without it.
    async fn move_seed_to_keyring(&self, path: &Path, file: &IdentityFile, identity: &Identity) {
        match self
            .secrets
            .store(
                SecretEntry::IdentitySeed {
                    pairing_id: &file.pairing_id,
                },
                &identity.key.to_bytes(),
            )
            .await
        {
            Ok(()) => {
                let stripped = IdentityFile {
                    pairing_id: file.pairing_id.clone(),
                    device_id: file.device_id.clone(),
                    seed: None,
                };
                match write_json(path, &stripped) {
                    Ok(()) => info!("moved the AirPlay identity seed into the system keyring"),
                    Err(e) => {
                        warn!(error = %e, "cannot remove the AirPlay identity seed from its file");
                    }
                }
            }
            Err(e) => warn_seed_in_file(&e),
        }
    }

    async fn save_identity(&self, path: &Path, identity: &Identity) {
        let seed = identity.key.to_bytes();
        let file_seed = match self
            .secrets
            .store(
                SecretEntry::IdentitySeed {
                    pairing_id: &identity.pairing_id,
                },
                &seed,
            )
            .await
        {
            Ok(()) => None,
            Err(e) => {
                warn_seed_in_file(&e);
                Some(hex::encode(seed))
            }
        };
        let file = IdentityFile {
            pairing_id: identity.pairing_id.clone(),
            device_id: identity.device_id.clone(),
            seed: file_seed,
        };
        if let Err(e) = write_json(path, &file) {
            warn!(error = %e, "cannot persist AirPlay identity; pairing will not survive restarts");
        }
    }

    /// Stored key of receiver `device_id`, if paired before.
    pub fn receiver(&self, device_id: &str) -> Option<ReceiverKey> {
        read_json(&self.path(device_id)?)
    }

    /// Remember receiver `device_id`'s key.
    pub fn save_receiver(&self, device_id: &str, key: &ReceiverKey) -> Result<()> {
        match self.path(device_id) {
            Some(path) => write_json(&path, key),
            None => Ok(()),
        }
    }

    /// Forget receiver `device_id` (e.g. after it rejected pair-verify).
    pub fn forget_receiver(&self, device_id: &str) {
        if let Some(path) = self.path(device_id) {
            remove_file(&path);
        }
    }

    /// What is known about receiver `device_id`'s access password.
    pub async fn password(&self, device_id: &str) -> PasswordState {
        let Some(path) = self.password_path(device_id) else {
            return PasswordState::NotRequired;
        };
        let Some(file) = read_json::<PasswordFile>(&path) else {
            return PasswordState::NotRequired;
        };
        let entry = SecretEntry::Password { device_id };
        if let Some(password) = file.password {
            match self.secrets.store(entry, password.as_bytes()).await {
                Ok(()) => info!("moved the AirPlay password into the system keyring"),
                // Without a keyring passwords are not kept at all; this one
                // still serves the current connection.
                Err(e) => {
                    warn!(error = %e, "dropping the clear-text AirPlay password; it will be asked again");
                }
            }
            if let Err(e) = write_json(&path, &PasswordFile::default()) {
                warn!(error = %e, "cannot remove the clear-text AirPlay password");
            }
            return PasswordState::Required(Some(password));
        }
        match self.secrets.load(entry).await {
            Ok(Some(bytes)) => {
                if let Ok(password) = String::from_utf8(bytes) {
                    PasswordState::Required(Some(password))
                } else {
                    warn!("ignoring a corrupt AirPlay password in the keyring");
                    PasswordState::Required(None)
                }
            }
            Ok(None) => PasswordState::Required(None),
            Err(e) => {
                debug!(error = %e, "no stored AirPlay password");
                PasswordState::Required(None)
            }
        }
    }

    /// Record that receiver `device_id` wants an access password.
    pub fn mark_password_required(&self, device_id: &str) {
        let Some(path) = self.password_path(device_id) else {
            return;
        };
        if let Err(e) = write_json(&path, &PasswordFile::default()) {
            warn!(error = %e, "cannot record that the AirPlay receiver wants a password");
        }
    }

    /// Remember `password`, which receiver `device_id` accepted. Without a
    /// keyring it is not kept: a file would leave it readable in backups
    /// and to anything running as the user.
    pub async fn save_password(&self, device_id: &str, password: &str) {
        self.mark_password_required(device_id);
        if let Err(e) = self
            .secrets
            .store(SecretEntry::Password { device_id }, password.as_bytes())
            .await
        {
            warn!(error = %e, "cannot store the AirPlay password; it will be asked again");
        }
    }

    /// Drop the stored password of receiver `device_id` (it was rejected);
    /// the receiver stays marked as wanting one.
    pub async fn forget_password(&self, device_id: &str) {
        // The backend logs failures; a password left behind is rejected
        // and dropped again on the next attempt.
        let _ = self
            .secrets
            .delete(SecretEntry::Password { device_id })
            .await;
    }

    /// Forget everything about receiver `device_id`'s password: it no
    /// longer asks for one.
    pub async fn clear_password(&self, device_id: &str) {
        if let Some(path) = self.password_path(device_id) {
            remove_file(&path);
        }
        self.forget_password(device_id).await;
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
}

/// Log that the identity seed stays in its 0600 file: without a keyring
/// that is the only way to keep pairings across restarts, and losing them
/// would mean entering a receiver PIN on every launch.
fn warn_seed_in_file(error: &SecretError) {
    if SEED_FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
        debug!(%error, "AirPlay identity seed kept in its file");
    } else {
        warn!(%error, "no usable system keyring; keeping the AirPlay identity seed in a file readable only by this user");
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text)
        .inspect_err(
            |e| warn!(path = %path.display(), error = %e, "ignoring corrupt AirPlay credentials"),
        )
        .ok()
}

fn remove_file(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!(path = %path.display(), error = %e, "cannot remove AirPlay credentials"),
    }
}

/// Write `value` to `path` atomically (temporary file, then rename), so a
/// crash never leaves a half-written file and rewriting a file without its
/// secret leaves either the old or the new content.
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
    let written = opts.open(&tmp).and_then(|mut file| {
        file.write_all(&json)?;
        file.sync_all()
    });
    if let Err(e) = written.and_then(|()| fs::rename(&tmp, path)) {
        // Don't leave a stray temporary file behind.
        let _ = fs::remove_file(&tmp);
        return Err(io_err(e));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::secret::NoKeyring;
    use super::secret::testing::MemoryKeyring;
    use super::*;

    const RECEIVER: &str = "AA:BB:CC:DD:EE:FF";
    const PASSWORD_FILE: &str = "AA_BB_CC_DD_EE_FF_password.json";

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            Self(
                std::env::temp_dir()
                    .join(format!("ferricast-airplay-store-{}", uuid::Uuid::new_v4())),
            )
        }

        fn file(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }

        fn read(&self, name: &str) -> String {
            fs::read_to_string(self.file(name)).unwrap()
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn memory_store(dir: &TempDir) -> (CredentialStore, Arc<MemoryKeyring>) {
        let keyring = Arc::new(MemoryKeyring::default());
        (
            CredentialStore::with_backend(&dir.0, keyring.clone()),
            keyring,
        )
    }

    fn password_entry() -> SecretEntry<'static> {
        SecretEntry::Password {
            device_id: RECEIVER,
        }
    }

    #[tokio::test]
    async fn identity_seed_lives_in_the_keyring() {
        let dir = TempDir::new();
        let (store, keyring) = memory_store(&dir);
        let a = store.identity().await;
        let b = store.identity().await;
        assert_eq!(a.pairing_id, b.pairing_id);
        assert_eq!(a.key.to_bytes(), b.key.to_bytes());
        assert_eq!(a.device_id.len(), 17);
        assert_eq!(
            keyring.get(SecretEntry::IdentitySeed {
                pairing_id: &a.pairing_id
            }),
            Some(a.key.to_bytes().to_vec())
        );
        let file = dir.read("identity.json");
        assert!(!file.contains("seed"), "{file}");
        assert!(!file.contains(&hex::encode(a.key.to_bytes())));
    }

    #[tokio::test]
    async fn clear_text_identity_seed_moves_to_the_keyring() {
        let dir = TempDir::new();
        let seed = [7_u8; 32];
        write_json(
            &dir.file("identity.json"),
            &IdentityFile {
                pairing_id: "PAIRING".into(),
                device_id: "11:22:33:44:55:66".into(),
                seed: Some(hex::encode(seed)),
            },
        )
        .unwrap();
        let (store, keyring) = memory_store(&dir);
        let identity = store.identity().await;
        assert_eq!(identity.pairing_id, "PAIRING");
        assert_eq!(identity.key.to_bytes(), seed);
        assert_eq!(
            keyring.get(SecretEntry::IdentitySeed {
                pairing_id: "PAIRING"
            }),
            Some(seed.to_vec())
        );
        assert!(!dir.read("identity.json").contains("seed"));
        assert!(!dir.file("identity.json.tmp").exists());
        // Later loads read the seed back from the keyring.
        assert_eq!(store.identity().await.key.to_bytes(), seed);
    }

    #[tokio::test]
    async fn identity_seed_stays_in_its_file_without_a_keyring() {
        let dir = TempDir::new();
        let store = CredentialStore::with_backend(&dir.0, Arc::new(NoKeyring));
        let a = store.identity().await;
        assert!(
            dir.read("identity.json")
                .contains(&hex::encode(a.key.to_bytes()))
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(dir.file("identity.json"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert_eq!(store.identity().await.key.to_bytes(), a.key.to_bytes());
    }

    #[tokio::test]
    async fn unreadable_keyring_does_not_replace_the_identity() {
        let dir = TempDir::new();
        let (store, _) = memory_store(&dir);
        let saved = store.identity().await;
        let before = dir.read("identity.json");
        let locked = CredentialStore::with_backend(&dir.0, Arc::new(NoKeyring));
        let temporary = locked.identity().await;
        assert_ne!(temporary.pairing_id, saved.pairing_id);
        assert_eq!(dir.read("identity.json"), before);
    }

    #[test]
    fn receiver_keys_persist() {
        let dir = TempDir::new();
        let (store, _) = memory_store(&dir);
        let key = ReceiverKey {
            id: "accessory".into(),
            public_key: hex::encode([3_u8; 32]),
        };
        store.save_receiver(RECEIVER, &key).unwrap();
        assert_eq!(store.receiver(RECEIVER), Some(key));
        store.forget_receiver(RECEIVER);
        assert_eq!(store.receiver(RECEIVER), None);
        store.forget_receiver(RECEIVER);
    }

    #[tokio::test]
    async fn passwords_live_in_the_keyring() {
        let dir = TempDir::new();
        let (store, keyring) = memory_store(&dir);
        assert_eq!(store.password(RECEIVER).await, PasswordState::NotRequired);
        store.mark_password_required(RECEIVER);
        assert_eq!(
            store.password(RECEIVER).await,
            PasswordState::Required(None)
        );
        store.save_password(RECEIVER, "secret").await;
        assert_eq!(
            store.password(RECEIVER).await,
            PasswordState::Required(Some("secret".into()))
        );
        assert!(!dir.read(PASSWORD_FILE).contains("secret"));
        assert_eq!(keyring.get(password_entry()), Some(b"secret".to_vec()));

        store.forget_password(RECEIVER).await;
        assert_eq!(
            store.password(RECEIVER).await,
            PasswordState::Required(None)
        );
        store.save_password(RECEIVER, "secret").await;
        store.clear_password(RECEIVER).await;
        assert_eq!(store.password(RECEIVER).await, PasswordState::NotRequired);
        assert_eq!(keyring.get(password_entry()), None);
    }

    #[tokio::test]
    async fn clear_text_password_moves_to_the_keyring() {
        let dir = TempDir::new();
        write_json(
            &dir.file(PASSWORD_FILE),
            &PasswordFile {
                password: Some("hunter2".into()),
            },
        )
        .unwrap();
        let (store, keyring) = memory_store(&dir);
        assert_eq!(
            store.password(RECEIVER).await,
            PasswordState::Required(Some("hunter2".into()))
        );
        assert!(!dir.read(PASSWORD_FILE).contains("hunter2"));
        assert_eq!(keyring.get(password_entry()), Some(b"hunter2".to_vec()));
        assert_eq!(
            store.password(RECEIVER).await,
            PasswordState::Required(Some("hunter2".into()))
        );
    }

    #[tokio::test]
    async fn passwords_are_not_kept_without_a_keyring() {
        let dir = TempDir::new();
        write_json(
            &dir.file(PASSWORD_FILE),
            &PasswordFile {
                password: Some("hunter2".into()),
            },
        )
        .unwrap();
        let store = CredentialStore::with_backend(&dir.0, Arc::new(NoKeyring));
        // The clear-text password serves this connection, then is gone.
        assert_eq!(
            store.password(RECEIVER).await,
            PasswordState::Required(Some("hunter2".into()))
        );
        assert!(!dir.read(PASSWORD_FILE).contains("hunter2"));
        assert_eq!(
            store.password(RECEIVER).await,
            PasswordState::Required(None)
        );
        store.save_password(RECEIVER, "hunter2").await;
        assert_eq!(
            store.password(RECEIVER).await,
            PasswordState::Required(None)
        );
        assert!(!dir.read(PASSWORD_FILE).contains("hunter2"));
    }
}
