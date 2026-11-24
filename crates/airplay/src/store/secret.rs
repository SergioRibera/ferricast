//! Secrets kept in the system keyring through the freedesktop Secret
//! Service (or the Secret portal inside a sandbox), behind a trait so the
//! credential store can be tested without a real keyring.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use tokio::sync::OnceCell;
use tracing::{debug, warn};

/// Upper bound for reaching the Secret Service. D-Bus activation of a
/// missing service can otherwise stall a connection attempt for ~25 s.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Upper bound for one keyring request. Generous because unlocking a
/// locked collection waits for the user to answer the unlock prompt.
const REQUEST_TIMEOUT: Duration = Duration::from_mins(1);

/// Attributes identifying a keyring item.
type Attributes = Vec<(&'static str, String)>;

/// A secret the credential store keeps in the keyring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SecretEntry<'a> {
    /// Seed of this sender's long-term Ed25519 pairing key, for the identity
    /// with this pairing id (one per data directory).
    IdentitySeed { pairing_id: &'a str },
    /// Access password of the receiver with this device id.
    Password { device_id: &'a str },
}

impl SecretEntry<'_> {
    fn label(self) -> String {
        match self {
            Self::IdentitySeed { .. } => "ferricast AirPlay pairing key".to_owned(),
            Self::Password { device_id } => format!("ferricast AirPlay password for {device_id}"),
        }
    }

    /// Attributes that identify the keyring item.
    fn attributes(self) -> Attributes {
        let mut attributes = vec![("application", "ferricast".to_owned())];
        match self {
            Self::IdentitySeed { pairing_id } => {
                attributes.push(("kind", "airplay-identity-seed".to_owned()));
                attributes.push(("pairing-id", pairing_id.to_owned()));
            }
            Self::Password { device_id } => {
                attributes.push(("kind", "airplay-password".to_owned()));
                attributes.push(("device-id", device_id.to_owned()));
            }
        }
        attributes
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SecretError {
    /// No keyring to talk to (headless session, no Secret Service).
    #[error("no system keyring available: {0}")]
    Unavailable(String),
    /// The keyring is there but refused or failed the request.
    #[error("keyring request failed: {0}")]
    Request(String),
}

pub(crate) type SecretFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, SecretError>> + Send + 'a>>;

/// Where secrets are kept.
pub(crate) trait SecretBackend: Send + Sync + fmt::Debug {
    /// The secret stored for `entry`, `None` when there is none.
    fn load<'a>(&'a self, entry: SecretEntry<'a>) -> SecretFuture<'a, Option<Vec<u8>>>;
    /// Store `secret` for `entry`, replacing any previous one.
    fn store<'a>(&'a self, entry: SecretEntry<'a>, secret: &'a [u8]) -> SecretFuture<'a, ()>;
    /// Remove the secret stored for `entry`, if any.
    fn delete<'a>(&'a self, entry: SecretEntry<'a>) -> SecretFuture<'a, ()>;
}

/// The system keyring, connected on first use. A failed connection is
/// remembered so a keyring-less session pays the timeout once.
#[derive(Default)]
pub(crate) struct SystemKeyring {
    keyring: OnceCell<Option<oo7::Keyring>>,
}

impl fmt::Debug for SystemKeyring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemKeyring")
            .field("connected", &self.keyring.get().map(Option::is_some))
            .finish()
    }
}

impl SystemKeyring {
    async fn keyring(&self) -> Result<&oo7::Keyring, SecretError> {
        self.keyring
            .get_or_init(|| async {
                match tokio::time::timeout(CONNECT_TIMEOUT, oo7::Keyring::new()).await {
                    Ok(Ok(keyring)) => Some(keyring),
                    Ok(Err(e)) => {
                        debug!(error = %e, "cannot open the system keyring");
                        None
                    }
                    Err(_) => {
                        debug!("timed out opening the system keyring");
                        None
                    }
                }
            })
            .await
            .as_ref()
            .ok_or_else(|| SecretError::Unavailable("no Secret Service on the session bus".into()))
    }

    async fn request<T>(
        &self,
        what: &str,
        op: impl Future<Output = oo7::Result<T>>,
    ) -> Result<T, SecretError> {
        match tokio::time::timeout(REQUEST_TIMEOUT, op).await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(e)) => Err(SecretError::Request(format!("{what}: {e}"))),
            Err(_) => Err(SecretError::Request(format!("{what}: timed out"))),
        }
    }
}

impl SecretBackend for SystemKeyring {
    fn load<'a>(&'a self, entry: SecretEntry<'a>) -> SecretFuture<'a, Option<Vec<u8>>> {
        Box::pin(async move {
            let keyring = self.keyring().await?;
            let items = self
                .request("search", keyring.search_items(&entry.attributes()))
                .await?;
            let Some(item) = items.first() else {
                return Ok(None);
            };
            if self.request("lock state", item.is_locked()).await? {
                self.request("unlock", item.unlock()).await?;
            }
            let secret = self.request("read secret", item.secret()).await?;
            Ok(Some(secret.as_bytes().to_vec()))
        })
    }

    fn store<'a>(&'a self, entry: SecretEntry<'a>, secret: &'a [u8]) -> SecretFuture<'a, ()> {
        Box::pin(async move {
            let keyring = self.keyring().await?;
            self.request("unlock", keyring.unlock()).await?;
            self.request(
                "store secret",
                keyring.create_item(&entry.label(), &entry.attributes(), secret, true),
            )
            .await
        })
    }

    fn delete<'a>(&'a self, entry: SecretEntry<'a>) -> SecretFuture<'a, ()> {
        Box::pin(async move {
            let keyring = self.keyring().await?;
            let result = self
                .request("delete secret", keyring.delete(&entry.attributes()))
                .await;
            if let Err(e) = &result {
                warn!(error = %e, ?entry, "cannot remove a secret from the keyring");
            }
            result
        })
    }
}

/// No keyring at all: every request fails as unavailable, so nothing is
/// kept. Backs stores that must not outlive the session.
#[derive(Debug, Default)]
pub(crate) struct NoKeyring;

impl SecretBackend for NoKeyring {
    fn load<'a>(&'a self, _entry: SecretEntry<'a>) -> SecretFuture<'a, Option<Vec<u8>>> {
        Box::pin(async { Err(SecretError::Unavailable("no keyring in use".into())) })
    }

    fn store<'a>(&'a self, _entry: SecretEntry<'a>, _secret: &'a [u8]) -> SecretFuture<'a, ()> {
        Box::pin(async { Err(SecretError::Unavailable("no keyring in use".into())) })
    }

    fn delete<'a>(&'a self, _entry: SecretEntry<'a>) -> SecretFuture<'a, ()> {
        Box::pin(async { Err(SecretError::Unavailable("no keyring in use".into())) })
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! In-memory backends for tests.

    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::{Attributes, SecretBackend, SecretEntry, SecretFuture};

    /// A keyring kept in memory.
    #[derive(Debug, Default)]
    pub(crate) struct MemoryKeyring {
        items: Mutex<HashMap<Attributes, Vec<u8>>>,
    }

    impl MemoryKeyring {
        pub(crate) fn get(&self, entry: SecretEntry<'_>) -> Option<Vec<u8>> {
            self.items
                .lock()
                .expect("invariant: no test panics while holding the lock")
                .get(&entry.attributes())
                .cloned()
        }
    }

    impl SecretBackend for MemoryKeyring {
        fn load<'a>(&'a self, entry: SecretEntry<'a>) -> SecretFuture<'a, Option<Vec<u8>>> {
            let value = self.get(entry);
            Box::pin(async move { Ok(value) })
        }

        fn store<'a>(&'a self, entry: SecretEntry<'a>, secret: &'a [u8]) -> SecretFuture<'a, ()> {
            self.items
                .lock()
                .expect("invariant: no test panics while holding the lock")
                .insert(entry.attributes(), secret.to_vec());
            Box::pin(async { Ok(()) })
        }

        fn delete<'a>(&'a self, entry: SecretEntry<'a>) -> SecretFuture<'a, ()> {
            self.items
                .lock()
                .expect("invariant: no test panics while holding the lock")
                .remove(&entry.attributes());
            Box::pin(async { Ok(()) })
        }
    }
}
