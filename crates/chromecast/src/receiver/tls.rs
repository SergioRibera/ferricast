//! Self-signed TLS material for the receiver-side server.
//!
//! Generated once per process launch and held only in memory. Real
//! Chromecast devices ship with a device-specific cert issued by
//! Google's Cast CA; their senders ask for it via the `deviceauth`
//! namespace and refuse to issue LOAD without a valid signature.
//! We can't replicate that chain (Google doesn't issue these for
//! third-party software), so the self-signed approach is the
//! lowest-common-denominator that interoperates with senders that
//! don't enforce CA validation — VLC, Stream2Chromecast,
//! BubbleUPnP, the Cast SDK in dev mode, etc.
//!
//! RSA-2048 is chosen over ECDSA because the Cast SDK on Android TV /
//! Google TV closes the TLS connection on ECDSA server certs. Real
//! Chromecast devices ship RSA-2048 certs, and some Cast sender
//! implementations only offer cipher suites compatible with RSA.

use std::sync::Arc;

use rcgen::{
    CertificateParams, DistinguishedName, DnType, KeyPair, PKCS_RSA_SHA256,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::ServerConfig;
use rsa::pkcs8::EncodePrivateKey as _;
use rsa::RsaPrivateKey;

use ferricast_core::{FerricastError, Result};

/// Build a rustls `ServerConfig` backed by a fresh self-signed RSA-2048
/// keypair. Subject Alternative Names cover the loopback + every
/// IP we might advertise on; receivers don't actually validate any
/// of this against the cert but a well-formed SAN list keeps
/// certain rustls versions happy.
pub fn build_server_config(advertised_ips: &[std::net::IpAddr]) -> Result<Arc<ServerConfig>> {
    let mut params = CertificateParams::new(
        std::iter::once("localhost".to_string())
            .chain(advertised_ips.iter().map(|ip| ip.to_string()))
            .collect::<Vec<_>>(),
    )
    .map_err(|e| FerricastError::Receiver(format!("rcgen params: {e}")))?;
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, "ferricast-receiver");
    params.distinguished_name = dn;

    let rsa_key = RsaPrivateKey::new(&mut rand::thread_rng(), 2048)
        .map_err(|e| FerricastError::Receiver(format!("rsa keygen: {e}")))?;
    let pkcs8_der = rsa_key
        .to_pkcs8_der()
        .map_err(|e| FerricastError::Receiver(format!("rsa pkcs8: {e}")))?;
    let pkcs8_ref = PrivatePkcs8KeyDer::from(pkcs8_der.as_bytes());
    let key_pair = KeyPair::from_pkcs8_der_and_sign_algo(&pkcs8_ref, &PKCS_RSA_SHA256)
        .map_err(|e| FerricastError::Receiver(format!("rcgen keypair: {e}")))?;
    let cert = params
        .self_signed(&key_pair)
        .map_err(|e| FerricastError::Receiver(format!("rcgen self-sign: {e}")))?;
    let cert_der = CertificateDer::from(cert.der().to_vec());
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8_der.as_bytes().to_vec()));

    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .map_err(|e| FerricastError::Receiver(format!("rustls server config: {e}")))?;
    Ok(Arc::new(config))
}
