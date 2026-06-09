//! TLS material for the HLS server.
//!
//! HTTPS HLS is opt-in via [`crate::HlsConfig::tls`]. Most Cast
//! senders accept plain `http://` on the LAN, but a growing share
//! (post-2023 Cast Application Framework, some smart-TV firmwares)
//! refuse anything but `https://` even for ad-hoc local URLs, so
//! the chromecast handler turns it on. We sign a fresh certificate
//! per session — receivers don't validate identity over an ad-hoc
//! local cast (their TLS stack accepts any cert presented), so the
//! handshake just buys us encryption on the wire.
//!
//! The puller side already opts into accepting invalid certs (see
//! `puller::build_client`), so cast-to-self with our own self-signed
//! material works end-to-end.

use std::sync::Arc;

use rcgen::{CertificateParams, DistinguishedName, DnType, KeyPair};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::ServerConfig;

use ferricast_core::{FerricastError, Result};

/// Build a rustls [`ServerConfig`] backed by a fresh self-signed
/// keypair. `advertised_ips` are added to the cert's Subject
/// Alternative Names list so a strict-validating client wouldn't
/// reject the cert outright on hostname mismatch — though in
/// practice every Cast sender just accepts the presented cert.
pub fn build_self_signed_server_config(
    advertised_ips: &[std::net::IpAddr],
) -> Result<Arc<ServerConfig>> {
    let mut params = CertificateParams::new(
        std::iter::once("localhost".to_string())
            .chain(advertised_ips.iter().map(|ip| ip.to_string()))
            .collect::<Vec<_>>(),
    )
    .map_err(|e| FerricastError::Hls(format!("rcgen params: {e}")))?;
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, "ferricast-hls");
    params.distinguished_name = dn;

    let key_pair = KeyPair::generate()
        .map_err(|e| FerricastError::Hls(format!("rcgen keypair: {e}")))?;
    let cert = params
        .self_signed(&key_pair)
        .map_err(|e| FerricastError::Hls(format!("rcgen self-sign: {e}")))?;
    let cert_der = CertificateDer::from(cert.der().to_vec());
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pair.serialize_der()));

    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .map_err(|e| FerricastError::Hls(format!("rustls server config: {e}")))?;
    Ok(Arc::new(config))
}
