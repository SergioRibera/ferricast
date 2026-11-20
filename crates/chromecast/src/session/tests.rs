//! End-to-end tests against a fake Chromecast: a local TLS server with
//! a self-signed certificate speaking just enough CASTv2.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use ferricast_core::{CastSession, ConnectOutcome, Device, FerricastError};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde_json::{Value, json};
use tokio::io::{ReadHalf, split};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::TlsAcceptor;
use tokio_rustls::server::TlsStream;

use super::ChromecastSession;
use super::monitor::Timing;
use crate::castv2::{
    CastMessage, DEFAULT_MEDIA_RECEIVER_APP_ID, DEFAULT_RECEIVER_ID, DEFAULT_SENDER_ID, namespace,
};
use crate::wire::{self, CastWriter, SharedWriter};

const TRANSPORT: &str = "web-7";
const SESSION: &str = "sess-7";

struct FakeCast {
    listener: TcpListener,
    acceptor: TlsAcceptor,
}

struct FakeConn {
    reader: ReadHalf<TlsStream<TcpStream>>,
    writer: SharedWriter,
    buf: BytesMut,
}

impl FakeCast {
    async fn start() -> Self {
        let certified = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert = CertificateDer::from(certified.cert.der().to_vec());
        let key =
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(certified.key_pair.serialize_der()));
        let config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_protocol_versions(&[&rustls::version::TLS12])
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert], key)
        .unwrap();
        Self {
            listener: TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap(),
            acceptor: TlsAcceptor::from(Arc::new(config)),
        }
    }

    fn device(&self) -> Device {
        Device {
            id: uuid::Uuid::nil(),
            name: "fake".into(),
            protocol: "chromecast",
            protocol_icon: Bytes::new(),
            addr: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: self.listener.local_addr().unwrap().port(),
            model: None,
            capabilities: ferricast_core::DeviceCapabilities::default(),
            metadata: HashMap::new(),
        }
    }

    async fn accept(&self) -> FakeConn {
        let (tcp, _) = self.listener.accept().await.unwrap();
        let tls = self.acceptor.accept(tcp).await.unwrap();
        let (reader, writer) = split(tls);
        FakeConn {
            reader,
            writer: CastWriter::shared(writer),
            buf: BytesMut::new(),
        }
    }
}

fn ty(msg: &CastMessage) -> String {
    msg.message_type().unwrap_or_default()
}

impl FakeConn {
    /// Next message of type `wanted`, skipping (and answering PINGs
    /// among) everything else.
    async fn expect(&mut self, wanted: &str) -> CastMessage {
        loop {
            let msg = self.recv().await.expect("sender closed the connection");
            if ty(&msg) == wanted {
                return msg;
            }
            if ty(&msg) == "PING" {
                self.send(
                    namespace::HEARTBEAT,
                    DEFAULT_RECEIVER_ID,
                    &json!({"type": "PONG"}),
                )
                .await;
            }
        }
    }

    /// Types of every message until the sender closes the connection.
    async fn drain(&mut self) -> Vec<String> {
        let mut types = Vec::new();
        tokio::time::timeout(Duration::from_secs(2), async {
            while let Some(msg) = self.recv().await {
                types.push(ty(&msg));
            }
        })
        .await
        .expect("sender kept the connection open");
        types
    }

    async fn recv(&mut self) -> Option<CastMessage> {
        wire::recv(&mut self.reader, &mut self.buf)
            .await
            .ok()
            .flatten()
    }

    async fn send(&self, ns: &str, src: &str, payload: &Value) {
        let msg = CastMessage::new_json(src, DEFAULT_SENDER_ID, ns, payload).unwrap();
        self.writer.send(&msg).await.unwrap();
    }

    async fn send_receiver_status(&self, apps: &Value) {
        self.send(
            namespace::RECEIVER,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "RECEIVER_STATUS", "requestId": 0, "status": {"applications": apps}}),
        )
        .await;
    }

    /// Play the receiver side of a successful launch.
    async fn launch(&mut self) {
        self.expect("CONNECT").await;
        let launch = self.expect("LAUNCH").await;
        assert_eq!(launch.namespace, namespace::RECEIVER);
        // Half-populated first status, as some firmwares send.
        self.send_receiver_status(&json!([{"appId": DEFAULT_MEDIA_RECEIVER_APP_ID}]))
            .await;
        self.send_receiver_status(&json!([{
            "appId": DEFAULT_MEDIA_RECEIVER_APP_ID,
            "transportId": TRANSPORT,
            "sessionId": SESSION,
        }]))
        .await;
        let app_connect = self.expect("CONNECT").await;
        assert_eq!(app_connect.destination_id, TRANSPORT);
    }
}

fn fast_session() -> ChromecastSession {
    ChromecastSession {
        timing: Timing {
            heartbeat: Duration::from_millis(100),
            inactivity: Duration::from_millis(600),
            nudge_interval: Duration::from_millis(50),
        },
        ..ChromecastSession::default()
    }
}

async fn wait_dead(session: &ChromecastSession) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while session.is_alive() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("session never noticed the end");
}

async fn connected(fake: &FakeCast) -> (ChromecastSession, FakeConn) {
    let mut session = fast_session();
    let device = fake.device();
    let (outcome, conn) = tokio::join!(session.connect(&device), async {
        let mut conn = fake.accept().await;
        conn.launch().await;
        conn
    });
    assert!(matches!(outcome.unwrap(), ConnectOutcome::Ready));
    assert!(session.is_alive());
    (session, conn)
}

#[tokio::test]
async fn heartbeat_flows_both_ways() {
    let fake = FakeCast::start().await;
    let (session, mut conn) = connected(&fake).await;

    // Our heartbeat reaches the receiver…
    conn.expect("PING").await;
    // …and the receiver's PING gets a PONG addressed back to it.
    conn.send(
        namespace::HEARTBEAT,
        DEFAULT_RECEIVER_ID,
        &json!({"type": "PING"}),
    )
    .await;
    let pong = conn.expect("PONG").await;
    assert_eq!(pong.destination_id, DEFAULT_RECEIVER_ID);
    assert!(session.is_alive());
}

#[tokio::test]
async fn another_app_taking_over_ends_with_remote_stop() {
    let fake = FakeCast::start().await;
    let (mut session, mut conn) = connected(&fake).await;

    conn.send_receiver_status(
        &json!([{"appId": "233637DE", "transportId": "yt", "sessionId": "yt-1"}]),
    )
    .await;
    wait_dead(&session).await;
    assert!(matches!(
        session.end_reason(),
        Some(FerricastError::RemoteStopped(_))
    ));
    session.stop().await.unwrap();
    // The reason survives stop() so the caller can still decide.
    assert!(matches!(
        session.end_reason(),
        Some(FerricastError::RemoteStopped(_))
    ));
    // The new app must not receive a STOP from us.
    let sent = conn.drain().await;
    assert!(!sent.iter().any(|t| t == "STOP"), "{sent:?}");
}

#[tokio::test]
async fn stop_tears_down_our_app() {
    let fake = FakeCast::start().await;
    let (mut session, mut conn) = connected(&fake).await;
    session.stop().await.unwrap();
    let sent = conn.drain().await;
    assert!(sent.iter().any(|t| t == "STOP"), "{sent:?}");
    assert!(session.end_reason().is_none());
}

#[tokio::test]
async fn app_close_ends_with_remote_stop() {
    let fake = FakeCast::start().await;
    let (session, conn) = connected(&fake).await;
    conn.send(namespace::CONNECTION, TRANSPORT, &json!({"type": "CLOSE"}))
        .await;
    wait_dead(&session).await;
    assert!(matches!(
        session.end_reason(),
        Some(FerricastError::RemoteStopped(_))
    ));
}

#[tokio::test]
async fn silent_receiver_is_detected() {
    let fake = FakeCast::start().await;
    let (session, conn) = connected(&fake).await;
    // Keep the socket open but never answer: a powered-off TV whose
    // TCP connection has not timed out yet.
    let started = tokio::time::Instant::now();
    wait_dead(&session).await;
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(matches!(
        session.end_reason(),
        Some(FerricastError::Connection(_))
    ));
    drop(conn);
}

#[tokio::test]
async fn launch_error_fails_connect_and_releases_the_connection() {
    let fake = FakeCast::start().await;
    let mut session = fast_session();
    let device = fake.device();
    let (outcome, ()) = tokio::join!(session.connect(&device), async {
        let mut conn = fake.accept().await;
        conn.expect("CONNECT").await;
        let launch = conn.expect("LAUNCH").await;
        let request_id = launch
            .parse_payload::<crate::castv2::GenericPayload>()
            .unwrap()
            .request_id;
        conn.send(
            namespace::RECEIVER,
            DEFAULT_RECEIVER_ID,
            &json!({"type": "LAUNCH_ERROR", "requestId": request_id, "reason": "NOT_ALLOWED"}),
        )
        .await;
        // The failed connect must not leave a task holding the channel:
        // the sender side has to close.
        tokio::time::timeout(Duration::from_secs(2), async {
            while conn.recv().await.is_some() {}
        })
        .await
        .expect("connection left open after a failed connect");
    });
    let err = outcome.unwrap_err();
    assert!(err.to_string().contains("NOT_ALLOWED"), "{err}");
    assert!(!session.is_alive());
}
