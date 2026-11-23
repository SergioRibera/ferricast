//! The reverse HTTP connection: a sender opens a connection with
//! `POST /reverse` (`Upgrade: PTTH/1.0`), the receiver answers
//! `101 Switching Protocols` and from then on sends *requests* on it:
//! `POST /event` with the playback state. The sender's replies are read
//! and discarded.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use plist::{Dictionary, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use tracing::{debug, info};

use super::EventState;

/// Events queued for one reverse connection, at most.
const EVENT_QUEUE: usize = 16;

/// One `/event` for the sender.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::receiver) struct Event {
    pub state: EventState,
    /// The sender's id of the playback (`uuid` of `/play`).
    pub playback_id: Option<String>,
}

impl Event {
    /// Wire form: an XML plist, as Apple TVs send it.
    fn encode(&self, session_id: &str) -> Vec<u8> {
        let mut dict = Dictionary::new();
        dict.insert("category".into(), Value::String("video".into()));
        dict.insert("sessionID".into(), Value::Integer(1.into()));
        dict.insert("state".into(), Value::String(self.state.as_str().into()));
        if let Some(id) = &self.playback_id {
            dict.insert("uuid".into(), Value::String(id.clone()));
        }
        let mut body = Vec::new();
        // Writing a dictionary of strings and integers into a Vec cannot fail.
        let _ = plist::to_writer_xml(&mut body, &Value::Dictionary(dict));
        let mut wire = format!(
            "POST /event HTTP/1.1\r\n\
             Content-Type: text/x-apple-plist+xml\r\n\
             Content-Length: {}\r\n\
             X-Apple-Session-ID: {session_id}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        wire.extend_from_slice(&body);
        wire
    }
}

/// The reverse connections of every sender, by session key.
#[derive(Default)]
pub(in crate::receiver) struct ReverseHub {
    channels: Mutex<HashMap<String, mpsc::Sender<Event>>>,
}

impl ReverseHub {
    /// Sends `event` to the sender of session `key`, if it opened a
    /// reverse connection. Never waits: a stalled sender loses events.
    pub fn send(&self, key: &str, event: Event) {
        let channels = self.channels.lock().unwrap_or_else(PoisonError::into_inner);
        match channels.get(key) {
            Some(tx) => {
                if tx.try_send(event).is_err() {
                    debug!(
                        key,
                        "AirPlay event dropped: reverse connection busy or gone"
                    );
                }
            }
            None => debug!(key, ?event, "no reverse connection for the AirPlay event"),
        }
    }

    fn register(&self, key: &str) -> (mpsc::Sender<Event>, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel(EVENT_QUEUE);
        self.channels
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key.to_owned(), tx.clone());
        (tx, rx)
    }

    /// Forgets `key`, unless a newer reverse connection replaced `tx`.
    fn unregister(&self, key: &str, tx: &mpsc::Sender<Event>) {
        let mut channels = self.channels.lock().unwrap_or_else(PoisonError::into_inner);
        if channels
            .get(key)
            .is_some_and(|current| current.same_channel(tx))
        {
            channels.remove(key);
        }
    }

    /// Serves a connection the sender reversed: writes the events of
    /// session `key` until either side closes it.
    pub async fn serve<S>(&self, key: &str, mut stream: S)
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        info!(key, "AirPlay reverse connection open");
        let (tx, mut rx) = self.register(key);
        let mut discard = [0_u8; 4096];
        loop {
            tokio::select! {
                event = rx.recv() => {
                    // `tx` is held here, so the channel never closes.
                    let Some(event) = event else { break };
                    debug!(key, state = event.state.as_str(), "AirPlay event");
                    let wire = event.encode(key);
                    if stream.write_all(&wire).await.is_err() || stream.flush().await.is_err() {
                        break;
                    }
                }
                read = stream.read(&mut discard) => match read {
                    // The sender's replies to the events carry nothing.
                    Ok(n) if n > 0 => {}
                    Ok(_) | Err(_) => break,
                },
            }
        }
        self.unregister(key, &tx);
        let _ = stream.shutdown().await;
        debug!(key, "AirPlay reverse connection closed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn events_reach_the_reversed_connection() {
        let hub = std::sync::Arc::new(ReverseHub::default());
        let (mut client, server) = tokio::io::duplex(8192);
        let serving = tokio::spawn({
            let hub = hub.clone();
            async move { hub.serve("S1", server).await }
        });
        // Registered once the task runs.
        while hub
            .channels
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty()
        {
            tokio::task::yield_now().await;
        }
        hub.send(
            "S1",
            Event {
                state: EventState::Paused,
                playback_id: Some("U1".into()),
            },
        );
        let mut buf = vec![0_u8; 4096];
        let n = client.read(&mut buf).await.unwrap();
        let text = String::from_utf8_lossy(&buf[..n]).into_owned();
        assert!(text.starts_with("POST /event HTTP/1.1\r\n"), "{text}");
        assert!(text.contains("X-Apple-Session-ID: S1\r\n"), "{text}");
        let body = &buf[text.find("\r\n\r\n").unwrap() + 4..n];
        let dict = plist::from_bytes::<Value>(body)
            .unwrap()
            .into_dictionary()
            .unwrap();
        assert_eq!(dict.get("state").and_then(Value::as_string), Some("paused"));
        assert_eq!(
            dict.get("category").and_then(Value::as_string),
            Some("video")
        );
        assert_eq!(dict.get("uuid").and_then(Value::as_string), Some("U1"));
        client
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        drop(client);
        serving.await.unwrap();
        assert!(
            hub.channels
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
        );
    }
}
