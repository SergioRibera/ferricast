//! The AirPlay video requests of a control connection: plain HTTP/1.1
//! carrying an `X-Apple-Session-ID`, never encrypted.

use plist::{Dictionary, Value};
use tracing::{debug, info, warn};

use super::{Connection, Next};
use crate::receiver::rtsp::{Request, Response};
use crate::receiver::video::{
    PlayRequest, VideoHandle, VideoRequest, rate_playing, scrub_position_us,
};
use crate::receiver::{MODEL, SOURCE_VERSION};

/// Content type of the XML plists video senders exchange.
const XML_PLIST: &str = "text/x-apple-plist+xml";
const PARAMETERS: &str = "text/parameters";
/// `features` of `/server-info`: the video-era bit set (video, photo,
/// FairPlay video, volume, HLS, slideshow, screen, audio), as Apple TVs
/// and UxPlay report it.
const SERVER_INFO_FEATURES: u64 = 0x27F;
/// Build reported by `/server-info`.
const OS_BUILD: &str = "12B435";

fn xml_response(dict: Dictionary) -> Response {
    let mut body = Vec::new();
    match plist::to_writer_xml(&mut body, &Value::Dictionary(dict)) {
        Ok(()) => Response::ok().body(XML_PLIST, body),
        Err(e) => {
            warn!(error = %e, "cannot encode a video reply");
            Response::status(500, "Internal Server Error")
        }
    }
}

/// `{"errorCode": 0}`, what `setProperty` answers.
fn no_error() -> Response {
    let mut dict = Dictionary::new();
    dict.insert("errorCode".into(), Value::Integer(0.into()));
    xml_response(dict)
}

impl Connection {
    /// Answers the request when it is an AirPlay video one.
    pub(super) async fn handle_video(&mut self, request: &Request) -> Option<(Response, Next)> {
        let route = (request.method.as_str(), request.path());
        let is_video = matches!(
            route,
            (
                "POST",
                "/reverse" | "/play" | "/scrub" | "/rate" | "/stop" | "/action"
            ) | ("GET", "/playback-info" | "/scrub" | "/server-info")
                | ("PUT", "/setProperty")
                | ("POST" | "GET", "/getProperty")
                | ("POST", "/fp-setup2")
        );
        if !is_video {
            return None;
        }
        if !self.shared.settings.config.video {
            debug!(?route, "AirPlay video request while video is off");
            return Some((Response::status(404, "Not Found"), Next::Continue));
        }
        let response = match route {
            ("POST", "/reverse") => {
                let key = self.video_key(request);
                let purpose = request.header("X-Apple-Purpose").unwrap_or("event");
                info!(peer = %self.peer, %key, purpose, "AirPlay sender reverses a connection");
                let response = Response::status(101, "Switching Protocols")
                    .header("Upgrade", "PTTH/1.0")
                    .header("Connection", "Upgrade");
                return Some((response, Next::Reverse(key)));
            }
            ("POST", "/play") => self.play(request).await,
            ("GET", "/playback-info") => {
                let info = self.video(request).ask(VideoRequest::Info).await.flatten();
                // UxPlay and Apple TVs answer an empty body when nothing
                // plays; senders read it as "playback over".
                info.map_or_else(Response::ok, xml_response)
            }
            ("GET", "/scrub") => {
                let text = self
                    .video(request)
                    .ask(VideoRequest::Position)
                    .await
                    .flatten()
                    .unwrap_or_else(|| "duration: 0.000000\nposition: 0.000000\n".to_owned());
                Response::ok().body(PARAMETERS, text.into_bytes())
            }
            ("POST", "/scrub") => match scrub_position_us(&request.uri) {
                Some(position_us) => {
                    debug!(position_us, "AirPlay video scrub");
                    self.video(request)
                        .send(VideoRequest::Seek(position_us))
                        .await;
                    Response::ok()
                }
                None => Response::status(400, "Bad Request"),
            },
            ("POST", "/rate") => match rate_playing(&request.uri) {
                Some(playing) => {
                    debug!(playing, "AirPlay video rate");
                    self.video(request).send(VideoRequest::Rate(playing)).await;
                    Response::ok()
                }
                None => Response::status(400, "Bad Request"),
            },
            ("POST", "/stop") => {
                self.video(request).send(VideoRequest::Stop).await;
                Response::ok()
            }
            ("POST", "/action") => self.action(request).await,
            ("GET", "/server-info") => self.server_info(),
            ("PUT", "/setProperty") => {
                let name = request.uri.split_once('?').map_or("", |(_, q)| q);
                debug!(name, "AirPlay video setProperty");
                no_error()
            }
            ("POST" | "GET", "/getProperty") => {
                let name = request.uri.split_once('?').map_or("", |(_, q)| q);
                debug!(name, "AirPlay video getProperty");
                no_error()
            }
            // FairPlay-protected video (bit 2) needs keys only Apple TVs have.
            _ => Response::status(421, "Misdirected Request"),
        };
        Some((response, Next::Continue))
    }

    /// The sender's session: `X-Apple-Session-ID`, or its address for
    /// senders that send none.
    fn video_key(&self, request: &Request) -> String {
        request.header("X-Apple-Session-ID").map_or_else(
            || format!("peer-{}", self.peer.ip()),
            |id| id.trim().to_owned(),
        )
    }

    /// The video session the request belongs to; the connection keeps it
    /// alive from now on.
    fn video(&mut self, request: &Request) -> VideoHandle {
        let key = self.video_key(request);
        if let Some(handle) = &self.video
            && handle.key() == key
        {
            return handle.clone();
        }
        let handle = self
            .shared
            .videos
            .handle(&self.shared, &self.ready, &key, self.peer.ip());
        self.video = Some(handle.clone());
        handle
    }

    async fn play(&mut self, request: &Request) -> Response {
        let play = match PlayRequest::parse(&request.body) {
            Ok(play) => play,
            Err(e) => {
                warn!(peer = %self.peer, error = %e, "AirPlay video: bad /play");
                return Response::status(400, "Bad Request");
            }
        };
        // Video replaces whatever this connection mirrored.
        self.media = None;
        let handle = self.video(request);
        match handle.ask(|reply| VideoRequest::Play(play, reply)).await {
            Some(Ok(())) => Response::ok(),
            Some(Err(reason)) => {
                warn!(peer = %self.peer, reason, "AirPlay video: /play refused");
                Response::status(400, "Bad Request")
            }
            None => Response::status(500, "Internal Server Error"),
        }
    }

    /// `/action`: playlist edits and the YouTube app's playlist fetches.
    async fn action(&mut self, request: &Request) -> Response {
        let Some(dict) = plist::from_bytes::<Value>(&request.body)
            .ok()
            .and_then(Value::into_dictionary)
        else {
            return Response::status(400, "Bad Request");
        };
        let kind = dict.get("type").and_then(Value::as_string).unwrap_or("");
        let item_uuid = dict
            .get("params")
            .and_then(Value::as_dictionary)
            .and_then(|p| p.get("item"))
            .and_then(Value::as_dictionary)
            .and_then(|item| item.get("uuid"))
            .and_then(Value::as_string)
            .map(str::to_owned);
        match (kind, item_uuid) {
            ("playlistRemove", Some(uuid)) => {
                debug!(%uuid, "AirPlay video: playlistRemove");
                self.video(request).send(VideoRequest::Remove(uuid)).await;
            }
            // Only sent in answer to `unhandledURLRequest` events, which
            // this receiver never sends (YouTube's mlhls:// is refused).
            ("unhandledURLResponse", _) => {
                debug!("AirPlay video: unexpected unhandledURLResponse");
            }
            (kind, _) => debug!(kind, "AirPlay video action ignored"),
        }
        Response::ok()
    }

    fn server_info(&self) -> Response {
        let id = self
            .shared
            .settings
            .identity
            .device_id_colons()
            .to_uppercase();
        let mut dict = Dictionary::new();
        dict.insert("deviceid".into(), Value::String(id.clone()));
        dict.insert(
            "features".into(),
            Value::Integer(SERVER_INFO_FEATURES.into()),
        );
        dict.insert("macAddress".into(), Value::String(id));
        dict.insert("model".into(), Value::String(MODEL.into()));
        dict.insert("osBuildVersion".into(), Value::String(OS_BUILD.into()));
        dict.insert("protovers".into(), Value::String("1.0".into()));
        dict.insert("srcvers".into(), Value::String(SOURCE_VERSION.into()));
        dict.insert("vv".into(), Value::Integer(2.into()));
        xml_response(dict)
    }
}
