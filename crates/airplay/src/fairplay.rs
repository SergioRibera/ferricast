//! FairPlay `fp-setup` and the `ekey` / base AES key used in `SETUP`.

use ferricast_core::Result;

use crate::rtsp::RtspClient;

/// `ekey` to send in `SETUP` and the AES key the receiver will derive from it.
pub(crate) struct FairPlayKey {
    pub ekey: Vec<u8>,
    pub aes_key: [u8; 16],
}

/// Run `fp-setup` with the precomputed key message.
#[cfg(feature = "fairplay")]
pub(crate) async fn setup(rtsp: &mut RtspClient) -> Result<FairPlayKey> {
    use ferricast_core::FerricastError;
    use ferricast_fairplay as fp;
    use tracing::warn;

    use crate::rtsp::CONTENT_OCTET;

    let headers = [("X-Apple-ET", "32")];
    let reply = rtsp
        .request_with(
            "POST",
            "/fp-setup",
            Some(CONTENT_OCTET),
            &fp::setup_message1(fp::MODE),
            &headers,
        )
        .await?
        .ensure_success("fp-setup (1)")?;
    if reply.body.len() != fp::REPLY1_LEN {
        return Err(FerricastError::Protocol(format!(
            "fp-setup (1): expected {} bytes, got {}",
            fp::REPLY1_LEN,
            reply.body.len()
        )));
    }
    let reply = rtsp
        .request_with(
            "POST",
            "/fp-setup",
            Some(CONTENT_OCTET),
            &fp::KEY_MESSAGE,
            &headers,
        )
        .await?
        .ensure_success("fp-setup (2)")?;
    if reply.body != fp::expected_reply2() {
        // Receivers that verify the key message answer a rejected one with
        // something else, then cannot unwrap the `ekey` and drop the
        // session on the mirroring `SETUP`.
        warn!(
            reply = %hex::encode(&reply.body),
            "fp-setup (2) reply differs from the expected echo; receiver rejected the key message"
        );
    }
    Ok(FairPlayKey {
        ekey: fp::EKEY.to_vec(),
        aes_key: fp::AES_KEY,
    })
}

/// Without FairPlay support `fp-setup` is skipped. UxPlay / RPiPlay then
/// fail `fairplay_decrypt` and keep an all-zero base key, which is what we
/// assume here.
#[cfg(not(feature = "fairplay"))]
#[expect(
    clippy::unused_async,
    reason = "same signature as the FairPlay-enabled build"
)]
pub(crate) async fn setup(_rtsp: &mut RtspClient) -> Result<FairPlayKey> {
    Ok(FairPlayKey {
        ekey: vec![0; 72],
        aes_key: [0; 16],
    })
}
