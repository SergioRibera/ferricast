//! PIN or password prompt shown when a receiver asks to pair (AirPlay
//! displays a one-time PIN on the TV, or wants its access password).
//!
//! Opens in its own top-level window, like the source picker, so it
//! also appears when the main window is hidden (`--background`).
//! Closing the window counts as cancelling.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use ferricast::PairingResponse;
use freya::prelude::*;
use freya::winit::window::WindowLevel;
use tokio::sync::oneshot;

/// The manager's one-shot answer slot from `ManagerEvent::PairingRequired`.
pub type PairingResponder = Arc<tokio::sync::Mutex<Option<oneshot::Sender<PairingResponse>>>>;

/// Answer `respond` without asking the user.
pub fn answer(respond: &PairingResponder, response: PairingResponse) {
    match respond.try_lock() {
        Ok(mut slot) => {
            if let Some(tx) = slot.take() {
                let _ = tx.send(response);
            }
        }
        Err(_) => tracing::warn!("pairing answer slot busy; prompt left unanswered"),
    }
}

/// What the pairing dialog asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Secret {
    /// A PIN of `digits` digits shown on the receiver.
    Pin { digits: u8 },
    /// The receiver's access password.
    /// The receiver's access password; `rejected` when the previous
    /// answer was refused.
    Password { rejected: bool },
}

impl Secret {
    /// Whether `text` can still become a valid answer while typing.
    fn accepts_partial(self, text: &str) -> bool {
        match self {
            Self::Pin { digits } => is_partial_pin(text, usize::from(digits)),
            Self::Password { .. } => true,
        }
    }

    /// Whether `text` is a complete answer.
    fn is_complete(self, text: &str) -> bool {
        match self {
            Self::Pin { digits } => {
                text.len() == usize::from(digits) && is_partial_pin(text, usize::from(digits))
            }
            Self::Password { .. } => !text.is_empty(),
        }
    }

    fn response(self, text: String) -> PairingResponse {
        match self {
            Self::Pin { .. } => PairingResponse::Pin(text),
            Self::Password { .. } => PairingResponse::Password(text),
        }
    }

    fn prompt(self, device_name: &str) -> String {
        match self {
            Self::Pin { digits } => format!("Enter the {digits}-digit PIN shown on {device_name}"),
            Self::Password { rejected: false } => {
                format!("Enter the AirPlay password of {device_name}")
            }
            Self::Password { rejected: true } => {
                format!("Wrong password. Enter the AirPlay password of {device_name} again")
            }
        }
    }
}

/// Ask for `secret` for `device_name` and hand the answer (or
/// `Cancelled`) to the manager.
pub fn open_pairing_dialog(
    platform: Platform,
    device_name: String,
    secret: Secret,
    respond: PairingResponder,
) {
    spawn(async move {
        let (tx, rx) = oneshot::channel::<PairingResponse>();
        let app = PairingDialog {
            device_name: device_name.clone(),
            secret,
            done: Rc::new(RefCell::new(Some(tx))),
        };
        let config = WindowConfig::new_app(app)
            .with_size(380., 220.)
            .with_background((20, 20, 28))
            .with_app_id("rs.sergioribera.ferricast.Pairing")
            .with_window_attributes(move |attrs, _| {
                attrs
                    .with_title(format!("Pair with {device_name}"))
                    .with_window_level(WindowLevel::AlwaysOnTop)
            });
        let wid = platform.launch_window(config).await;
        platform.focus_window(Some(wid));
        // A dropped sender means the window was closed.
        let response = rx.await.unwrap_or(PairingResponse::Cancelled);
        platform.close_window(wid);
        if let Some(tx) = respond.lock().await.take() {
            let _ = tx.send(response);
        }
    });
}

/// Whether `text` can still become a valid PIN of `digits` digits.
fn is_partial_pin(text: &str, digits: usize) -> bool {
    text.len() <= digits && text.chars().all(|c| c.is_ascii_digit())
}

struct PairingDialog {
    device_name: String,
    secret: Secret,
    done: Rc<RefCell<Option<oneshot::Sender<PairingResponse>>>>,
}

impl PairingDialog {
    fn reply(done: &Rc<RefCell<Option<oneshot::Sender<PairingResponse>>>>, r: PairingResponse) {
        if let Some(tx) = done.borrow_mut().take() {
            let _ = tx.send(r);
        }
    }
}

impl App for PairingDialog {
    fn render(&self) -> impl IntoElement {
        let text = use_state(String::new);
        let secret = self.secret;
        let ready = secret.is_complete(&text.read());

        let on_submit = {
            let done = self.done.clone();
            move |value: String| {
                if secret.is_complete(&value) {
                    Self::reply(&done, secret.response(value));
                }
            }
        };
        let on_pair = {
            let done = self.done.clone();
            move |_| Self::reply(&done, secret.response(text.read().clone()))
        };
        let on_cancel = {
            let done = self.done.clone();
            move |_| Self::reply(&done, PairingResponse::Cancelled)
        };
        let (placeholder, mode, width) = match secret {
            Secret::Pin { .. } => ("PIN", InputMode::Shown, 160.),
            Secret::Password { .. } => ("Password", InputMode::new_password(), 240.),
        };

        rect()
            .expanded()
            .center()
            .padding(20.)
            .vertical()
            .spacing(14.)
            .child(
                label()
                    .text(secret.prompt(&self.device_name))
                    .font_size(14.)
                    .color((230, 230, 240)),
            )
            .child(
                Input::new(text)
                    .placeholder(placeholder)
                    .mode(mode)
                    .auto_focus(true)
                    .width(Size::px(width))
                    .on_validate(move |v: InputValidator| {
                        let valid = secret.accepts_partial(&v.text());
                        v.set_valid(valid);
                    })
                    .on_submit(on_submit),
            )
            .child(
                rect()
                    .horizontal()
                    .spacing(10.)
                    .child(Button::new().on_press(on_cancel).child("Cancel"))
                    .child(
                        Button::new()
                            .filled()
                            .enabled(ready)
                            .on_press(on_pair)
                            .child("Pair"),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_pins_are_digits_up_to_the_length() {
        assert!(is_partial_pin("", 4));
        assert!(is_partial_pin("12", 4));
        assert!(is_partial_pin("1234", 4));
        assert!(!is_partial_pin("12345", 4));
        assert!(!is_partial_pin("12a", 4));
    }

    #[test]
    fn secrets_validate_and_answer_by_kind() {
        let pin = Secret::Pin { digits: 4 };
        assert!(pin.accepts_partial("12"));
        assert!(!pin.accepts_partial("12a"));
        assert!(!pin.is_complete("123"));
        assert!(pin.is_complete("1234"));
        assert!(matches!(pin.response("1234".into()), PairingResponse::Pin(p) if p == "1234"));

        let password = Secret::Password { rejected: false };
        assert!(password.accepts_partial("any text 12!"));
        assert!(!password.is_complete(""));
        assert!(password.is_complete("s"));
        assert!(matches!(
            password.response("secret".into()),
            PairingResponse::Password(p) if p == "secret"
        ));
        assert!(!password.prompt("TV").contains("Wrong"));
        assert!(
            Secret::Password { rejected: true }
                .prompt("TV")
                .starts_with("Wrong password")
        );
    }

    #[tokio::test]
    async fn answer_sends_once() {
        let (tx, rx) = oneshot::channel();
        let respond: PairingResponder = Arc::new(tokio::sync::Mutex::new(Some(tx)));
        answer(&respond, PairingResponse::Confirmed);
        answer(&respond, PairingResponse::Cancelled);
        assert!(matches!(rx.await, Ok(PairingResponse::Confirmed)));
    }
}
