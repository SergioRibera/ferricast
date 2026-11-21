//! PIN or password prompt shown when a receiver asks to pair (AirPlay
//! displays a one-time PIN on the TV, or wants its access password).
//!
//! Opens in its own top-level window, like the source picker, so it
//! also appears when the main window is hidden (`--background`).
//! Closing the window or pressing Esc counts as cancelling; Enter in
//! the field answers once the input is complete. The device name sits
//! on its own line, cut with an ellipsis when long, so the prompt under
//! it wraps within the window.

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

    /// What to type, under the device name.
    fn prompt(self) -> String {
        match self {
            Self::Pin { digits } => format!("Enter the {digits}-digit PIN shown on the receiver."),
            Self::Password { rejected: false } => "Enter its AirPlay password.".into(),
            Self::Password { rejected: true } => {
                "Wrong password. Enter its AirPlay password again.".into()
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
            .with_size(420., 260.)
            .with_min_size(340., 240.)
            .with_resizable(true)
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

/// Show `pin`, the code an AirPlay sender asks its user to type, in its
/// own always-on-top window. The caller closes it.
pub async fn open_pin_window(platform: &Platform, pin: String) -> freya::winit::window::WindowId {
    let config = WindowConfig::new_app(PinDisplay { pin })
        .with_size(360., 220.)
        .with_resizable(false)
        .with_background((20, 20, 28))
        .with_app_id("rs.sergioribera.ferricast.Pin")
        .with_window_attributes(|attrs, _| {
            attrs
                .with_title("AirPlay PIN")
                .with_window_level(WindowLevel::AlwaysOnTop)
        });
    platform.launch_window(config).await
}

struct PinDisplay {
    pin: String,
}

impl App for PinDisplay {
    fn render(&self) -> impl IntoElement {
        rect()
            .expanded()
            .center()
            .vertical()
            .spacing(12.)
            .child(
                label()
                    .text("Enter this PIN on your device")
                    .font_size(15.)
                    .color((210, 210, 225)),
            )
            .child(
                label()
                    .text(self.pin.clone())
                    .font_size(56.)
                    .font_weight(FontWeight::BOLD)
                    .color((240, 240, 250)),
            )
    }
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
        let on_global_key_down = {
            let done = self.done.clone();
            move |e: Event<KeyboardEventData>| {
                if e.key == Key::Named(NamedKey::Escape) {
                    Self::reply(&done, PairingResponse::Cancelled);
                }
            }
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
            .on_global_key_down(on_global_key_down)
            .child(
                label()
                    .text(self.device_name.clone())
                    .width(Size::fill())
                    .max_lines(1)
                    .text_overflow(TextOverflow::Ellipsis)
                    .text_align(TextAlign::Center)
                    .font_size(16.)
                    .font_weight(FontWeight::BOLD)
                    .color((240, 240, 250)),
            )
            .child(
                label()
                    .text(secret.prompt())
                    .width(Size::fill())
                    .text_align(TextAlign::Center)
                    .font_size(14.)
                    .color((210, 210, 225)),
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
        assert!(!password.prompt().contains("Wrong"));
        assert!(
            Secret::Password { rejected: true }
                .prompt()
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
