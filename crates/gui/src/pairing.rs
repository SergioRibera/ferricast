//! PIN prompt shown when a receiver asks to pair (AirPlay displays a
//! one-time PIN on the TV).
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

/// Ask for a `digits`-long PIN for `device_name` and hand the answer
/// (or `Cancelled`) to the manager.
pub fn open_pin_dialog(
    platform: Platform,
    device_name: String,
    digits: u8,
    respond: PairingResponder,
) {
    spawn(async move {
        let (tx, rx) = oneshot::channel::<PairingResponse>();
        let app = PinDialog {
            device_name: device_name.clone(),
            digits,
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

struct PinDialog {
    device_name: String,
    digits: u8,
    done: Rc<RefCell<Option<oneshot::Sender<PairingResponse>>>>,
}

impl PinDialog {
    fn reply(done: &Rc<RefCell<Option<oneshot::Sender<PairingResponse>>>>, r: PairingResponse) {
        if let Some(tx) = done.borrow_mut().take() {
            let _ = tx.send(r);
        }
    }
}

impl App for PinDialog {
    fn render(&self) -> impl IntoElement {
        let pin = use_state(String::new);
        let digits = usize::from(self.digits);
        let ready = pin.read().len() == digits;

        let on_submit = {
            let done = self.done.clone();
            move |text: String| {
                if text.len() == digits {
                    Self::reply(&done, PairingResponse::Pin(text));
                }
            }
        };
        let on_pair = {
            let done = self.done.clone();
            move |_| Self::reply(&done, PairingResponse::Pin(pin.read().clone()))
        };
        let on_cancel = {
            let done = self.done.clone();
            move |_| Self::reply(&done, PairingResponse::Cancelled)
        };

        rect()
            .expanded()
            .center()
            .padding(20.)
            .vertical()
            .spacing(14.)
            .child(
                label()
                    .text(format!(
                        "Enter the {digits}-digit PIN shown on {}",
                        self.device_name
                    ))
                    .font_size(14.)
                    .color((230, 230, 240)),
            )
            .child(
                Input::new(pin)
                    .placeholder("PIN")
                    .auto_focus(true)
                    .width(Size::px(160.))
                    .on_validate(move |v: InputValidator| {
                        let valid = is_partial_pin(&v.text(), digits);
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

    #[tokio::test]
    async fn answer_sends_once() {
        let (tx, rx) = oneshot::channel();
        let respond: PairingResponder = Arc::new(tokio::sync::Mutex::new(Some(tx)));
        answer(&respond, PairingResponse::Confirmed);
        answer(&respond, PairingResponse::Cancelled);
        assert!(matches!(rx.await, Ok(PairingResponse::Confirmed)));
    }
}
