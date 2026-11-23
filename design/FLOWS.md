# ferricast — user flows

Stack: Rust + Freya (desktop GUI), zbus provider (Linux), tray-icon (Linux/Windows/macOS). Receiver is a separate window (or the same binary launched with `--receive`).

Two personas drive the design: **Sender** (someone wanting to throw content) and **Receiver** (a machine that *is* a cast target). One person can be both.

## 0 · Entry surfaces

| Surface | OS | Trigger | Default |
| --- | --- | --- | --- |
| **Tray popup** | all | Click tray icon → anchored popup | always-on, startup |
| **D-Bus service** | Linux | `org.gnome.NetworkDisplays` / portal `screen-cast` compatible API | runs headless, no popup |
| **GUI window** | all | Tray → "Open ferricast…" or `ferricast --window` | optional |
| **Receiver fullscreen** | all | `ferricast --receive` or systemd service | always advertises via mDNS |
| **CLI** | all | `ferricast send <device>`, `ferricast devices` | power users |

The GUI, D-Bus and tray are **three faces of the same state**. Starting a cast from the CLI appears live in the tray popup; stopping from the tray stops the D-Bus session. One source of truth.

## 1 · Sender · happy path

```mermaid
flowchart TD
    A[User clicks tray icon] --> B{Devices ready?}
    B -- yes --> C[Pick device]
    B -- no, empty --> D[Hint: "Searching network…"]
    D --> C
    C --> E{Needs source?}
    E -- yes --> F[Native portal if available]
    F -. portal declined .-> G[ferricast source picker]
    E -- no (audio-only to speaker) --> H
    F --> H
    G --> H[Negotiating codecs…]
    H -- ok --> I[LIVE · overlay card + tray badge]
    H -- codec mismatch --> J[Toast: fallback applied, continue]
    J --> I
    I --> K[User stops → tray / overlay / hotkey]
```

**UX decisions:**
- Peak-end (Law): the win moment is the first frame appearing on the TV. Keep the "negotiating" phase under 1.5 s, show nothing between portal dismiss and first frame if we can.
- Jakob (Law): on Linux, the **native portal** owns source selection unless it fails. Our custom picker is fallback only; its row layout mirrors the GNOME portal's order (monitors → windows → areas).
- Hick (Law): tray popup shows max 7 device rows; beyond that an "N more…" collapse row opens a scrollable list inside the same popup (no new window).
- Fitts: tray popup rows are 40 px high, name click = cast; a secondary dot opens device actions (rename, forget) so users don't need to go to settings.

## 2 · Sender · key flows

### 2.1 First-run (any OS)

1. App launched (or installed and auto-started).
2. **Permissions screen** — the only full-screen the sender ever sees. One line per OS:
   - macOS: "Screen Recording" (System Settings link).
   - Windows: "Audio loopback requires nothing" / "Firewall private + public" (toggle).
   - Linux Wayland: "Portal access" (xdg-desktop-portal check) + optional "wlr-screencopy direct" opt-in.
3. One **Cast your screen now** button → device picker in the tray popup (not a modal — teach the surface they will use every day).
4. If zero devices: brand empty-state with mDNS/Bonjour/SSDP badges showing which discovery is live.

### 2.2 Normal cast (tray)

1. Click tray → popup opens anchored to tray.
2. Device list grouped by protocol: **Nearby** (mDNS/Bonjour), **Paired recently** (by MAC/UUID history), **Entered manually**. Collapsed by default except "Nearby".
3. Click device → if device needs a source selection, our source picker slides in (replaces list in-place, Esc / arrow-back returns).
4. Live state: device row flips to **Casting** mode with ember border, "Stop" button replaces its name-click action, elapsed timer + chip chips for codec, resolution, bitrate.

### 2.3 Settings / Device actions

- Settings reachable via tray footer gear OR window → Settings tab.
- Device actions (right-click or ⋯ dot on row): Rename, Forget, Pin on top, Default audio-only, Set as default.

### 2.4 Receiver interaction (sender side)

When the picked device is another ferricast **receiver**, the sender sends its display name + capabilities; the receiver either auto-accepts (configured) or shows a confirm overlay with a 4-digit code (matched to pass).

## 3 · Receiver · flows

### 3.1 Idle screen (fullscreen)

Centered column, breathing cast badge (one motion token: `motion.breathing = 1600ms ease-in-out`), the device's chosen **display name** + a **4-digit code**. Big enough to read from a sofa.

### 3.2 Incoming cast

- If sender is pre-approved: slide-in toast top-right, auto-dismiss after 2 s, cast starts.
- If sender needs approval: dimmed overlay, `Accept` / `Reject` buttons, Accept has focus-ring (Fitts), auto-reject after 15 s with a visible countdown.
- FairPlay / DRM rejection: friendly line, link to logs (hidden `g` shortcut opens the log tail).

### 3.3 Live receiver

Video goes fullscreen, no chrome. A **pull-down handle** at the top edge (hover reveals a strip) lets viewer stop, open logs, mute audio. Peak-end: when the sender stops, we fade **to the idle hero**, not to black — reinforces identity, invites the next cast.

## 4 · Shared edge cases

| Case | Behaviour |
| --- | --- |
| Codec fallback | Toast with chip + "Why?" opens short popover; keeps playing |
| Network interruption <5 s | Overlay shows pulse (same breathing motion), no toast |
| Network interruption >5 s | Row flips to "Reconnecting… 0:04", stop button still works |
| Permission revoked mid-session | Overlay: "macOS removed Screen Recording. Fix and retry" + System Settings deeplink |
| FairPlay refused | Session ends, receiver shows line (not a popup) |
| No codec at all (sender) | Picker row dims, chip reads "no encoder" with link to install |
| Receiver loses network | Idle hero shows `offline` chip; returns silently when back |

## 5 · Screen list to build (what we'll draw in Penpot)

Sender (page 05–07):
- 05.a Tray popup · empty (searching)
- 05.b Tray popup · 1 device
- 05.c Tray popup · many devices (grouped)
- 05.d Tray popup · one live (collapsed)
- 05.e Tray popup · casting (expanded live card)
- 05.f Tray popup · codec fallback toast
- 06.a Window · Devices tab
- 06.b Window · Settings tab · General
- 06.c Window · Settings tab · Protocols
- 06.d Window · Activity (history)
- 07.a Source picker · wlroots / portal present (native)
- 07.b Source picker · our fallback · Monitors / Windows / Areas
- 07.c Source picker · one-window preview

Receiver (page 08):
- 08.a Idle hero (ferricast + code)
- 08.b Incoming cast · needs confirm
- 08.c Live receiver · hover reveals handle
- 08.d Reconnecting overlay
- 08.e FairPlay / DRM rejection line

Onboarding + permissions (page 09):
- 09.a First-run hero
- 09.b macOS permission card
- 09.c Windows permission card
- 09.d Linux portal card + Wayland-direct opt-in

States (page 10):
- 10.a Zero devices (sender) + listening badges
- 10.b 1 device (sender)
- 10.c Many devices (long names, non-Latin, emoji)
- 10.d Loading device after click
- 10.e Codec negotiating
- 10.f Codec failed
- 10.g Offline receiver
- 10.h Reduced motion variant of idle hero
