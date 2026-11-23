# ferricast — design source

Single source of truth for product design across Linux, Windows, macOS, sender + receiver.
Everything derives from the tokens in `design/tokens.json` (and the Penpot local library).

## Identity

- **Name:** ferricast (always lowercase).
- **Voice:** direct, technical, humble. Short sentences. "Qualified guess over flashy promise." No exclamation marks. Error text says what, why, what to try. Spanish + English parity from day one.
- **Posture:** *honest hardware feel*. The app acknowledges that cast involves real codecs, real network, real portals. Hides nothing; mutes complexity. No "magic", no cloud icons.
- **Signature move:** the 45° cast wedge from the logo repeats as a 1-pixel positive bevel on the top-left corner of primary emphasis surfaces (device cards when live, receiver idle hero). Nowhere else. Scarcity is the point.

## Registers per surface

| Surface | Register | Lead rules |
| --- | --- | --- |
| Tray menu | operate | Jakob (OS pattern), Fitts (44 px hit), Hick (≤7 items) |
| Device picker (overlay) | operate | Fitts, Hick, Peak-end (success flow lands here) |
| Source picker (overlay) | operate | Native portal first, custom second; visual parity |
| GUI main window | operate | Density allowed; typography + alignment lead |
| Settings | operate | Progressive disclosure (Tesler); defaults good enough |
| Onboarding / permission | persuade restraint | Identity in frame; one call to action per step |
| Receiver idle fullscreen | persuade restraint | Scarcity: name + pairing code; big, calm, breathing |
| Receiver active overlay | operate | 2-second auto-hide; keyboard-reachable |

## Palette

Three-layer tokens: primitive → semantic → component. Light + dark parity.
The brand metal is a **desaturated iron-red**, shifted off the standard Rust-crate palette so it doesn't collide with the Rust language brand.

### Primitives

| Token | Hex | Notes |
| --- | --- | --- |
| `iron.50` | `#F6F1EC` | Warm neutral base, light mode canvas tint |
| `iron.100` | `#E8DED2` |  |
| `iron.300` | `#B0A090` |  |
| `iron.500` | `#6C5D4E` | Warm grey — neutral text light mode |
| `iron.700` | `#3A3128` | Headings light mode |
| `iron.900` | `#1C1814` | Near-black, headlines + logo stroke |
| `ember.300` | `#E88057` | Hover / active state accent |
| `ember.500` | `#C85A2D` | **Primary brand (cast)** — desaturated iron red |
| `ember.700` | `#8A3B1A` | Pressed state |
| `steel.50` | `#F2F5F8` |  |
| `steel.500` | `#5F6B78` | Neutral text dark-mode on light cards |
| `steel.900` | `#0E1218` | Dark canvas |
| `signal.cast` | `#C85A2D` | Live-casting badge (= ember.500) |
| `signal.ready` | `#4F8F6B` | Peer ready / green |
| `signal.warn` | `#C89A2D` | Codec fallback warning |
| `signal.error` | `#B7311F` | Peer error / lost |

### Semantic (light)

`bg.canvas = iron.50`, `bg.surface = white`, `bg.surface.alt = iron.100`, `fg.default = iron.900`, `fg.muted = iron.500`, `fg.subtle = iron.300`, `accent = ember.500`, `accent.fg = white`, `border = iron.100`, `focus.ring = ember.500 @ 60%`.

### Semantic (dark)

`bg.canvas = steel.900`, `bg.surface = #161B22`, `bg.surface.alt = #1D232C`, `fg.default = iron.50`, `fg.muted = #9AA4B1`, `fg.subtle = #5F6B78`, `accent = ember.500`, `accent.fg = iron.900`, `border = #262C35`, `focus.ring = ember.300 @ 70%`.

Contrast gates: body 4.5:1 min, UI 3:1 min. All token pairs above pass; re-check if palette shifts.

## Typography

- **Headings + brand:** `Inter` 700 (SIL OFL free, ships in flake). Fallback: system sans.
- **UI body:** `Inter` 500 (headings) / 400 (body).
- **Monospace (device IDs, codecs, logs):** `JetBrains Mono` 400. Fallback: system mono.

Type scale (desktop):

| Role | Size / line | Use |
| --- | --- | --- |
| display | 40 / 48 | Receiver idle name only |
| h1 | 24 / 32 | Window header |
| h2 | 18 / 24 | Section header |
| body.lg | 15 / 22 | Primary rows (device name) |
| body | 13 / 20 | Standard body |
| mono | 12 / 18 | Codec, bitrate, IP, ID |
| caption | 11 / 16 | Timestamps, hints |

Tray popup: use `body` as base (13), `h2` for popup header.
Receiver idle: `display` for name + `h1` for 4-digit code.

## Spacing

4/8 grid. Tokens: `space.4`, `space.8`, `space.12`, `space.16`, `space.24`, `space.32`, `space.48`, `space.64`.
- Row gap inside a card = `space.8`
- Section gap = `space.24`
- Window padding = `space.16` (tray) / `space.24` (window)
- Card internal padding = `space.12` (tray) / `space.16` (window)

## Radii + elevation

- `radius.xs = 4`, `radius.sm = 8`, `radius.md = 12`, `radius.lg = 16`, `radius.pill = 999`
- Tray popup card: `radius.md`. Devices rows: `radius.sm`. Buttons: `radius.sm`. Chips: `radius.pill`.
- Shadows: tray uses OS popover shadow (don't fake it); custom overlays use `shadow.overlay = 0 10 28 rgba(0,0,0,0.18)`.

## Motion

Reduced motion (user setting, OS pref) replaces all easing with a 0 ms cut. No decorative motion ever.
- `motion.enter = 180 ms · cubic-bezier(.2,.9,.2,1)` — popups, overlays
- `motion.exit = 140 ms · cubic-bezier(.4,0,.6,1)` — overlays
- `motion.state = 120 ms · linear` — row hover, button press
- `motion.live-pulse = 1600 ms · ease-in-out` — the cast badge "breathes" while live. ONE place.
- Never animate numbers (bitrate counters), never slide text, never spin loaders ≥ 1 s without a progress fraction.

## Density per form factor

- **Tray popup (360 × variable, ≤ 480):** compact density. Rows 40 high.
- **Window (desktop):** comfortable. Rows 56. Device cards 64.
- **Receiver fullscreen:** oversize type, centered column max 720.

## Icons

- Set: 20 px canvas, 1.5 px stroke, square caps, geometric.
- Protocol marks: AirPlay / Chromecast / Miracast / "custom" shown as *signal glyphs* in a 1-colour token form. We do not reproduce vendor logos in UI (trademark risk + visual noise) — we show a labelled chip with our own protocol glyph.

## Voice samples (copy)

- Empty tray: "No casting devices nearby. ferricast is listening on this network."
- Picker empty: "Searching your network — Bonjour, mDNS and SSDP are active."
- Codec fallback: "This Chromecast doesn't advertise AC3. Falling back to AAC-LC."
- Receiver idle: `ferricast` · `code 7482` · "Say the code to your sender, or pick this device by name."
- Error (FairPlay): "Your Mac asked for FairPlay DRM. ferricast can't verify this clip — try a different source."

## File locations

- Tokens: `design/tokens.json` (primitives + semantics) and mirrored in Penpot local library.
- Icons: `design/icons/*.svg`.
- Logo: `design/logo/`.
- Mockups: live in Penpot, exported snapshots in `design/screens/`.
