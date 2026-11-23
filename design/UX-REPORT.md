# UX report · ferricast

**Intent:** Build (full identity + full UX from scratch).
**Stack:** Rust workspace · Freya desktop GUI · zbus D-Bus provider · multi-platform tray.
**Form factor:** desktop (Linux, Windows, macOS) + fullscreen receiver.
**Register:** operate for every live surface (tray, picker, window, overlays); persuade-restraint for onboarding + receiver idle only.
**Depth:** build (no review loop yet — screens are just drawn).
**Skills run:** `definitive-ux` router · `logo-design` · `design-source` · `map-user-flows` · `laws-of-ux` (as decision layer) · `break-ui`.

**Verdict:** **Ship with polish list** — identity and system are decided, 24 Penpot screens cover the signature flows, token sets map 1-to-1 to a Freya theme, brand kit is already in the repo. Polish list below is for the implementation pass; no blocking design question remains open.

## Delivery

| Where | What |
| --- | --- |
| `assets/brand/` | Brand kit — symbol (+ small-cut), lockups (outlined), colour variants, Linux hicolor tree, Windows `.ico`, macOS iconset, favicon + webmanifest, tray template/live/attention, `ferricast.desktop`, `palette.json`, `BRAND.md` |
| `design/DESIGN.md` | Identity, registers per surface, palette strategy, type scale, spacing, radii, motion, density, voice |
| `design/tokens.json` | Three-layer tokens (primitive · semantic · component) — ready to generate a `theme.rs` |
| `design/FLOWS.md` | Sender + receiver user flows, screen list |
| `design/BREAK-UI.md` | Worst-case data pass: what breaks, what was fixed, what still needs real hardware |
| Penpot file | 10 pages: Foundations · Components · Flow sender · Flow receiver · Sender tray · Sender window · Picker overlays · Receiver · Onboarding · States + edge cases |

Logo chosen: **A — Cast-iron F** (monogram with two 45° cast wedges). Rationale and kit usage in `assets/brand/BRAND.md`.

## Blocking

None.

## Important (pick these up during the Freya implementation pass)

1. **Freya `theme.rs` generator.** Pull `design/tokens.json` into a `build.rs` that writes a module exposing semantic names (`fg.default`, `bg.surface`, etc.). Rationale: hard-coding hex into components will drift; the three-layer design pivots on resolved-value correctness.
2. **Linux tray is hybrid, not a popup port.** SNI/Dbusmenu only lets the shell render plain menus. Ship a minimal DBusmenu (Stop · Nearby submenu · Open ferricast… · Settings · Quit) plus a frameless rich window-popup that opens on left-click or hotkey. See page 05 variants G (DBusmenu), H (window-popup), I (trigger map).
3. **Tray template icon honoured per OS.** On macOS, the SVG must be marked template; on KDE Plasma 6 Wayland, verify the `-symbolic` suffix route (BREAK-UI risk). Ship both `-mono` and a coloured fallback.
4. **D-Bus provider surfaces same state as tray/window.** The tray shows real-time sessions that may originate from D-Bus, CLI, or GUI. Treat D-Bus as a UI client, not a shadow path.
5. **Receiver idle must be AMOLED-safe when target screen reports OLED.** Auto-swap to `receiver.amoled` tokens when `is-oled` is true (Wayland output name match on known brands as first heuristic).
6. **Reduced motion is system-level, not app-level.** Hook into `prefers-reduced-motion` via `mundy` on macOS/Linux + the Windows Settings API. Breathing dot is the single motion that MUST die when it's set. The Accept-button countdown collapses to 3 discrete steps (0/50/100%).
7. **Noise policy.** No raw fingerprints, IP:port strings, BSSIDs, RSSI or MAC addresses in UI. If the user cannot act on it, it goes to `Settings · About · Copy diagnostics`, not the main surfaces.

## Polish

- Row truncation uses ellipsis with a 4-char suffix revealed on hover (nice for identifying `-lab` and `-dev` suffixes that disambiguate dev boxes).
- "codec fallback" chip gets a tooltip = the full codec negotiation log excerpt.
- Codec-failure banner links to an OS-specific install guide (Nix package name on NixOS, `winget` on Windows, Homebrew on macOS).
- Settings page ships "Export logs" button near Help; FairPlay tickets are useless without them.

## Risks (need data to confirm)

- AirPlay FairPlay against latest tvOS — handshake only testable with a real receiver.
- Miracast on Win 11 24H2 WiFi-switch session drops.
- Chromecast audio-only chip extraction logic against newest Audio devices.
- KDE Plasma 6 Wayland `-symbolic` icon theme rules.

(See `design/BREAK-UI.md` → "Still unknown" for context.)

## Suggestions (product-side, stated as assumptions)

- Default identity is "network-local". If user wants cross-network relays later, wire it as a separate signed service — never fold it into the core app's UI. Keeps the privacy claim ("no cloud, no account") credible.
- CLI surface = `ferricast devices`, `ferricast send <device>`, `ferricast --receive`, `ferricast --window`. Reflect these in the `.desktop` actions (already done) and help in the window.
- Allow custom device-name override on receiver. The default is the hostname, which is ugly on Linux boxes (`livingroom-pi`). Settings → Receiver → Display name.

## Keep

- Lowercase brand everywhere — including window title.
- "Cast now" is primary across every surface. Verb aligned.
- Tray-first, window-optional. GUI is the fallback, not the default.
- Native portal first, our picker second — mirror portal row order so muscle memory survives.

## Verification run

- 24 Penpot screens render with tokens.
- Brand kit audit 99–100 on master vectors.
- Icons generated for Linux hicolor (SVG + PNG 16…512), Windows `.ico` multi-size, macOS iconset, web favicon + manifest.
- Break-ui dataset applied to tray, window, picker, receiver — findings logged.

## Next step (single highest-leverage change)

Generate the Freya theme from `design/tokens.json`. Everything else can proceed on top of a live theme; without it, the first component merged will fork the palette and drift.

**Concrete first commit:** add `build.rs` in `crates/core/` that reads `design/tokens.json` and writes `src/theme/generated.rs` exposing `Color`, `Space`, `Radius`, `Motion` modules keyed by semantic names, with a `const THEME_LIGHT` / `THEME_DARK` struct. Public API matches what the Penpot Components page assumes.
