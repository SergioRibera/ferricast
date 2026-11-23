# ferricast — break-ui pass

Worst-case data fed to each surface, what broke and what we did about it. Page 10 in Penpot (`10 · States + edge cases`) renders the visual tests alongside this doc.

## Datasets

1. **Zero.** No devices on the network.
2. **One.** Exactly one device, happiest path.
3. **Many.** 5–7 devices, mixed protocols, mixed states.
4. **Long names.** `Projector-ABCDE-FGHIJ-KLMNO-PQRST`, `home-office-chromecast-v2-lab`.
5. **Emoji + non-Latin.** `Tía Rosa’s 🎬 Living Room — 4K OLED`, `リビング・サムスン TV`.
6. **One character.** `a` — device name is a single letter.
7. **Codec negotiation stalls.** AC3 unavailable, AAC-LC pending.
8. **Codec failure.** No encoder backend compiled in → session can't start.
9. **Network offline.** wlan0 and eth0 down on sender; receiver loses DHCP.
10. **Reconnect window.** Peer silent 0–10 s before giving up.
11. **DRM refused.** FairPlay content rejected mid-session.
12. **Reduced motion.** `prefers-reduced-motion = reduce` (OS setting).

## Findings (sender tray)

| # | Surface | Expected failure | Result | Fix |
| --- | --- | --- | --- | --- |
| 1 | Tray popup empty | Blank, looks "broken" | Branded empty with mDNS/Bonjour/SSDP pills + searching progress bar | `10.A` |
| 3 | Tray popup list | Rows crowd out footer when > 7 | Collapse "N more…" row at row 7; scrollable inside popup | `05.C` |
| 4 | Row name truncation | Long names overflow 328 px row | Row width fixed, single-line truncation; mono spec stays visible as `meta` | `10.C` |
| 5 | Emoji + CJK | Font fallback breaks baseline | Inter 600 + system CJK fallback; row height fixed 44 px for CJK ascenders | `10.C` |
| 6 | Single-character name | Name baseline looks orphaned | Icon chip (28 × 28) keeps left anchor stable; row stays 44 px | `10.C` |
| 7 | Negotiating stalls | No progress sign → feels dead | Accent progress bar (determinate if we have CT, indeterminate if not) + step line | `10.D` |
| 8 | Codec failure | Opaque error toast | Banner with `Why?` disclosure + install-guide link; row flips to error border | `10.E` |
| 9 | Sender offline | Device list stays stale | Warning banner; tray icon turns neutral grey, not ember; discovery paused notice | `10.F` |
| 11| DRM refused | Modal interrupts receiver | On sender shown as a row-level error line, not a popup (Postel); session just ends | `10.H` |

## Findings (receiver)

| # | Surface | Expected failure | Result | Fix |
| --- | --- | --- | --- | --- |
| 1 | Idle hero on OLED TV | Burn-in from static UI | AMOLED-override theme (`receiver.amoled`) makes canvas pure black + accent ember.300, breathing dot pulses so pixels shift | `08.C` |
| 2 | Incoming confirm | User away, auto-accepts → privacy risk | 15 s visible countdown, auto-**reject** (not accept), fingerprint line shows if we've seen this sender before | `08.D` |
| 10| Peer silent 0–10 s | Fall to black → feels crashed | Reconnecting overlay with countdown; fades to idle hero on timeout, not to black | `08.F` |
| 12| Reduced motion | Breathing dot violates OS pref | Entire motion token layer disabled by system pref; accent becomes a static disc (no scale animation); receiver hero is a static composition | `10.I` |

## Still unknown (needs real hardware)

- AirPlay FairPlay handshake with modern tvOS in the wild (the pile of DRM error codes is impossible to emulate without a receiver running a real session).
- Miracast under Windows 11 24H2 sender quirks (connection drops when switching WiFi radios mid-session).
- Chromecast HLS with audio-only receivers (chip extraction logic unverified against newest Google audio Chromecasts).
- Tray template icons on KDE Plasma 6 Wayland — may not pick up `-symbolic` suffix for recolouring.

Mark these as Risks in the consolidated report, not Blocking — they're external surface we discover at integration time.

## Keep

- Empty state copy: declarative, no apology, names the discovery protocols. "This is normal — some devices take 5–30 s." removes a common support ticket.
- Codec fallback is a toast + chip, not a modal. Keeps the stream alive and documents the compromise.
- `press  g  for logs` hint on receiver is tiny but consistent; power users ask for it once and never forget.
