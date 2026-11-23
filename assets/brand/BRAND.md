# ferricast · brand kit

Ready-to-use brand assets. Drop them straight into packaging, read this file when you're deciding where something goes.

## Identity in one paragraph

**ferricast** is a cross-platform, cross-protocol cast sender and receiver (AirPlay, Miracast, Chromecast; more to come). The mark is a monogram **F** cast in a heavy geometric slab: two 45° wedges cut from its top and middle arms "emit" the signal outward. The name is always spelled **ferricast** in lowercase. The voice is direct, technical, humble; writing avoids exclamation marks and cloud metaphors.

## What's in this folder

```
assets/brand/
├── symbol/                     master vector
│   ├── ferricast-symbol.svg          · 256×256, use ≥ 24 px
│   └── ferricast-symbol-small.svg    · optimised for 16–24 px (one wedge, flat mid-arm)
├── lockup/                     symbol + wordmark
│   ├── ferricast-horizontal.svg       · default lockup
│   ├── ferricast-stacked.svg          · square containers, cards, product shots
│   └── ferricast-wordmark.svg         · text only
├── color/                      single-use colour variants of the symbol
│   ├── ferricast-symbol-black.svg
│   ├── ferricast-symbol-white.svg
│   ├── ferricast-symbol-ember.svg
│   └── ferricast-symbol-on-dark.svg   · rounded tile for dock/launcher reversed look
├── icon/
│   ├── favicon/                html/web icons + site.webmanifest + <head> snippet
│   ├── linux/hicolor/          Freedesktop icon theme tree (16…512 px + scalable + symbolic)
│   ├── windows/ferricast.ico   multi-size .ico (16, 24, 32, 48, 64, 128, 256)
│   ├── macos/ferricast.iconset mac iconset (needs `iconutil` to pack .icns, see below)
│   └── tray/                   tray glyphs · template (Linux/macOS) + live + attention
├── png/                        rendered PNGs of symbol and lockups for docs/share
├── palette.json                machine-readable palette + per-theme semantic
└── ferricast.desktop           Freedesktop launcher template
```

All SVGs ship with **outlined** paths. No live `<text>`, no embedded fonts, no filters, no gradients.

## Install into an app bundle

### Linux (Freedesktop)

Install the hicolor tree alongside the binary:

```
install -Dm644 assets/brand/icon/linux/hicolor/scalable/apps/ferricast.svg \
  $out/share/icons/hicolor/scalable/apps/ferricast.svg
for S in 16 22 24 32 48 64 96 128 192 256 512; do
  install -Dm644 assets/brand/icon/linux/hicolor/${S}x${S}/apps/ferricast.png \
    $out/share/icons/hicolor/${S}x${S}/apps/ferricast.png
done
install -Dm644 assets/brand/icon/linux/hicolor/symbolic/apps/ferricast-symbolic.svg \
  $out/share/icons/hicolor/symbolic/apps/ferricast-symbolic.svg
install -Dm644 assets/brand/ferricast.desktop $out/share/applications/ferricast.desktop
```

Tray (Linux): use `ferricast-symbolic.svg` so the system theme can tint it.

### Windows

- Binary resource: embed `assets/brand/icon/windows/ferricast.ico` as the `.rc` `IDI_ICON1`.
- MSIX / installer: use the `icon/favicon/icon-512.png` and `apple-touch-icon.png` for Start Menu tiles if you need PNG assets.
- System tray: Windows 11 tints template icons if you pass them as monochrome; use `icon/tray/ferricast-tray-mono.svg` converted to a 32 px ICO for the notify icon, and the `-live.svg` / `-attention.svg` variants for state changes.

### macOS

The `.iconset` folder has every size Apple expects. Pack on a Mac:

```
iconutil -c icns assets/brand/icon/macos/ferricast.iconset \
  -o assets/brand/icon/macos/ferricast.icns
```

Drop `ferricast.icns` into your `.app/Contents/Resources/`.
Menubar status item: ship the `icon/tray/ferricast-tray-mono.svg` as a Template image (filename must end in `Template` or set `.isTemplate = true` in Swift/Objective-C bridging; the SVG uses `currentColor` so macOS tints light/dark automatically).

### Web / docs / README

```
<link rel="icon" type="image/svg+xml" href="/assets/brand/icon/favicon/favicon.svg">
<link rel="icon" type="image/png" sizes="32x32" href="/assets/brand/icon/favicon/favicon-32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/assets/brand/icon/favicon/favicon-16.png">
<link rel="apple-touch-icon" href="/assets/brand/icon/favicon/apple-touch-icon.png">
<link rel="manifest" href="/assets/brand/icon/favicon/site.webmanifest">
```

## Usage rules

- **Clear space:** one stem height around the mark in any lockup. That's the height of the F's crossbar (30 units in the viewBox).
- **Minimum size:** symbol 16 px (use `ferricast-symbol-small.svg` below 32 px); horizontal lockup 96 px wide; stacked 64 px wide.
- **Monocromo:** always one colour against a solid surface. Do not place on gradients, photos, or vivid backgrounds.
- **Do not:** stroke the mark, rotate it, skew it, change the wedge angles, enclose it in a frame that competes with its own counters, mix it with another brand's cast icon (AirPlay, Chromecast, Miracast) in the same locked-up unit, or recolour the wedges.
- **Spoken name:** "ferricast", always lowercase, one word. In running text keep it lowercase even at the start of a sentence if your typography supports it; otherwise "Ferricast" is fine.

## Palette (summary)

| Role | Hex | Pairs well with |
| --- | --- | --- |
| Primary (brand / cast live) | `#C85A2D` (ember.500) | `#F6F1EC` light, `#0E1218` dark |
| Ink | `#1C1814` (iron.900) | always |
| Canvas light | `#F6F1EC` (iron.50) | ink, ember |
| Canvas dark | `#0E1218` (steel.900) | iron.50 as fg, ember |
| Receiver AMOLED | `#000000` | ember.300 fg, iron.50 fg |

Full palette lives in `palette.json` and the Penpot local library (sets: `primitive`, `semantic.light`, `semantic.dark`, `semantic.hcLight`, `semantic.hcDark`, `receiver.amoled`).

## Typography (lockups + UI)

- Sans: **Inter** · 400 / 500 / 700. Fallback: system sans. SIL OFL, redistribute freely with the binary.
- Mono: **JetBrains Mono** · 400 / 500. Fallback: system mono. Apache-2.0.

The provided lockups are already outlined. If you need to regenerate a lockup, re-outline with Inkscape 1.3+ or `fontTools.subset` after setting type.

## Audit scores

Every SVG in `symbol/`, `color/`, and the small-size cut passes the logo-design audit at 99–100/100 (anchors, angles, margins). Lockups score 89–98 after outlining (added anchors from letterforms, intentional overflow-safe margins).

## Changing the mark later

The master lives in `symbol/ferricast-symbol.svg`. If you edit it, re-run:

```
python3 scripts/logo/export_variants.py symbol/ferricast-symbol.svg --web-icons --favicon-source symbol/ferricast-symbol-small.svg --title ferricast --mono '#C85A2D' --icon-bg '#F6F1EC'
```

...plus the Linux hicolor regeneration loop in `BRAND.md` so every size stays pixel-synced.
