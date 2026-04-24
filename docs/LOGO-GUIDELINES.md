# Glassbox AI — Logo Guidelines

**Brand System v2.0 · April 2026**

---

## The Mark

The Glassbox AI mark is an **isometric glass cube** with a three-layer neural network drawn on its right face. The hexagonal outline represents the cube in isometric projection (pointy-top orientation). Three face planes are visible — top, right, left — each at precisely calibrated opacity to create depth without obscuring the internal structure. The neural network on the right face shows the 2→3→2 layer architecture used in mechanistic interpretability research: every node and connection maps to the actual science.

The divider lines from center to three hex vertices divide the cube into its faces. The top face carries parallel circuit traces and via pads. The mark is geometrically exact at every size.

---

## Files

| File | Use |
|------|-----|
| `docs/logo-mark.svg` | Icon only — favicon, app icons, small placements |
| `docs/logo.svg` | Full horizontal lockup — nav bars, headers |
| `docs/glassbox-brand.png` | Brand identity sheet — README, presentations, OG image |

---

## Color Palette

| Role | Hex | CSS Variable | Usage |
|------|-----|-------------|-------|
| **Primary Cyan** | `#00C8E8` | `--indigo` | All mark elements, primary actions, CTA buttons |
| **Deep Cyan** | `#009AB5` | `--indigo-d` | Hover states, pressed states |
| **Light Cyan** | `#38D8F0` | `--indigo-l` | Glow effects, gradient highlights |
| **Sky** | `#38BDF8` | `--sky` | Secondary accents |
| **Cream** | `#EBE7DE` | `--t1` | Primary text on dark backgrounds |
| **Background** | `#050709` | `--bg` | Dark base — all surfaces |

All orange values (`#E8724A`, `#F0946E`, `#C45A30`) are retired. Do not use them.

---

## Mark Geometry (100×100 viewBox)

| Element | Coordinates |
|---------|-------------|
| Hex vertices (pointy-top) | V0(50,19.69) V1(85,34.85) V2(85,65.15) V3(50,80.31) V4(15,65.15) V5(15,34.85) |
| Center | C(50,50) |
| Dividers | C→V0, C→V2, C→V4 |
| NN Layer 0 (input) | n0(79.75,45.60) n1(79.75,58.94) |
| NN Layer 1 (hidden) | n2(67.5,47.88) n3(67.5,57.58) n4(67.5,67.27) |
| NN Layer 2 (output) | n5(56.3,55.75) n6(56.3,69.10) |

Face fill opacities: top=0.055, right=0.105, left=0.030.

---

## Typography

| Role | Font | Weight | Usage |
|------|------|--------|-------|
| **Display / Wordmark** | Big Shoulders Display / Syne | 700 | Hero titles, wordmark |
| **Body / UI** | DM Sans | 300–500 | Body text, labels, nav |
| **Code / Mono** | JetBrains Mono | 400–500 | Code samples, metrics |
| **Technical Labels** | Jura | 300 | Fine print, technical tags, "AI" sub-tag |

Wordmark letter-spacing: `−0.03em`. The "AI" tag uses `+0.45em` tracking at 9px.

---

## Logo Usage Rules

### Do
- Use the mark on dark backgrounds (`#050709` or equivalent near-black)
- Maintain minimum clear space of 1× mark height on all sides
- Use the full lockup (mark + wordmark) in nav bars and marketing
- Use the mark alone at 24px–64px for icons and favicons
- The hex outline works best at 24px and above — at smaller sizes, drop the NN detail

### Do not
- Recolor the mark — cyan is identity-critical
- Place on white/light backgrounds without a dark container
- Stretch or distort the hexagonal aspect ratio
- Add drop shadows or overlays that obscure the circuit traces
- Use a different font for the wordmark
- Use the retired orange palette (`#E8724A` or variants)

---

## Minimum Sizes

| Context | Minimum size |
|---------|-------------|
| Mark only (digital) | 24×24 px |
| Mark only (print) | 8mm × 8mm |
| Full lockup (digital) | 160px wide |
| Full lockup (print) | 50mm wide |

---

## The Concept in One Line

> **An isometric glass cube showing the exact neural circuit of the model — transparent, precise, and three-dimensional. The logo is the product.**
