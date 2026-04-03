# Glassbox Website Design Snapshot

> **FROZEN DESIGN — DO NOT OVERWRITE `docs/index.html` with a redesign.**

`index.html` in this folder is the canonical reference snapshot of the approved
Glassbox v3.6.0 website design, locked on **2026-04-03**.

---

## What is locked

| Layer | Description |
|-------|-------------|
| **Layout** | Split hero (headline + CTAs left, live-typing terminal right), scroll-progress bar, stats ticker, compliance panel, animated circuit analyzer |
| **Color palette** | Warm amber `#e8a000` / anthracite `#0e0d0b` / cream `#f0ebe0` (Anthropic-inspired warm dark theme) |
| **Typography** | DM Serif Display headings, Inter body, JetBrains Mono terminal |
| **Animations** | Spring physics `cubic-bezier(0.16,1,0.3,1)`, magnetic CTAs, scroll-triggered counters, terminal self-typing |
| **Sections** | Hero → Stats bar → r=0.009 finding → How it works → Live analyzer → Compliance → CTA → Footer |

---

## What IS allowed to change inside `docs/index.html`

- **Research numbers**: r, sufficiency, comprehensiveness, F1 values (if the paper updates)
- **Version badge**: the PyPI version string
- **Canonical/og URLs**: if the Vercel project URL changes
- **API endpoint**: the HF Space URL in the circuit analyzer fetch call
- **Text copy**: feature descriptions, footer links

## What is NOT allowed to change

- Color variables (`--amber`, `--bg`, `--fg`, `--muted`)
- Font families
- Section order or section existence
- Animation keyframes and timing curves
- Overall layout structure (split hero, terminal, ticker etc.)

---

## How to restore

If `docs/index.html` is accidentally overwritten with a different design:

```bash
cp docs/_design-snapshot/index.html docs/index.html
git add docs/index.html
git commit -m "revert: restore approved Glassbox website design from snapshot"
```

---

## Diff check

To verify `docs/index.html` hasn't drifted in structure from this snapshot:

```bash
diff <(grep -E "^(\.|\#|@keyframes|:root|section|<div|<header|<footer)" docs/index.html) \
     <(grep -E "^(\.|\#|@keyframes|:root|section|<div|<header|<footer)" docs/_design-snapshot/index.html)
```
