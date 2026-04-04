#!/usr/bin/env python3
"""
Build-time script: fetch PyPI download stats and patch docs/index.html.

Runs during `vercel build` (server-side) so there are no browser CORS
restrictions. Vercel then serves the already-patched static HTML.
"""
import json
import re
import urllib.request
from pathlib import Path

HTML = Path(__file__).parent.parent / "docs" / "index.html"
PACKAGE = "glassbox-mech-interp"
URL = f"https://pypistats.org/api/packages/{PACKAGE}/recent"


def fmt(n: int) -> str:
    """Format an integer as a human-readable string with K suffix."""
    if n >= 1_000_000:
        v = n / 1_000_000
        s = f"{v:.1f}".rstrip("0").rstrip(".")
        return f"{s}M"
    if n >= 1_000:
        v = n / 1_000
        s = f"{v:.1f}".rstrip("0").rstrip(".")
        return f"{s}K"
    return str(n)


def main() -> None:
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "glassbox-build/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())["data"]
        weekly  = int(data.get("last_week",  0))
        monthly = int(data.get("last_month", 0))
        # Prefer monthly — it's more stable and representative than a single week
        # which can dip during quiet periods.  Fall back to weekly if monthly is 0.
        count = monthly or weekly
        scope = "Downloads/mo" if monthly else "Downloads/wk"
    except Exception as exc:
        print(f"[patch_download_count] fetch failed ({exc}); skipping patch")
        return

    if count <= 0:
        print("[patch_download_count] count is 0; skipping patch")
        return

    raw = fmt(count)
    # Build the innerHTML: e.g. "1.5<sup>K</sup>" or "657"
    if raw.endswith("K") or raw.endswith("M"):
        suffix = raw[-1]
        num    = raw[:-1]
        inner  = f'{num}<sup>{suffix}</sup>'
    else:
        inner = raw

    html = HTML.read_text(encoding="utf-8")

    # Patch the dl-count span
    html = re.sub(
        r'(<div class="sn" id="dl-count">)[^<]*(?:<sup>[^<]*</sup>)?',
        rf'\g<1>{inner}',
        html,
    )
    # Patch the sibling label (Downloads/wk or Downloads/mo)
    html = re.sub(
        r'(id="dl-count">[^<]*(?:<sup>[^<]*</sup>)?</div><div class="sl">)[^<]*',
        rf'\g<1>{scope}',
        html,
    )

    HTML.write_text(html, encoding="utf-8")
    print(f"[patch_download_count] patched to {inner} ({scope}); raw count={count}")


if __name__ == "__main__":
    main()
