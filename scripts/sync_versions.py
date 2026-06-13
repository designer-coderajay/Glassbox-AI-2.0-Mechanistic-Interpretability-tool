#!/usr/bin/env python3
"""
sync_versions.py — Single source of truth for the package version.

Reads the canonical version from pyproject.toml and patches every file
that contains a stale version reference.  Run automatically in CI before
every Vercel deploy and HF Space deploy so nothing ever drifts.

Usage:
    python3 scripts/sync_versions.py          # dry-run (prints changes)
    python3 scripts/sync_versions.py --apply  # write changes to disk
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def get_canonical_version() -> str:
    """Read version from pyproject.toml without requiring tomllib (3.11+)."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find version in pyproject.toml")
    return m.group(1)


def patch_file(path: Path, patterns: list[tuple], version: str, apply: bool) -> bool:
    """Apply regex substitutions to a file.  Returns True if any change was made."""
    text = path.read_text(encoding="utf-8")
    original = text
    for pattern, replacement_template in patterns:
        repl = replacement_template.format(version=version)
        # MULTILINE so ^-anchored patterns (e.g. mcp/pyproject.toml's
        # `^version = "..."`) match mid-file, not only at byte 0.
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)
    if text == original:
        return False
    if apply:
        path.write_text(text, encoding="utf-8")
        print(f"  [patched]  {path.relative_to(ROOT)}")
    else:
        print(f"  [would patch]  {path.relative_to(ROOT)}")
    return True


def main() -> None:
    apply = "--apply" in sys.argv
    version = get_canonical_version()
    print(f"Canonical version: {version}  ({'applying' if apply else 'dry-run'})")

    changed = 0

    # ── glassbox/__init__.py (the package's own __version__) ──────────────────
    # Critical: this is what `glassbox.__version__`, the CLI --version, the MCP
    # server info, and Annex IV report headers all read. Must stay in sync with
    # the canonical pyproject version or the installed package misreports itself.
    changed += patch_file(
        ROOT / "glassbox" / "__init__.py",
        [
            (r'__version__\s*=\s*"\d+\.\d+\.\d+"',
             '__version__ = "{version}"'),
        ],
        version, apply,
    )

    # ── dashboard/app.py ──────────────────────────────────────────────────────
    changed += patch_file(
        ROOT / "dashboard" / "app.py",
        [
            # Docstring header
            (r"Glassbox \d+\.\d+\.\d+ — Causal Mechanistic",
             "Glassbox {version} — Causal Mechanistic"),
            (r"HuggingFace Space — v\d+\.\d+\.\d+",
             "HuggingFace Space — v{version}"),
            # Inline version references
            (r"Glassbox v\d+\.\d+\.\d+ ·",
             "Glassbox v{version} ·"),
            (r"Built on TransformerLens &nbsp;&middot;&nbsp; v\d+\.\d+\.\d+",
             "Built on TransformerLens &nbsp;&middot;&nbsp; v{version}"),
            (r"version = \{\d+\.\d+\.\d+\}",
             "version = {{{version}}}"),
        ],
        version, apply,
    )

    # ── README.md ─────────────────────────────────────────────────────────────
    changed += patch_file(
        ROOT / "README.md",
        [
            (r"# Glassbox \d+\.\d+\.\d+", "# Glassbox {version}"),
            (r"\|\s*Version\s*\|\s*`v\d+\.\d+\.\d+`\s*\|",
             "| Version | `v{version}` |"),
        ],
        version, apply,
    )

    # ── .github/workflows/deploy_hf.yml ──────────────────────────────────────
    changed += patch_file(
        ROOT / ".github" / "workflows" / "deploy_hf.yml",
        [
            (r"glassbox-mech-interp>=\d+\.\d+\.\d+",
             "glassbox-mech-interp>={version}"),
            (r"\*\*Version\*\*: \d+\.\d+\.\d+ \(GitHub",
             "**Version**: {version} (GitHub"),
        ],
        version, apply,
    )

    # ── docs/index.html ──────────────────────────────────────────────────────
    changed += patch_file(
        ROOT / "docs" / "index.html",
        [
            (r"glassbox-mech-interp==\d+\.\d+\.\d+",
             "glassbox-mech-interp=={version}"),
            # Nav + footer version badges and JSON-LD softwareVersion
            (r'<span class="v">v\d+\.\d+\.\d+</span>',
             '<span class="v">v{version}</span>'),
            (r'"softwareVersion":"\d+\.\d+\.\d+"',
             '"softwareVersion":"{version}"'),
        ],
        version, apply,
    )

    # ── mcp/server.py ────────────────────────────────────────────────────────
    changed += patch_file(
        ROOT / "mcp" / "server.py",
        [
            (r"glassbox-mech-interp v\d+\.\d+\.\d+",
             "glassbox-mech-interp v{version}"),
        ],
        version, apply,
    )

    # ── mcp/glassbox_mcp/server.py ───────────────────────────────────────────
    changed += patch_file(
        ROOT / "mcp" / "glassbox_mcp" / "server.py",
        [
            (r"glassbox-mech-interp v\d+\.\d+\.\d+",
             "glassbox-mech-interp v{version}"),
        ],
        version, apply,
    )

    # ── mcp/requirements.txt ─────────────────────────────────────────────────
    changed += patch_file(
        ROOT / "mcp" / "requirements.txt",
        [
            (r"glassbox-mech-interp>=\d+\.\d+\.\d+",
             "glassbox-mech-interp>={version}"),
        ],
        version, apply,
    )

    # ── mcp/pyproject.toml ───────────────────────────────────────────────────
    changed += patch_file(
        ROOT / "mcp" / "pyproject.toml",
        [
            (r'^version\s*=\s*"\d+\.\d+\.\d+"',
             'version = "{version}"'),
            (r"glassbox-mech-interp>=\d+\.\d+\.\d+",
             "glassbox-mech-interp>={version}"),
        ],
        version, apply,
    )

    # ── mcp/glassbox_mcp/__init__.py ─────────────────────────────────────────
    changed += patch_file(
        ROOT / "mcp" / "glassbox_mcp" / "__init__.py",
        [
            (r'__version__\s*=\s*"\d+\.\d+\.\d+"',
             '__version__ = "{version}"'),
        ],
        version, apply,
    )

    # ── mcp/README.md ─────────────────────────────────────────────────────────
    changed += patch_file(
        ROOT / "mcp" / "README.md",
        [
            (r"glassbox-mcp==\d+\.\d+\.\d+",
             "glassbox-mcp=={version}"),
            (r"pip install glassbox-mcp==\d+\.\d+\.\d+",
             "pip install glassbox-mcp=={version}"),
            (r"\*\*Current version:\*\* \d+\.\d+\.\d+",
             "**Current version:** {version}"),
            (r"Current version: \d+\.\d+\.\d+",
             "Current version: {version}"),
            (r"glassbox-mcp v\d+\.\d+\.\d+",
             "glassbox-mcp v{version}"),
        ],
        version, apply,
    )

    # ── ENTERPRISE.md ─────────────────────────────────────────────────────────
    changed += patch_file(
        ROOT / "ENTERPRISE.md",
        [
            (r"Glassbox AI v\d+\.\d+\.\d+",
             "Glassbox AI v{version}"),
        ],
        version, apply,
    )

    print(f"\n{'Applied' if apply else 'Would apply'} changes to {changed} file(s).")
    if not apply and changed:
        print("Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
