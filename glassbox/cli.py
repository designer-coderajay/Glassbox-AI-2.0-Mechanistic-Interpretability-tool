"""
glassbox.cli — Command-line interface for Glassbox.

Sub-commands
------------
analyze  Run circuit discovery + faithfulness on a single prompt.
doctor   Check that all dependencies are correctly installed.
version  Print the installed Glassbox version.

Usage
-----
    glassbox-ai analyze \\
        --prompt "When Mary and John went to the store, John gave a drink to" \\
        --correct " Mary" --incorrect " John"

    glassbox-ai doctor
    glassbox-ai version
"""
from __future__ import annotations

import argparse
import sys

BANNER = """
  +==============================================+
  |   G L A S S B O X  2 . 6                   |
  |  Mechanistic Interpretability Toolkit       |
  +==============================================+
"""

# ANSI colour codes — disabled automatically when not on a TTY
_GREEN  = "\033[92m" if sys.stdout.isatty() else ""
_YELLOW = "\033[93m" if sys.stdout.isatty() else ""
_RED    = "\033[91m" if sys.stdout.isatty() else ""
_RESET  = "\033[0m"  if sys.stdout.isatty() else ""

_OK   = f"{_GREEN}✓{_RESET}"
_WARN = f"{_YELLOW}~{_RESET}"
_FAIL = f"{_RED}✗{_RESET}"


# ---------------------------------------------------------------------------
# Sub-command: analyze
# ---------------------------------------------------------------------------

def _run_analyze(args: argparse.Namespace) -> int:
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2

    print(BANNER)
    print(f"  Model  : {args.model}")
    print(f"  Prompt : {args.prompt!r}")
    print(f"  Correct: {args.correct!r}   Incorrect: {args.incorrect!r}\n")

    model = HookedTransformer.from_pretrained(args.model)
    gb    = GlassboxV2(model)

    result = gb.analyze(args.prompt, args.correct, args.incorrect)
    faith  = result["faithfulness"]
    meta   = result.get("model_metadata", {})

    print(f"  Sufficiency      : {faith['sufficiency']:.1%}")
    print(f"  Comprehensiveness: {faith['comprehensiveness']:.1%}")
    print(f"  F1-score         : {faith['f1']:.1%}")
    print(f"  Category         : {faith['category']}")
    if faith.get("suff_is_approx"):
        print("  Note             : Sufficiency is a 1st-order Taylor approximation.")
    print()

    print(f"  {'Head':<12} {'Attribution':>12}")
    print(f"  {'-'*12} {'-'*12}")
    attrs = result["attributions"]
    for (layer, head) in result["circuit"]:
        score = attrs.get(str((layer, head)), 0.0)
        print(f"  L{layer:02d}H{head:02d}      {score:>12.4f}")

    if meta:
        print(f"\n  Model: {meta.get('model_name','?')}  "
              f"{meta.get('n_layers','?')}L × {meta.get('n_heads','?')}H  "
              f"d_model={meta.get('d_model','?')}  "
              f"glassbox=v{meta.get('glassbox_version','?')}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: doctor
# ---------------------------------------------------------------------------

def _run_doctor(_args: argparse.Namespace) -> int:
    """Print a dependency health report."""
    print(BANNER)
    print("  Dependency diagnostics\n")
    print(f"  {'Package':<32} {'Version / Status':<28} {'Required'}")
    print(f"  {'-'*32} {'-'*28} {'-'*8}")

    checks = []

    # Python
    pv = sys.version_info
    py_str = f"{pv.major}.{pv.minor}.{pv.micro}"
    py_ok  = pv >= (3, 8)
    checks.append(("Python", py_str, py_ok, True))

    # PyTorch
    try:
        import torch
        cuda = "  [CUDA]" if torch.cuda.is_available() else "  [CPU only]"
        checks.append(("torch", torch.__version__ + cuda, True, True))
    except ImportError:
        checks.append(("torch", "NOT INSTALLED", False, True))

    # TransformerLens
    try:
        import transformer_lens  # noqa: F401
        # __version__ may be absent in some editable installs; fall back to importlib
        tl_ver = getattr(transformer_lens, "__version__", None)
        if tl_ver is None:
            try:
                from importlib.metadata import version as _iv
                tl_ver = _iv("transformer_lens")
            except Exception:
                tl_ver = "installed (version unknown)"
        checks.append(("transformer_lens", tl_ver, True, True))
    except ImportError:
        checks.append(("transformer_lens", "NOT INSTALLED", False, True))

    # einops
    try:
        import einops
        checks.append(("einops", einops.__version__, True, True))
    except ImportError:
        checks.append(("einops", "NOT INSTALLED", False, True))

    # numpy
    try:
        import numpy
        checks.append(("numpy", numpy.__version__, True, True))
    except ImportError:
        checks.append(("numpy", "NOT INSTALLED", False, True))

    # Glassbox itself
    try:
        import glassbox
        checks.append(("glassbox-mech-interp", glassbox.__version__, True, True))
    except ImportError:
        checks.append(("glassbox-mech-interp", "NOT INSTALLED", False, True))

    # sae-lens (optional)
    try:
        import sae_lens
        checks.append(("sae-lens", sae_lens.__version__, True, False))
    except ImportError:
        checks.append(("sae-lens", "not installed (optional SAE features)", None, False))

    # streamlit (optional, for dashboard)
    try:
        import streamlit
        checks.append(("streamlit", streamlit.__version__, True, False))
    except ImportError:
        checks.append(("streamlit", "not installed (optional dashboard)", None, False))

    all_required_ok = True
    for name, status, ok, required in checks:
        if ok is True:
            icon = _OK
        elif ok is False:
            icon = _FAIL
            if required:
                all_required_ok = False
        else:
            icon = _WARN   # optional / not installed
        req_flag = "required" if required else "optional"
        print(f"  {icon} {name:<32} {status:<28} {req_flag}")

    print()
    if all_required_ok:
        print(f"  {_GREEN}All required dependencies OK. Glassbox is ready to use.{_RESET}\n")
        return 0
    else:
        print(f"  {_RED}Some required dependencies are missing.{_RESET}")
        print(f"  Run:  pip install glassbox-mech-interp\n")
        return 1


# ---------------------------------------------------------------------------
# Sub-command: version
# ---------------------------------------------------------------------------

def _run_version(_args: argparse.Namespace) -> int:
    import glassbox
    print(f"glassbox-mech-interp {glassbox.__version__}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="glassbox-ai",
        description="Glassbox 2.6 — Mechanistic Interpretability Toolkit",
        epilog="""Examples:
  glassbox-ai analyze \\
      --prompt "When Mary and John went to the store, John gave a drink to" \\
      --correct " Mary" --incorrect " John"

  glassbox-ai doctor
  glassbox-ai version
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="cmd")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a prompt circuit")
    p_analyze.add_argument("--prompt",    required=True, help="Input prompt")
    p_analyze.add_argument("--correct",   required=True, help="Correct completion token")
    p_analyze.add_argument("--incorrect", required=True, help="Incorrect/distractor token")
    p_analyze.add_argument("--model",     default="gpt2",
                           help="HuggingFace model name (default: gpt2)")

    # doctor
    sub.add_parser("doctor",  help="Check all dependencies are correctly installed")

    # version
    sub.add_parser("version", help="Print installed Glassbox version")

    args = parser.parse_args()

    if args.cmd == "analyze":
        sys.exit(_run_analyze(args))
    elif args.cmd == "doctor":
        sys.exit(_run_doctor(args))
    elif args.cmd == "version":
        sys.exit(_run_version(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
