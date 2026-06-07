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
  |   G L A S S B O X  4 . 2 . 6               |
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
    print(f"  Model      : {args.model}")
    print(f"  Prompt     : {args.prompt!r}")
    print(f"  Correct    : {args.correct!r}   Incorrect: {args.incorrect!r}")
    print(f"  Strategy   : {args.strategy or 'auto (will be selected based on prompt)'}\n")

    # Dtype handling for large models
    dtype = None
    if args.dtype:
        import torch
        dtype_map = {"float32": torch.float32, "float16": torch.float16,
                     "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(args.dtype)
        if dtype is None:
            print(f"  ERROR: unknown dtype {args.dtype!r}. Use float32/float16/bfloat16")
            return 1

    load_kwargs: dict = {}
    if dtype is not None:
        load_kwargs["dtype"] = dtype

    model = HookedTransformer.from_pretrained(args.model, **load_kwargs)
    gb    = GlassboxV2(model)

    # Memory estimate for large models
    d = getattr(model.cfg, "d_model", 768)
    n_layers = getattr(model.cfg, "n_layers", 12)
    from glassbox.large_model import estimate_memory, classify_model_size
    mem = estimate_memory(n_layers, d, dtype=dtype)
    size_class = classify_model_size(n_layers, d)
    if size_class in ("large", "xlarge", "xxlarge"):
        print(f"  Memory estimate : {mem.attribution_gb:.1f} GB for attribution pass")
        print(f"  Recommendation  : {mem.recommend_strategy}")
        for w in mem.warnings:
            print(f"  ⚠  {w}")
        print()

    # Use large-model adapter if strategy specified or model is large
    strategy = args.strategy or "auto"
    if size_class in ("large", "xlarge", "xxlarge") or strategy != "auto":
        from glassbox.large_model import analyze_large
        result = analyze_large(gb, args.prompt, args.correct, args.incorrect,
                               strategy=strategy, dtype=dtype)
    else:
        result = gb.analyze(args.prompt, args.correct, args.incorrect,
                            corruption_strategy=strategy)

    faith  = result["faithfulness"]
    meta   = result.get("model_metadata", {})
    corruption = result.get("corruption_metadata", {})

    if corruption:
        print(f"  Corruption strategy : {corruption.get('strategy', 'unknown')}")
        print(f"  Rationale           : {corruption.get('rationale', '')[:80]}")
        print()

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
# Sub-command: estimate-memory  (large-model planning)
# ---------------------------------------------------------------------------

def _run_estimate_memory(args: argparse.Namespace) -> int:
    """Predict VRAM requirements before loading any model."""
    import torch
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.float32)

    from glassbox.large_model import estimate_memory, classify_model_size

    mem = estimate_memory(
        n_layers=args.n_layers,
        d_model=args.d_model,
        seq_len=args.seq_len,
        dtype=dtype,
    )
    size_class = classify_model_size(args.n_layers, args.d_model)

    print(BANNER)
    print(f"  n_layers : {args.n_layers}")
    print(f"  d_model  : {args.d_model}")
    print(f"  seq_len  : {args.seq_len}")
    print(f"  dtype    : {args.dtype}")
    print()
    print(f"  Estimated parameters     : {mem.n_params / 1e9:.2f}B")
    print(f"  Model weights (VRAM)     : {mem.param_gb:.1f} GB")
    print(f"  Activations / fwd pass   : {mem.activation_gb:.1f} GB")
    print(f"  Attribution (3 passes)   : {mem.attribution_gb:.1f} GB  ← peak requirement")
    print(f"  Size class               : {size_class}")
    print(f"  Recommended strategy     : {mem.recommend_strategy}")
    if mem.warnings:
        print()
        for w in mem.warnings:
            print(f"  ⚠  {w}")
    print()
    print("  Usage:")
    print(f"    from glassbox.large_model import analyze_large")
    print(f"    result = analyze_large(gb, prompt, correct, incorrect,")
    print(f"                          strategy='{mem.recommend_strategy}',")
    print(f"                          dtype=torch.{args.dtype})")
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
        description="Glassbox 4.3.0 — Mechanistic Interpretability + EU AI Act Compliance",
        epilog="""Examples:
  # Any prompt, any domain — auto-selects corruption strategy
  glassbox-ai analyze \\
      --prompt "Loan application. Annual income: €42,000. Decision:" \\
      --correct " Approved" --incorrect " Denied"

  # Large model with memory-efficient attribution
  glassbox-ai analyze \\
      --prompt "Patient presents with chest pain. Priority:" \\
      --correct " Urgent" --incorrect " Routine" \\
      --model meta-llama/Llama-3-8B --dtype bfloat16 --strategy auto

  # Check VRAM before loading a large model
  glassbox-ai estimate-memory --n-layers 80 --d-model 8192 --dtype bfloat16

  glassbox-ai doctor
  glassbox-ai version
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="cmd")

    # ── analyze ──────────────────────────────────────────────────────────────
    p_analyze = sub.add_parser("analyze", help="Run circuit discovery + faithfulness on any prompt")
    p_analyze.add_argument("--prompt",    required=True,
                           help="Input text (any domain — credit, medical, HR, legal, etc.)")
    p_analyze.add_argument("--correct",   required=True,
                           help="Correct next token (e.g. ' Approved', ' Urgent')")
    p_analyze.add_argument("--incorrect", required=True,
                           help="Distractor token (e.g. ' Denied', ' Routine')")
    p_analyze.add_argument("--model",     default="gpt2",
                           help="HuggingFace model name (default: gpt2). "
                                "Supports all 11 architecture families.")
    p_analyze.add_argument("--dtype",     default=None,
                           choices=["float32", "float16", "bfloat16"],
                           help="Model dtype. Use bfloat16 for 7B+ models to halve VRAM.")
    p_analyze.add_argument("--strategy",  default=None,
                           choices=["auto", "name_swap", "random_token",
                                    "antonym", "semantic_negation"],
                           help="Corruption strategy. Default: auto (recommended). "
                                "auto selects the best strategy for your prompt type.")

    # ── estimate-memory ───────────────────────────────────────────────────────
    p_mem = sub.add_parser("estimate-memory",
                           help="Predict VRAM requirements before loading a large model")
    p_mem.add_argument("--n-layers", type=int, required=True,
                       help="Number of transformer layers (e.g. 32 for Llama-3-8B, 80 for 70B)")
    p_mem.add_argument("--d-model",  type=int, required=True,
                       help="Hidden dimension (e.g. 4096 for 8B, 8192 for 70B)")
    p_mem.add_argument("--seq-len",  type=int, default=256,
                       help="Input sequence length (default: 256)")
    p_mem.add_argument("--dtype",    default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="Model dtype (default: bfloat16)")

    # ── doctor ────────────────────────────────────────────────────────────────
    sub.add_parser("doctor",  help="Check all dependencies are correctly installed")

    # ── version ───────────────────────────────────────────────────────────────
    sub.add_parser("version", help="Print installed Glassbox version")

    args = parser.parse_args()

    if args.cmd == "analyze":
        sys.exit(_run_analyze(args))
    elif args.cmd == "estimate-memory":
        sys.exit(_run_estimate_memory(args))
    elif args.cmd == "doctor":
        sys.exit(_run_doctor(args))
    elif args.cmd == "version":
        sys.exit(_run_version(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
