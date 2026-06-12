#!/usr/bin/env python3
"""
docker_scan.py — Glassbox AI Standalone Compliance Scanner v4.3.0
==================================================================

Entrypoint for the glassbox-scan Docker image.

Runs circuit discovery + faithfulness analysis on any prompt,
then generates a full EU AI Act Annex IV evidence package as PDF + JSON.

Usage inside the container:
    python3 docker_scan.py \\
        --model gpt2 \\
        --prompt "Loan application. Annual income: €42,000. Decision:" \\
        --correct " Approved" \\
        --incorrect " Denied" \\
        --purpose "Credit risk scoring" \\
        --provider "Acme Bank NV" \\
        --output /output/annex_iv.pdf

All arguments are optional except --prompt, --correct, --incorrect.
The container mounts /output as a volume to retrieve the generated files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _banner() -> None:
    print("""
  ╔══════════════════════════════════════════════════════╗
  ║   Glassbox AI — Compliance Scanner v4.3.0           ║
  ║   EU AI Act Annex IV Evidence Package Generator     ║
  ╚══════════════════════════════════════════════════════╝
""")


def _check_output_dir(output_path: str) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_model(model_name: str, dtype_str: str | None):
    """Load HookedTransformer with optional dtype and memory planning."""
    import torch
    from transformer_lens import HookedTransformer

    from glassbox.large_model import classify_model_size, estimate_memory

    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str or "float32", torch.float32)

    print(f"  Loading model: {model_name}")
    print(f"  Dtype        : {dtype_str or 'float32 (default)'}")

    t0 = time.time()
    load_kwargs = {}
    if dtype != torch.float32:
        load_kwargs["dtype"] = dtype

    model = HookedTransformer.from_pretrained(model_name, **load_kwargs)
    elapsed = time.time() - t0

    # Memory estimate
    n_layers = model.cfg.n_layers
    d_model  = getattr(model.cfg, "d_model", 768)
    mem = estimate_memory(n_layers, d_model, dtype=dtype)
    size_class = classify_model_size(n_layers, d_model)

    print(f"  Loaded in    : {elapsed:.1f}s")
    print(f"  Parameters   : {mem.n_params / 1e9:.2f}B ({size_class})")
    print(f"  Attribution  : ~{mem.attribution_gb:.1f} GB VRAM")
    for w in mem.warnings:
        print(f"  ⚠  {w}")
    print()

    return model, dtype, mem.recommend_strategy


def _run_analysis(model, gb, args, strategy: str):
    """Run circuit discovery and return result dict."""
    import torch

    from glassbox.large_model import analyze_large, classify_model_size

    n_layers = model.cfg.n_layers
    d_model  = getattr(model.cfg, "d_model", 768)
    size_class = classify_model_size(n_layers, d_model)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype or "float32", torch.float32)

    # Use large-model adapter for 7B+ models
    if size_class in ("large", "xlarge", "xxlarge") or args.strategy:
        effective_strategy = args.strategy or strategy
        result = analyze_large(
            gb, args.prompt, args.correct, args.incorrect,
            strategy=effective_strategy, dtype=dtype,
        )
    else:
        result = gb.analyze(
            args.prompt, args.correct, args.incorrect,
            corruption_strategy=args.strategy or "auto",
        )
    return result


def _generate_report(result: dict, args: argparse.Namespace, output_pdf: Path) -> dict:
    """Generate Annex IV PDF + JSON from analysis result."""
    try:
        from glassbox.compliance import AnnexIVReport, DeploymentContext
    except ImportError:
        print("  WARNING: compliance module not available — generating JSON only")
        return result

    # Map user-specified context string to DeploymentContext enum
    context_map = {
        "finance":       DeploymentContext.FINANCIAL_SERVICES,
        "financial":     DeploymentContext.FINANCIAL_SERVICES,
        "healthcare":    DeploymentContext.HEALTHCARE,
        "medical":       DeploymentContext.HEALTHCARE,
        "hr":            DeploymentContext.EMPLOYMENT,
        "employment":    DeploymentContext.EMPLOYMENT,
        "education":     DeploymentContext.EDUCATION,
        "legal":         DeploymentContext.CRITICAL_INFRASTRUCTURE,
        "infrastructure":DeploymentContext.CRITICAL_INFRASTRUCTURE,
    }
    ctx_key = (args.context or "").lower()
    context = context_map.get(ctx_key, DeploymentContext.GENERAL)

    report = AnnexIVReport(
        model_name         = args.model,
        system_purpose     = args.purpose or "AI system analysis via Glassbox v4.3.0",
        provider_name      = args.provider or "Organisation name not provided",
        provider_address   = args.address  or "Address not provided",
        deployment_context = context,
    )
    report.add_analysis(result)

    # Save PDF
    try:
        report.to_pdf(str(output_pdf))
        print(f"  ✓ Annex IV PDF  : {output_pdf}")
    except Exception as e:
        print(f"  ✗ PDF generation failed: {e}")
        print("    (Saving JSON only — install weasyprint for PDF support)")

    # Save JSON
    json_path = output_pdf.with_suffix(".json")
    try:
        report.to_json(str(json_path))
        print(f"  ✓ Evidence JSON : {json_path}")
    except Exception:
        # Fallback: save raw result dict
        json_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  ✓ Raw JSON      : {json_path}")

    return result


def _print_summary(result: dict) -> None:
    """Print circuit + faithfulness summary to stdout."""
    faith = result.get("faithfulness", {})
    corruption = result.get("corruption_metadata", {})
    circuit = result.get("circuit", [])

    print("\n  ─── Analysis Summary ───────────────────────────────────────")
    if corruption:
        print(f"  Corruption strategy : {corruption.get('strategy', '?')}")
    print(f"  Faithfulness F1     : {faith.get('f1', 0):.1%}")
    print(f"  Sufficiency         : {faith.get('sufficiency', 0):.1%}")
    print(f"  Comprehensiveness   : {faith.get('comprehensiveness', 0):.1%}")
    print(f"  Category            : {faith.get('category', '?')}")
    print(f"  Circuit size        : {len(circuit)} heads")
    print(f"  Clean logit diff    : {result.get('clean_ld', 0):.4f}")
    print()

    if circuit:
        attrs = result.get("attributions", {})
        print(f"  {'Head':<12} {'Attribution':>12}")
        print(f"  {'─'*12} {'─'*12}")
        for lh in circuit[:10]:
            l, h = lh
            score = attrs.get(str((l, h)), 0.0)
            print(f"  L{l:02d}H{h:02d}       {score:>12.4f}")
        if len(circuit) > 10:
            print(f"  ... and {len(circuit) - 10} more heads")


def main() -> int:
    _banner()

    parser = argparse.ArgumentParser(
        prog="glassbox-scan",
        description="Glassbox AI — Standalone EU AI Act Annex IV report generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Credit scoring:
    docker run --rm -v $(pwd)/output:/output glassbox-scan \\
      --model gpt2 \\
      --prompt "Loan application. Annual income: €42,000. Decision:" \\
      --correct " Approved" --incorrect " Denied" \\
      --purpose "Credit risk scoring" --provider "Acme Bank NV"

  Medical triage (large model):
    docker run --rm -v $(pwd)/output:/output glassbox-scan \\
      --model meta-llama/Llama-3-8B --dtype bfloat16 \\
      --prompt "Patient presents with chest pain. Priority:" \\
      --correct " Urgent" --incorrect " Routine" \\
      --context healthcare --purpose "Medical triage AI"
""",
    )

    # Required
    parser.add_argument("--prompt",    required=True,
                        help="Input prompt (any domain)")
    parser.add_argument("--correct",   required=True,
                        help="Correct next token (e.g. ' Approved')")
    parser.add_argument("--incorrect", required=True,
                        help="Distractor token (e.g. ' Denied')")

    # Model
    parser.add_argument("--model",    default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--dtype",    default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype. Use bfloat16 for 7B+ models.")
    parser.add_argument("--strategy", default=None,
                        choices=["auto", "name_swap", "random_token",
                                 "antonym", "semantic_negation"],
                        help="Corruption strategy (default: auto)")

    # Report metadata
    parser.add_argument("--purpose",  default=None,
                        help="System purpose for Annex IV report (e.g. 'Credit risk scoring')")
    parser.add_argument("--provider", default=None,
                        help="Provider/organisation name")
    parser.add_argument("--address",  default=None,
                        help="Provider address")
    parser.add_argument("--context",  default=None,
                        choices=["finance", "financial", "healthcare", "medical",
                                 "hr", "employment", "education", "legal",
                                 "infrastructure"],
                        help="Deployment context (determines Annex III risk category)")

    # Output
    parser.add_argument("--output",   default="/output/annex_iv_report.pdf",
                        help="Output PDF path (default: /output/annex_iv_report.pdf)")
    parser.add_argument("--json-only", action="store_true",
                        help="Skip PDF, output JSON only")

    args = parser.parse_args()

    # Print config
    print(f"  Prompt    : {args.prompt!r}")
    print(f"  Correct   : {args.correct!r}")
    print(f"  Incorrect : {args.incorrect!r}")
    print(f"  Model     : {args.model}")
    print(f"  Output    : {args.output}")
    print()

    output_pdf = _check_output_dir(args.output)

    t_start = time.time()

    # Load model
    try:
        model, dtype, recommended_strategy = _load_model(args.model, args.dtype)
    except Exception as e:
        print(f"\n  ERROR loading model: {e}")
        print("  If loading from HuggingFace Hub, ensure HF_TOKEN is set for gated models.")
        return 1

    # Create GlassboxV2
    from glassbox import GlassboxV2
    gb = GlassboxV2(model)

    # Run analysis
    print("  Running circuit discovery + faithfulness analysis...")
    try:
        result = _run_analysis(model, gb, args, recommended_strategy)
    except Exception as e:
        print(f"\n  ERROR during analysis: {e}")
        import traceback; traceback.print_exc()
        return 1

    # Print summary
    _print_summary(result)

    # Generate report
    if not args.json_only:
        print("  Generating Annex IV evidence package...")
        _generate_report(result, args, output_pdf)
    else:
        json_path = output_pdf.with_suffix(".json")
        json_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"  ✓ JSON saved: {json_path}")

    elapsed = time.time() - t_start
    print(f"\n  ✓ Complete in {elapsed:.1f}s")
    print(f"  Files are in: {output_pdf.parent}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
