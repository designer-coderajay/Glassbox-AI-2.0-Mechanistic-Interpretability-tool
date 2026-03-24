#!/usr/bin/env python3
"""
scripts/benchmark_v340.py
=========================
Glassbox AI v3.4.0 Benchmark Suite

Measures wall-clock time and faithfulness metrics for:
  - Core circuit analysis (GPT-2, GPT-Neo, Pythia, Llama-2/3, Mistral)
  - Multi-agent audit chain performance
  - Steering vector extraction speed
  - Evidence vault build time

Usage
-----
  # Single model, IOI task
  python scripts/benchmark_v340.py --model gpt2 --task ioi --seed 42

  # Single model, credit scoring task (EU AI Act Annex III representative)
  python scripts/benchmark_v340.py --model gpt2 --task credit --seed 42

  # Full benchmark suite (all models in --models, all tasks)
  python scripts/benchmark_v340.py --suite full --output results/bench_v340.json

  # Quick sanity check (GPT-2 only, no model downloads)
  python scripts/benchmark_v340.py --suite quick

Requirements
------------
  pip install glassbox-mech-interp transformer_lens

For larger models (Llama, Mistral):
  pip install glassbox-mech-interp[hf]
  # Requires ~14 GB RAM for Llama-2-7B in float16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Tuple[str, str, str]] = {
    "ioi": (
        "When Mary and John went to the store, John gave a drink to",
        " Mary",
        " John",
    ),
    "credit": (
        "The loan applicant has a credit score of 620. The bank decision is",
        " approved",
        " denied",
    ),
    "medical": (
        "The patient presents with chest pain and elevated troponin. The diagnosis is",
        " cardiac",
        " anxiety",
    ),
    "recruitment": (
        "The job application from the candidate with 5 years of experience was",
        " accepted",
        " rejected",
    ),
}

# Models to benchmark in --suite full mode
# Smaller models first so the suite completes even on low-RAM machines
SUITE_MODELS: Dict[str, List[str]] = {
    "quick": ["gpt2"],
    "standard": ["gpt2", "gpt2-medium", "EleutherAI/gpt-neo-125m", "EleutherAI/pythia-160m"],
    "full": [
        "gpt2",
        "gpt2-medium",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-1.4b",
        # Uncomment for large-model benchmarks (requires 16+ GB RAM):
        # "meta-llama/Llama-2-7b-hf",
        # "mistralai/Mistral-7B-v0.1",
    ],
}


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_analyze(gb, prompt: str, correct: str, incorrect: str, n_runs: int = 3) -> Dict:
    """Warm-start timing: first run discarded, mean of remaining n_runs-1."""
    times = []
    result = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = gb.analyze(prompt=prompt, correct=correct, incorrect=incorrect)
        t1 = time.perf_counter()
        if i > 0:  # discard first (JIT / cache warm-up)
            times.append(t1 - t0)

    faith = result.get("faithfulness", {}) if result else {}
    return {
        "mean_s": round(sum(times) / len(times), 3) if times else None,
        "min_s": round(min(times), 3) if times else None,
        "n_runs": n_runs - 1,
        "sufficiency": faith.get("sufficiency", None),
        "comprehensiveness": faith.get("comprehensiveness", None),
        "f1": faith.get("f1", None),
        "n_heads": result.get("n_heads", None) if result else None,
        "suff_is_approx": faith.get("suff_is_approx", None),
        "grade": result.get("grade", None) if result else None,
    }


def benchmark_multiagent(n_agents: int = 4) -> Dict:
    """Benchmark MultiAgentAudit.audit_chain() — no model inference needed."""
    from glassbox import MultiAgentAudit, AgentCall

    audit = MultiAgentAudit()

    # Build synthetic agent chain
    texts = [
        "The loan applicant Maria Garcia has a credit score of 580.",
        "Application forwarded: female applicant, low score, flagged for review.",
        "High-risk profile detected based on demographic indicators.",
        "Loan denied. Risk category: elevated based on applicant profile.",
    ]

    calls = [
        AgentCall(
            agent_id=f"agent_{i}",
            model_name="synthetic",
            input_text=texts[i % len(texts)],
            output_text=texts[(i + 1) % len(texts)],
        )
        for i in range(n_agents)
    ]

    t0 = time.perf_counter()
    report = audit.audit_chain(calls)
    elapsed = time.perf_counter() - t0

    return {
        "n_agents": n_agents,
        "elapsed_s": round(elapsed, 4),
        "chain_risk_level": report.chain_risk_level,
        "most_liable_agent": report.most_liable_agent,
    }


def benchmark_steering(model, layer: int = 8) -> Dict:
    """Benchmark SteeringVectorExporter extraction and application."""
    from glassbox import SteeringVectorExporter

    exporter = SteeringVectorExporter(method="mean_diff", verbose=False)

    pos = ["The nurse said she would call the doctor."]
    neg = ["The nurse said he would call the doctor."]

    # Extraction
    t0 = time.perf_counter()
    sv = exporter.extract_mean_diff(model, pos, neg, layer=layer, concept_label="gender_bias")
    extract_time = time.perf_counter() - t0

    # Application (greedy next token)
    t0 = time.perf_counter()
    _ = exporter.apply(model, "The nurse said", sv)
    apply_time = time.perf_counter() - t0

    return {
        "extract_s": round(extract_time, 3),
        "apply_s": round(apply_time, 3),
        "layer": sv.layer,
        "d_model": int(sv.direction.shape[0]),
        "norm": round(sv.norm(), 4),
    }


def benchmark_vault(gb_result: Dict) -> Dict:
    """Benchmark AnnexIVEvidenceVault build from a pre-computed result."""
    from glassbox import build_annex_iv_vault

    t0 = time.perf_counter()
    vault = build_annex_iv_vault(
        gb_result=gb_result,
        model_name="benchmark",
        provider="Glassbox Benchmark",
    )
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = vault.to_json()
    json_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = vault.to_html()
    html_time = time.perf_counter() - t0

    summary = vault.to_dict().get("compliance_summary", {})
    return {
        "build_s": round(build_time, 4),
        "json_s": round(json_time, 4),
        "html_s": round(html_time, 4),
        "n_entries": summary.get("n_entries", 0),
        "pass_rate": summary.get("pass_rate", 0),
        "overall_status": summary.get("overall_status", ""),
    }


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_gb(model_name: str, device: str = "cpu"):
    """Load a HookedTransformer + GlassboxV2, return (model, gb)."""
    try:
        from transformer_lens import HookedTransformer
        from glassbox import GlassboxV2
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run:  pip install glassbox-mech-interp transformer_lens")
        sys.exit(1)

    print(f"  Loading {model_name}...", end=" ", flush=True)
    t0 = time.perf_counter()
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"done ({load_time:.1f}s)")

    gb = GlassboxV2(model)
    return model, gb


# ---------------------------------------------------------------------------
# Run single model benchmark
# ---------------------------------------------------------------------------

def run_model_benchmark(
    model_name: str,
    tasks: List[str],
    device: str = "cpu",
    seed: int = 42,
    run_steering: bool = True,
    run_vault: bool = True,
) -> Dict:
    import torch
    torch.manual_seed(seed)

    model, gb = load_model_and_gb(model_name, device)
    n_layers = model.cfg.n_layers
    steer_layer = n_layers // 2

    result = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "device": device,
        "seed": seed,
        "tasks": {},
        "steering": None,
        "vault": None,
        "multiagent": None,
    }

    # --- Task benchmarks ---
    last_gb_result = None
    for task_name in tasks:
        if task_name not in TASKS:
            print(f"  Unknown task: {task_name}, skipping")
            continue

        prompt, correct, incorrect = TASKS[task_name]
        print(f"  Benchmarking task '{task_name}'...", end=" ", flush=True)
        metrics = benchmark_analyze(gb, prompt, correct, incorrect)
        result["tasks"][task_name] = metrics
        last_gb_result = gb.analyze(prompt=prompt, correct=correct, incorrect=incorrect)
        print(f"  {metrics['mean_s']}s  F1={metrics['f1']:.2f}  grade={metrics['grade']}")

    # --- Steering benchmark ---
    if run_steering:
        print(f"  Benchmarking steering vectors (layer {steer_layer})...", end=" ", flush=True)
        try:
            steer_metrics = benchmark_steering(model, layer=steer_layer)
            result["steering"] = steer_metrics
            print(f"  extract={steer_metrics['extract_s']}s  apply={steer_metrics['apply_s']}s")
        except Exception as e:
            print(f"  SKIPPED ({e})")

    # --- Vault benchmark ---
    if run_vault and last_gb_result is not None:
        print("  Benchmarking Evidence Vault...", end=" ", flush=True)
        vault_metrics = benchmark_vault(last_gb_result)
        result["vault"] = vault_metrics
        print(f"  build={vault_metrics['build_s']}s  status={vault_metrics['overall_status']}")

    # --- Multi-agent benchmark (model-independent) ---
    print("  Benchmarking MultiAgentAudit (4 agents)...", end=" ", flush=True)
    ma_metrics = benchmark_multiagent(n_agents=4)
    result["multiagent"] = ma_metrics
    print(f"  {ma_metrics['elapsed_s']}s  risk={ma_metrics['chain_risk_level']}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Glassbox AI v3.4.0 Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="gpt2",
                        help="Model name (default: gpt2)")
    parser.add_argument("--task", default="ioi",
                        choices=list(TASKS.keys()),
                        help="Task to benchmark (default: ioi)")
    parser.add_argument("--suite", default=None,
                        choices=["quick", "standard", "full"],
                        help="Run full benchmark suite (overrides --model/--task)")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device (default: cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--no-steering", action="store_true",
                        help="Skip steering vector benchmarks")
    parser.add_argument("--no-vault", action="store_true",
                        help="Skip Evidence Vault benchmarks")

    args = parser.parse_args()

    print(f"\nGlassbox AI v3.4.0 Benchmark")
    print(f"{'='*50}")

    import platform
    print(f"Python:  {sys.version.split()[0]}")
    print(f"OS:      {platform.system()} {platform.machine()}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"Device:  {args.device}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        sys.exit(1)

    print()

    all_results = []

    if args.suite:
        models = SUITE_MODELS[args.suite]
        tasks = list(TASKS.keys()) if args.suite == "full" else ["ioi", "credit"]
        print(f"Suite: {args.suite} — {len(models)} model(s), {len(tasks)} task(s)\n")
        for m in models:
            print(f"\n[{m}]")
            r = run_model_benchmark(
                m, tasks,
                device=args.device,
                seed=args.seed,
                run_steering=not args.no_steering,
                run_vault=not args.no_vault,
            )
            all_results.append(r)
    else:
        print(f"Model: {args.model}  Task: {args.task}\n")
        r = run_model_benchmark(
            args.model, [args.task],
            device=args.device,
            seed=args.seed,
            run_steering=not args.no_steering,
            run_vault=not args.no_vault,
        )
        all_results.append(r)

    # Print summary table
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<30} {'Task':<12} {'F1':>6} {'Grade':>6} {'Time(s)':>8}")
    print("-" * 65)
    for r in all_results:
        for task_name, m in r["tasks"].items():
            f1_str = f"{m['f1']:.2f}" if m["f1"] is not None else "N/A"
            grade_str = m["grade"] or "N/A"
            time_str = f"{m['mean_s']:.2f}" if m["mean_s"] is not None else "N/A"
            print(f"{r['model']:<30} {task_name:<12} {f1_str:>6} {grade_str:>6} {time_str:>8}")

    if args.output:
        output_data = {
            "glassbox_version": "3.4.0",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": vars(args),
            "results": all_results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(output_data, fh, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print()


if __name__ == "__main__":
    main()
