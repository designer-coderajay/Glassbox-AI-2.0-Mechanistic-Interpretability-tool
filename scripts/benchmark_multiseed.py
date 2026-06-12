#!/usr/bin/env python3
"""
scripts/benchmark_multiseed.py
==============================
Multi-seed benchmark runner — the credibility upgrade over single-run numbers.

Runs the existing benchmark suite (scripts/benchmark.py) across N seeds,
aggregates mean ± std for every timing and faithfulness metric, and emits:

  1. results/bench_multiseed_<model>.json   — raw per-seed data + summary
  2. A markdown table on stdout, ready to paste into BENCHMARKS.md

Usage
-----
  # The canonical credibility run (do this on the M1, ~15 min):
  python scripts/benchmark_multiseed.py --model gpt2 --tasks ioi credit --seeds 10

  # Quick sanity (3 seeds):
  python scripts/benchmark_multiseed.py --model gpt2 --tasks credit --seeds 3

  # Specific seed list:
  python scripts/benchmark_multiseed.py --model gpt2 --tasks ioi --seed-list 1 7 42 123 999

Notes
-----
- Wall-clock timings use the same warm-start protocol as benchmark.py
  (first run discarded per seed).
- Faithfulness metrics may be identical across seeds for deterministic
  pipelines — that is itself a publishable result (report std = 0.0).
- Hardware, commit hash, and library versions are recorded in the JSON
  so the run is reproducible and citable.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _env_fingerprint() -> dict:
    """Record everything needed to reproduce/cite this run."""
    fp = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        fp["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, check=False,
        ).stdout.strip() or None
    except OSError:
        fp["git_commit"] = None
    for mod in ("torch", "transformer_lens", "glassbox"):
        try:
            m = __import__(mod)
            fp[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except ImportError:
            fp[f"{mod}_version"] = None
    return fp


def _agg(values: list) -> dict:
    """Mean/std/min/max for a list of numbers (None-safe)."""
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.fmean(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-seed Glassbox benchmark")
    ap.add_argument("--model", default="gpt2", help="TransformerLens model name")
    ap.add_argument("--tasks", nargs="+", default=["ioi", "credit"],
                    help="Tasks from benchmark.py TASKS (ioi, credit, ...)")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Number of seeds (uses 1..N). Ignored if --seed-list given.")
    ap.add_argument("--seed-list", nargs="+", type=int, default=None,
                    help="Explicit seed values")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: results/bench_multiseed_<model>.json)")
    args = ap.parse_args()

    seeds = args.seed_list if args.seed_list else list(range(1, args.seeds + 1))
    if len(seeds) < 10:
        print(f"NOTE: {len(seeds)} seeds < 10 — fine for a sanity check, "
              f"but use >=10 for any number you publish.\n")

    # Import the existing suite (heavy deps load here, after arg parsing).
    try:
        import benchmark as bench  # scripts/benchmark.py
    except ImportError as err:
        print(f"Could not import scripts/benchmark.py ({err}).\n"
              "Run from the repo root: python scripts/benchmark_multiseed.py ...")
        return 1

    runs = []
    t_start = time.perf_counter()
    for i, seed in enumerate(seeds, 1):
        print(f"── seed {seed} ({i}/{len(seeds)}) ─ model={args.model} "
              f"tasks={','.join(args.tasks)}")
        try:
            r = bench.run_model_benchmark(
                model_name=args.model,
                tasks=args.tasks,
                device=args.device,
                seed=seed,
                run_steering=False,   # timing focus; steering benched separately
                run_vault=False,
            )
            r["seed"] = seed
            runs.append(r)
        except Exception as exc:  # record, don't abort the whole campaign
            print(f"  seed {seed} FAILED: {exc}")
            runs.append({"seed": seed, "error": str(exc), "tasks": {}})
    total_s = round(time.perf_counter() - t_start, 1)

    # ── Aggregate per task ────────────────────────────────────────────────
    summary: dict = {}
    for task in args.tasks:
        metrics = [r["tasks"].get(task, {}) for r in runs if r.get("tasks", {}).get(task)]
        if not metrics:
            summary[task] = {"error": "no successful runs"}
            continue
        summary[task] = {
            "time_s": _agg([m.get("mean_s") for m in metrics]),
            "sufficiency": _agg([m.get("sufficiency") for m in metrics]),
            "comprehensiveness": _agg([m.get("comprehensiveness") for m in metrics]),
            "f1": _agg([m.get("f1") for m in metrics]),
            "n_heads": _agg([m.get("n_heads") for m in metrics]),
            "grades": sorted({str(m.get("grade")) for m in metrics}),
            "suff_is_approx": sorted({str(m.get("suff_is_approx")) for m in metrics}),
        }

    payload = {
        "model": args.model,
        "device": args.device,
        "seeds": seeds,
        "n_successful": sum(1 for r in runs if not r.get("error")),
        "total_wall_clock_s": total_s,
        "environment": _env_fingerprint(),
        "summary": summary,
        "runs": runs,
    }

    out = Path(args.output) if args.output else ROOT / "results" / f"bench_multiseed_{args.model.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n✓ Raw data → {out}")

    # ── Markdown table for BENCHMARKS.md ──────────────────────────────────
    print("\nPaste into BENCHMARKS.md:\n")
    print(f"### {args.model} — {len(seeds)} seeds, {args.device}, "
          f"commit {payload['environment'].get('git_commit')}\n")
    print("| Task | Time (s, mean±std) | Sufficiency | Comprehensiveness | F1 | Heads | Grade(s) |")
    print("|---|---|---|---|---|---|---|")
    for task, s in summary.items():
        if "error" in s:
            print(f"| {task} | — failed — | | | | | |")
            continue
        def f(k):
            a = s[k]
            return f"{a['mean']} ± {a['std']}" if a["mean"] is not None else "—"
        print(f"| {task} | {f('time_s')} | {f('sufficiency')} | "
              f"{f('comprehensiveness')} | {f('f1')} | {f('n_heads')} | "
              f"{', '.join(s['grades'])} |")
    print(f"\n_Total campaign: {total_s}s · {payload['n_successful']}/{len(seeds)} seeds succeeded · "
          "first run per seed discarded (warm-start protocol, see scripts/benchmark.py)._")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
