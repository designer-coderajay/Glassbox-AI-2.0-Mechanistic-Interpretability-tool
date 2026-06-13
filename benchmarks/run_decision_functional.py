#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
V5 decision-functional faithfulness benchmark (ROADMAP_V5_FOUNDATIONS.md Part 9.3).

Runs the non-IOI decision suite (benchmarks/decision_tasks.py) through
``GlassboxV2.analyze()`` using V5 verbalizer sets, and reports, per task:

  * faithfulness — sufficiency / comprehensiveness / F1 (ERASER metrics, adapted
    to circuits; Carton et al. / DeYoung et al. 2020);
  * an attribution-concentration baseline — the ERASER gap fix: faithfulness
    numbers with no random comparator cannot detect anti-faithfulness
    (cf. "Normalized AOPC", ACL 2025). We report what fraction of total
    attribution mass the discovered circuit captures versus the fraction a
    random same-size head set would capture (uniform expectation = n/N);
  * an evidence tier (glassbox.evidence_tier) — single-prompt runs are honestly
    flagged underpowered; distributional confidence intervals are Phase B.

Terminology follows the 2025 Mechanistic Interpretability Benchmark (MIB):
faithfulness and minimality/concentration.

The model path is lazy-imported, so the scoring and aggregation functions in this
module import and unit-test without torch.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Dict, List, Optional, Sequence

from benchmarks.decision_tasks import DECISION_TASKS, DecisionTask
from glassbox.evidence_tier import TierEngine, TierSignals

__all__ = [
    "attribution_concentration",
    "assess_tier",
    "run_task",
    "build_report",
    "format_table",
    "run_all",
]


# ---------------------------------------------------------------------------
# Pure scoring helpers (no model dependency — unit-tested directly)
# ---------------------------------------------------------------------------
def attribution_concentration(
    attributions: Dict[str, float], n_circuit: int
) -> Dict[str, Any]:
    """ERASER-fix random comparator, computed from public attribution output.

    A circuit that is no more concentrated than a random same-size head set is a
    red flag for anti-faithfulness. We compare the attribution-mass fraction held
    by the top ``n_circuit`` heads against the uniform expectation ``n/N``.

    Args:
        attributions: ``{str((layer, head)): attribution}`` for ALL heads
            (the ``attributions`` field of ``analyze()``).
        n_circuit: Size of the discovered minimal faithful circuit.

    Returns:
        Dict with the circuit's mass fraction, the random expectation, their
        ratio, and an ``above_random`` flag.
    """
    mags = [abs(float(v)) for v in attributions.values()]
    total = sum(mags)
    n_total = len(mags)
    if total <= 0.0 or n_total == 0 or n_circuit <= 0:
        return {
            "circuit_mass_fraction": 0.0,
            "random_expected_fraction": 0.0,
            "concentration_ratio": 0.0,
            "above_random": False,
            "n_heads": max(0, n_circuit),
            "n_total": n_total,
        }
    n = min(n_circuit, n_total)
    top = sorted(mags, reverse=True)[:n]
    circuit_frac = sum(top) / total
    random_frac = n / n_total
    ratio = circuit_frac / random_frac if random_frac > 0 else float("inf")
    return {
        "circuit_mass_fraction": round(circuit_frac, 4),
        "random_expected_fraction": round(random_frac, 4),
        "concentration_ratio": round(ratio, 2),
        # Require a genuine margin over the uniform-random expectation: a circuit
        # that merely ties random (ratio == 1.0) is not evidence of concentration.
        "above_random": round(ratio, 2) > 1.0,
        "n_heads": n,
        "n_total": n_total,
    }


def assess_tier(
    *,
    has_weights: bool,
    counterfactual_valid: Optional[bool],
    hessian_reliable: Optional[bool],
    sample_n: Optional[int],
) -> Dict[str, Any]:
    """Run the evidence-tier engine for one task result."""
    sig = TierSignals(
        has_weights=has_weights,
        counterfactual_valid=counterfactual_valid,
        hessian_reliable=hessian_reliable,
        behavioral_possible=True,
        sample_n=sample_n,
        min_sample_n=20,
    )
    return TierEngine().assess(sig).to_dict()


def run_task(
    engine: Any,
    task: DecisionTask,
    method: str = "taylor",
    *,
    ld_floor: float = 0.10,
) -> Dict[str, Any]:
    """Audit one decision task. ``engine`` needs an ``analyze()`` method.

    Returns one report row. Does not raise on a model that fails the task; the
    failure is recorded (``matches_expected=False``) so it is visible, not hidden.
    """
    result = engine.analyze(
        task.prompt,
        list(task.positive_variants),
        list(task.negative_variants),
        method=method,
    )
    faith = result.get("faithfulness", {}) or {}
    clean_ld = float(result.get("clean_ld", 0.0))
    circuit = result.get("circuit", []) or []
    n_heads = int(result.get("n_heads", len(circuit)))
    attributions = result.get("attributions", {}) or {}

    model_decision = "positive" if clean_ld > 0 else "negative"
    matches_expected = model_decision == task.expected

    conc = attribution_concentration(attributions, n_heads)

    # Counterfactual is "valid" only if the model actually performs the task and
    # the decision value is meaningfully non-zero (otherwise ΔD measures noise).
    cf_valid: Optional[bool] = True if (matches_expected and abs(clean_ld) >= ld_floor) else None
    # Single prompt → sample_n=1 → tier engine flags underpowered (honest).
    tier = assess_tier(
        has_weights=True,
        counterfactual_valid=cf_valid,
        hessian_reliable=None,
        sample_n=1,
    )

    return {
        "task": task.name,
        "domain": task.domain,
        "annex_iii": task.annex_iii_ref,
        "expected": task.expected,
        "model_decision": model_decision,
        "matches_expected": matches_expected,
        "clean_ld": round(clean_ld, 4),
        "n_heads": n_heads,
        "sufficiency": faith.get("sufficiency"),
        "comprehensiveness": faith.get("comprehensiveness"),
        "f1": faith.get("f1"),
        "category": faith.get("category"),
        "concentration": conc,
        "tier": tier["tier"],
        "tier_label": tier["label"],
    }


def build_report(rows: List[Dict[str, Any]], *, model: str, method: str) -> Dict[str, Any]:
    """Aggregate per-task rows into a benchmark report."""
    scored = [r for r in rows if r.get("matches_expected") and r.get("f1") is not None]

    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in scored if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 4) if vals else None

    return {
        "benchmark": "decision_functional_faithfulness",
        "model": model,
        "method": method,
        "n_tasks": len(rows),
        "n_model_correct": sum(1 for r in rows if r.get("matches_expected")),
        "n_above_random": sum(1 for r in rows if r["concentration"]["above_random"]),
        "mean_sufficiency": _mean("sufficiency"),
        "mean_comprehensiveness": _mean("comprehensiveness"),
        "mean_f1": _mean("f1"),
        "note": (
            "Means are over tasks the model actually performs. Single-prompt "
            "tiers are underpowered by design; distributional CIs are Phase B."
        ),
        "rows": rows,
    }


def format_table(report: Dict[str, Any]) -> str:
    """Render a compact text table for the console."""
    lines = [
        f"Decision-functional benchmark — model={report['model']} method={report['method']}",
        f"{'task':<22} {'exp':<4} {'got':<4} {'suff':>6} {'comp':>6} {'F1':>6} {'conc×':>6} {'tier':>5}",
        "-" * 72,
    ]
    for r in report["rows"]:
        def _f(x: Any) -> str:
            return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
        lines.append(
            f"{r['task']:<22} {r['expected'][:3]:<4} {r['model_decision'][:3]:<4} "
            f"{_f(r['sufficiency']):>6} {_f(r['comprehensiveness']):>6} {_f(r['f1']):>6} "
            f"{r['concentration']['concentration_ratio']:>5}x {r['tier']:>5}"
        )
    lines.append("-" * 72)
    lines.append(
        f"model-correct: {report['n_model_correct']}/{report['n_tasks']}  ·  "
        f"above-random: {report['n_above_random']}/{report['n_tasks']}  ·  "
        f"mean F1 (correct tasks): {report['mean_f1']}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model path (lazy import: torch only loads when actually running)
# ---------------------------------------------------------------------------
def run_all(
    model: str = "gpt2",
    method: str = "taylor",
    tasks: Optional[Sequence[DecisionTask]] = None,
) -> Dict[str, Any]:
    """Load the model and run the full suite. Requires torch + transformer_lens."""
    from transformer_lens import HookedTransformer  # lazy

    from glassbox import GlassboxV2  # lazy

    hooked = HookedTransformer.from_pretrained(model)
    engine = GlassboxV2(hooked)
    rows = [run_task(engine, t, method=method) for t in (tasks or DECISION_TASKS)]
    return build_report(rows, model=model, method=method)


def main() -> None:
    parser = argparse.ArgumentParser(description="V5 decision-functional faithfulness benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--method", default="taylor", choices=["taylor", "integrated_gradients"])
    parser.add_argument("--out", default=None, help="Write the JSON report to this path")
    args = parser.parse_args()

    report = run_all(model=args.model, method=args.method)
    print(format_table(report))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
