#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
scripts/validate_models_matrix.py
=================================
Run the Glassbox pipeline across MULTIPLE open-weight model families and report a
pass/fail matrix. This is the honest version of "test across all open-weight
models": not literally every model (thousands, GPU-bound), but a representative
set spanning different ARCHITECTURES that TransformerLens can load on a laptop —
GPT-2, Pythia (GPT-NeoX), GPT-Neo, OPT. If the gate + audit work across these,
the architecture-agnostic claim has real evidence behind it.

For each model it:
  1. loads it via TransformerLens,
  2. builds a head-level AuditableModel adapter and runs the CONFORMANCE GATE
     (determinism + patch-identity) on real forward hooks,
  3. runs a real analyze() on a known task (IOI) and records circuit +
     faithfulness,
  4. prints a row and (at the end) a summary; writes reports/model_matrix.json.

Per-model failures are caught and reported as FAIL with the reason, so one
unsupported model never kills the whole run.

HONEST SCOPE
------------
- Only OPEN-WEIGHT models loadable by TransformerLens. Closed APIs (GPT-4,
  Claude) are out of scope by construction — no activations/gradients.
- Large / gated models (Llama, Mistral, Phi) need an HF token, more RAM/GPU, and
  TL support; add them with --models once you have the hardware. The defaults are
  small and ungated so this runs on a laptop CPU.
- IOI is used as a consistent cross-model probe (a known capability), so the
  numbers measure "does the tool work on this architecture", not a business task.

USAGE
-----
    python scripts/validate_models_matrix.py                  # default small set
    python scripts/validate_models_matrix.py --max-units 16   # faster gate
    python scripts/validate_models_matrix.py --models gpt2 distilgpt2 EleutherAI/pythia-160m
    python scripts/validate_models_matrix.py --device cuda     # on a GPU box
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, List

# Run-from-checkout convenience (no editable install needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glassbox.auditable import UnitSpec, run_conformance  # noqa: E402

# Default: small, ungated, CPU-runnable, spanning 4 architecture families.
DEFAULT_MODELS = [
    "distilgpt2",                 # GPT-2 family (distilled)
    "gpt2",                       # GPT-2 family
    "EleutherAI/pythia-70m",      # GPT-NeoX family
    "EleutherAI/pythia-160m",     # GPT-NeoX family
    "EleutherAI/gpt-neo-125M",    # GPT-Neo family
    "facebook/opt-125m",          # OPT family
]

# IOI probe (a known capability that exists across small LMs).
PROMPT = "When Mary and John went to the store, John gave a drink to"
CORRECT = " Mary"
INCORRECT = " John"


def _grade(f1: float) -> str:
    if f1 >= 0.80:
        return "A"
    if f1 >= 0.65:
        return "B"
    if f1 >= 0.50:
        return "C"
    return "D"


class _TLHeadAdapter:
    """Minimal AuditableModel over a TransformerLens model (head granularity)."""

    def __init__(self, model: Any, max_units: int | None = None) -> None:
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self._max_units = max_units

    def forward(self, tokens: Any) -> Any:
        return self.model(tokens)[:, -1, :]

    def units(self) -> List[UnitSpec]:
        out = [
            UnitSpec(name=f"L{l}H{h}", layer=l, kind="head", index=h)
            for l in range(self.n_layers)
            for h in range(self.n_heads)
        ]
        return out if self._max_units is None else out[: self._max_units]

    def read(self, unit: UnitSpec, tokens: Any) -> Any:
        name = f"blocks.{unit.layer}.attn.hook_z"
        _, cache = self.model.run_with_cache(tokens, names_filter=name)
        return cache[name][:, :, unit.index, :].clone()

    @contextmanager
    def patch(self, unit: UnitSpec, value: Any) -> Iterator[None]:
        name = f"blocks.{unit.layer}.attn.hook_z"

        def _ov(act: Any, hook: Any) -> Any:  # noqa: ARG001
            act[:, :, unit.index, :] = value
            return act

        with self.model.hooks(fwd_hooks=[(name, _ov)]):
            yield


def _audit_one(name: str, device: str, max_units: int, dtype: str = "float32",
               exact_circuit: bool = False, max_circuit_heads: int = 30) -> dict:
    import torch
    from transformer_lens import HookedTransformer

    from glassbox import GlassboxV2

    t0 = time.time()
    torch_dtype = getattr(torch, dtype)
    model = HookedTransformer.from_pretrained(name, device=device, dtype=torch_dtype)
    model.eval()
    gb = GlassboxV2(model)

    with torch.no_grad():
        adapter = _TLHeadAdapter(model, max_units=max_units)
        conf = run_conformance(adapter, model.to_tokens(PROMPT))

    result = gb.analyze(PROMPT, CORRECT, INCORRECT, exact_circuit=exact_circuit,
                        max_circuit_heads=max_circuit_heads)
    f = result.get("faithfulness", {})
    f1 = float(f.get("f1", 0.0))
    circuit = result.get("circuit", [])

    # Control #2 — specificity: comp of a RANDOM same-size circuit. If the
    # discovered circuit's comp is high while a random one's is low, comp is
    # specific (meaningful). If they're similar, comp=1.0 is not discriminating.
    comp_random = None
    if circuit:
        import random as _random
        tc = model.to_tokens(PROMPT)
        corr = model.to_tokens(gb._name_swap(PROMPT, "Mary", "John"))
        tt = model.to_single_token(CORRECT)
        dt = model.to_single_token(INCORRECT)
        _, clean_ld = gb.attribution_patching(tc, corr, tt, dt)
        all_heads = [(layer, head)
                     for layer in range(model.cfg.n_layers)
                     for head in range(model.cfg.n_heads)]
        rand_circuit = _random.Random(0).sample(all_heads, len(circuit))
        comp_random = round(float(
            gb._comp(rand_circuit, tc, corr, clean_ld, tt, dt)
        ), 3)

    return {
        "model": name,
        "arch": f"{model.cfg.n_layers}L x {model.cfg.n_heads}H",
        "conformance": "PASS" if conf.passed else "FAIL",
        "circuit_size": len(circuit),
        "sufficiency": round(float(f.get("sufficiency", 0.0)), 3),
        "comprehensiveness": round(float(f.get("comprehensiveness", 0.0)), 3),
        "comp_random": comp_random,
        "f1": round(f1, 3),
        "grade": _grade(f1),
        "seconds": round(time.time() - t0, 1),
        "ok": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-units", type=int, default=24,
                    help="heads tested by the conformance gate (lower = faster)")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="load precision. float16 halves memory (use on a 16GB T4 "
                         "for 2.8B-6.9B); bfloat16 needs Ampere+ (A100/L4). "
                         "Attribution gradients are less stable in fp16 — verify.")
    ap.add_argument("--exact-circuit", action="store_true",
                    help="scale-aware circuit selection: grow the circuit using "
                         "MEASURED (exact) sufficiency until truly sufficient. "
                         "Slower (2 passes/head) but fixes the 1-head under-sizing "
                         "that collapses F1 on >1.4B models (see VALIDATION_LOG Run 5).")
    ap.add_argument("--max-circuit-heads", type=int, default=30,
                    help="cap on circuit size for --exact-circuit. Raise it to test "
                         "whether a circuit that hit the cap is truly that size or "
                         "just budget-limited (saturation control).")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import transformer_lens  # noqa: F401
    except ImportError as e:
        print(f"FAIL — real stack not installed: {e}", file=sys.stderr)
        return 1

    rows = []
    for name in args.models:
        print(f"\n>>> {name}")
        try:
            row = _audit_one(name, args.device, args.max_units, args.dtype,
                             args.exact_circuit, args.max_circuit_heads)
        except Exception as e:  # noqa: BLE001 - report, never abort the matrix
            row = {"model": name, "ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"    FAIL — {row['error']}")
        else:
            print(f"    conformance={row['conformance']} | circuit={row['circuit_size']} "
                  f"| suff={row['sufficiency']} comp={row['comprehensiveness']} "
                  f"comp_random={row['comp_random']} F1={row['f1']} ({row['grade']}) "
                  f"| {row['seconds']}s")
        rows.append(row)

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'model':<26}{'arch':<11}{'conf':<6}{'circ':<6}{'suff':<7}{'comp':<7}"
          f"{'cmpRnd':<8}{'F1':<7}{'gr':<4}{'sec':<6}")
    print("-" * 92)
    for r in rows:
        if r.get("ok"):
            print(f"{r['model']:<26}{r['arch']:<11}{r['conformance']:<6}"
                  f"{r['circuit_size']:<6}{r['sufficiency']:<7}{r['comprehensiveness']:<7}"
                  f"{str(r.get('comp_random')):<8}{r['f1']:<7}{r['grade']:<4}{r['seconds']:<6}")
        else:
            print(f"{r['model']:<28}{'ERROR — ' + r['error'][:40]}")
    print("=" * 78)
    n_ok = sum(1 for r in rows if r.get("ok"))
    n_conf = sum(1 for r in rows if r.get("conformance") == "PASS")
    print(f"loaded+audited: {n_ok}/{len(rows)}   conformance PASS: {n_conf}/{len(rows)}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_matrix.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("Wrote reports/model_matrix.json")
    print("\nNote: IOI is a cross-model PROBE — these numbers say whether the tool "
          "works on each architecture, not a business decision. Add larger/gated "
          "models with --models once you have an HF token + GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
