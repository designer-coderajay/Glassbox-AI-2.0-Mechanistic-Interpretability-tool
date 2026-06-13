#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
validate_auditable_gpt2.py — real-model validation of the V5 Phase B/C cores.

WHY THIS EXISTS
---------------
The Phase B/C modules ship with pure, unit-tested cores, but several carry an
honest docstring caveat: "validated against real models, not here." This script
removes that caveat for the pieces that can be validated on a small real model.
It does NOT run in the CI sandbox (no torch); run it locally / on the M1 / on a
GPU box where the real stack is installed.

WHAT IT VALIDATES
-----------------
1. The conformance gate (#5) against a REAL GPT-2 Small via a ~40-line
   TransformerLens adapter that implements the 5-capability AuditableModel
   protocol (forward / units / read / patch) with real forward hooks on
   `hook_z`. This is the governance gate the whole architecture rests on —
   "pass conformance, ship." Proving a real adapter passes is the point.
2. The monitoring cores (#7) on REAL activations: JLProjector fingerprints a
   real head activation, CircuitCache hits on a matching fingerprint and misses
   on a drifted one, CusumDetector flags a synthetic drift in a scalar stream.
3. The framework packs (#12) emit a NIST AI RMF + ISO 42001 cross-walk.

HONEST SCOPE
------------
This validates the conformance checks expressible from the minimal protocol
(determinism, patch-identity). The model-dependent checks named in
auditable.run_conformance (known-circuit recovery, memory envelope) and the
reconstruction check (needs a per-unit `contributions` map) are NOT exercised
here — they require the full attribution backend and a reference circuit, which
is the next build, not this validation. Treat a PASS as "the gate accepts a real
model adapter and the pure cores operate on real tensors", nothing more.

USAGE
-----
    pip install "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6"
    pip install -e . --no-deps
    python scripts/validate_auditable_gpt2.py                 # all 144 heads
    python scripts/validate_auditable_gpt2.py --max-units 12  # quick smoke
    python scripts/validate_auditable_gpt2.py --device cuda    # on GPU

Exit code 0 = all validations passed; 1 = something failed (details printed).
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from typing import Any, Iterator, List

from glassbox.auditable import UnitSpec, run_conformance
from glassbox.frameworks import framework_pack
from glassbox.monitoring import CircuitCache, CusumDetector, JLProjector


class TLHeadAdapter:
    """Minimal AuditableModel over a TransformerLens model, head-granular.

    Implements the five capabilities the attribution math needs. Units are
    attention heads; read/patch act on ``blocks.{l}.attn.hook_z`` (shape
    ``[batch, pos, n_heads, d_head]``). forward returns the last-position logits
    (the decision-relevant slice), keeping the conformance comparisons light.
    """

    def __init__(self, model: Any, max_units: int | None = None) -> None:
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self._max_units = max_units

    def forward(self, tokens: Any) -> Any:
        # Last-position logits only: [batch, d_vocab]. Deterministic at eval.
        return self.model(tokens)[:, -1, :]

    def units(self) -> List[UnitSpec]:
        out: List[UnitSpec] = []
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                out.append(UnitSpec(name=f"L{layer}H{head}", layer=layer,
                                    kind="head", index=head))
        return out if self._max_units is None else out[: self._max_units]

    def read(self, unit: UnitSpec, tokens: Any) -> Any:
        name = f"blocks.{unit.layer}.attn.hook_z"
        _, cache = self.model.run_with_cache(tokens, names_filter=name)
        # [batch, pos, d_head] for this head — clone so it survives the next run.
        return cache[name][:, :, unit.index, :].clone()

    @contextmanager
    def patch(self, unit: UnitSpec, value: Any) -> Iterator[None]:
        name = f"blocks.{unit.layer}.attn.hook_z"

        def _overwrite(act: Any, hook: Any) -> Any:  # noqa: ARG001
            act[:, :, unit.index, :] = value
            return act

        with self.model.hooks(fwd_hooks=[(name, _overwrite)]):
            yield


def _validate_monitoring(model: Any, tokens: Any) -> bool:
    """Exercise JLProjector / CircuitCache / CusumDetector on real activations."""
    import torch

    name = "blocks.0.attn.hook_z"
    _, cache = model.run_with_cache(tokens, names_filter=name)
    vec = cache[name][0, -1].flatten().float().cpu().numpy()  # real activation

    proj = JLProjector(d_in=vec.shape[0], d_out=128, seed=0)
    fp = proj.project(vec)

    cache_obj = CircuitCache(fingerprint_tol=0.05)
    cache_obj.put("ioi-family", circuit=[(9, 6), (10, 0)], fingerprint=fp)
    hit = cache_obj.get("ioi-family", fingerprint=fp)
    drifted = fp + 1.0  # well outside tol
    miss = cache_obj.get("ioi-family", fingerprint=drifted)

    det = CusumDetector(target=0.70, slack=0.02, threshold=0.5)
    alarmed = False
    for x in [0.71, 0.69, 0.70, 0.55, 0.50, 0.48, 0.45]:  # a clear downward drift
        if det.update(x)["alarm"]:
            alarmed = True
            break

    ok = (hit is not None) and (miss is None) and alarmed and (fp.shape[0] == 128)
    print(f"  JLProjector: real activation {vec.shape[0]}-d -> {fp.shape[0]}-d fingerprint")
    print(f"  CircuitCache: matching fp hit={hit is not None}, drifted fp miss={miss is None}")
    print(f"  CusumDetector: flagged downward drift={alarmed}")
    del torch
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2", help="TransformerLens model name")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--prompt",
                    default="When Mary and John went to the store, John gave a drink to")
    ap.add_argument("--max-units", type=int, default=None,
                    help="limit heads tested (faster smoke test)")
    args = ap.parse_args()

    try:
        import torch
        from transformer_lens import HookedTransformer
    except ImportError as e:
        print(f"FAIL — real stack not installed: {e}", file=sys.stderr)
        print("Install: pip install torch transformer-lens transformers", file=sys.stderr)
        return 1

    print(f"Loading {args.model} on {args.device} ...")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()
    tokens = model.to_tokens(args.prompt)

    all_ok = True
    with torch.no_grad():
        # 1. Conformance gate against the real model -----------------------
        adapter = TLHeadAdapter(model, max_units=args.max_units)
        n = len(adapter.units())
        print(f"\n[1] Conformance gate on real {args.model} ({n} head-units)...")
        report = run_conformance(adapter, tokens)
        print("   ", report.summary_line())
        for c in report.checks:
            print(f"    - {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.detail}")
        all_ok = all_ok and report.passed

        # 2. Monitoring cores on real activations -------------------------
        print("\n[2] Monitoring cores on real activations...")
        mon_ok = _validate_monitoring(model, tokens)
        print(f"    monitoring cores: {'PASS' if mon_ok else 'FAIL'}")
        all_ok = all_ok and mon_ok

    # 3. Framework packs --------------------------------------------------
    print("\n[3] Framework packs (NIST AI RMF + ISO 42001)...")
    pack = framework_pack(["faithfulness", "annex_iv", "drift_monitoring", "evidence_tier"])
    print(f"    NIST functions covered: {sorted(pack['nist_ai_rmf'])}")
    print(f"    ISO 42001 objectives covered: {sorted(pack['iso_42001'])}")
    pack_ok = bool(pack["nist_ai_rmf"]) and bool(pack["iso_42001"])
    all_ok = all_ok and pack_ok

    print("\n" + ("=" * 60))
    print("RESULT:", "ALL VALIDATIONS PASSED" if all_ok else "SOME VALIDATIONS FAILED")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
