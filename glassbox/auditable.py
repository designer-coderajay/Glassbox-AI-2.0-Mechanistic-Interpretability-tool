# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.auditable
==================
V5 Auditable Interface + conformance suite (ROADMAP_V5_FOUNDATIONS.md Part 4).

The math needs only five capabilities from a model; anything implementing them is
auditable at white-box tier. This module defines that minimal contract
(:class:`AuditableModel`) and the gate every backend must pass
(:func:`run_conformance`) — "the conformance suite, not trust, is the gatekeeper"
(Part 4.3). New-architecture playbook: write a ~100-300 line adapter, pass
conformance, ship.

Scope (honest): the protocol and the conformance *checker* are pure and unit-
tested here against a mock adapter. The three production backends that implement
the protocol — native HF (forward hooks on `transformers`), TransformerLens, and
black-box (audit.py) — are torch-dependent; the native HF backend in particular
is a multi-week build (Part 4.2) and is validated against real models, not here.
Two conformance checks (known-circuit recovery, memory envelope) require a real
reference model and live in the torch test suite; the three checks here
(determinism, patch-identity, reconstruction) run on any adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

__all__ = [
    "UnitSpec",
    "AuditableModel",
    "ConformanceCheck",
    "ConformanceReport",
    "run_conformance",
]


@dataclass(frozen=True)
class UnitSpec:
    """One element of the computation-graph partition (Roadmap 2.2).

    A unit is the granularity attribution flows through: an attention head, an
    MLP block, or — for MoE — an expert (+ router). Adapters declare their units;
    the math never changes, only the partition does.

    Attributes:
        name: Stable identifier, e.g. "L9H6", "mlp_3", "expert_2".
        layer: Layer index the unit lives in.
        kind: "head" | "mlp" | "expert" | "neuron" | "feature" | ...
        index: Within-layer index (head number, expert id); 0 for whole-layer units.
    """

    name: str
    layer: int
    kind: str
    index: int = 0


@runtime_checkable
class AuditableModel(Protocol):
    """The minimal contract the attribution math requires (Part 4.1).

    Five capabilities. Anything implementing them is fully auditable. A black-box
    backend implements the same protocol with ``units() == []``, which forces a
    behavioral-tier report — one API, three tiers, honest labels.
    """

    def forward(self, tokens: Any) -> Any:
        """Run the model and return logits (gradient-tracking where supported)."""
        ...

    def units(self) -> List[UnitSpec]:
        """Return the computation-graph partition. Empty list -> black-box tier."""
        ...

    def read(self, unit: UnitSpec, tokens: Any) -> Any:
        """Read a unit's activation for the given input (hook read)."""
        ...

    def patch(self, unit: UnitSpec, value: Any) -> Any:
        """Return a context manager that overwrites a unit's activation (hook write)."""
        ...


# ---------------------------------------------------------------------------
# Conformance suite (pure: runs on any adapter, used to gate new backends)
# ---------------------------------------------------------------------------
@dataclass
class ConformanceCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class ConformanceReport:
    checks: List[ConformanceCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks]}

    def summary_line(self) -> str:
        n_ok = sum(1 for c in self.checks if c.passed)
        status = "PASS" if self.passed else "FAIL"
        return f"Conformance [{status}] {n_ok}/{len(self.checks)} checks"


def _flatten(x: Any) -> List[float]:
    """Flatten lists/tuples/tensors/numbers to a flat float list."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for e in x:
            out.extend(_flatten(e))
        return out
    return [float(x)]


def _allclose(a: Any, b: Any, tol: float) -> bool:
    fa, fb = _flatten(a), _flatten(b)
    if len(fa) != len(fb):
        return False
    return all(abs(x - y) <= tol for x, y in zip(fa, fb))


def run_conformance(
    adapter: Any,
    tokens: Any,
    *,
    tol: float = 1e-6,
    recon_tol: float = 1e-3,
) -> ConformanceReport:
    """Run the architecture-agnostic conformance checks on an adapter.

    Checks (Part 4.3) expressible from the minimal protocol:
      * determinism   — same input, same logits, twice;
      * patch_identity — patching a unit with its own activation changes nothing;
      * reconstruction — Σ unit contributions ≈ logits (only if the adapter
        exposes ``contributions(tokens) -> {unit_name: vector}``).

    The model-dependent checks (known-circuit recovery, memory envelope) require a
    real reference model and live in the torch conformance tests.

    Returns:
        A :class:`ConformanceReport`; ``report.passed`` is the accept/reject gate.
    """
    checks: List[ConformanceCheck] = []

    # 1. Determinism
    a1 = adapter.forward(tokens)
    a2 = adapter.forward(tokens)
    det = _allclose(a1, a2, tol)
    checks.append(ConformanceCheck(
        "determinism", det,
        "two forwards on the same input are bit-for-bit equal" if det
        else "forwards differ on identical input",
    ))

    # 2. Patch identity
    base = adapter.forward(tokens)
    bad_unit = None
    for u in adapter.units():
        with adapter.patch(u, adapter.read(u, tokens)):
            patched = adapter.forward(tokens)
        if not _allclose(patched, base, tol):
            bad_unit = u.name
            break
    checks.append(ConformanceCheck(
        "patch_identity", bad_unit is None,
        "patching each unit with its own activation is a no-op" if bad_unit is None
        else f"patching unit {bad_unit} with its own activation changed the output",
    ))

    # 3. Reconstruction (optional — needs per-unit contributions)
    if hasattr(adapter, "contributions"):
        contribs = adapter.contributions(tokens)
        logits = _flatten(adapter.forward(tokens))
        summed: List[float] = [0.0] * len(logits)
        ok_len = True
        for vec in contribs.values():
            fv = _flatten(vec)
            if len(fv) != len(logits):
                ok_len = False
                break
            summed = [s + v for s, v in zip(summed, fv)]
        recon = ok_len and _allclose(summed, logits, recon_tol)
        checks.append(ConformanceCheck(
            "reconstruction", recon,
            "sum of unit contributions reconstructs the logits within tolerance"
            if recon else "unit contributions do not sum to the logits",
        ))

    return ConformanceReport(checks=checks)
