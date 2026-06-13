# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
"""
glassbox.monitoring
===================
V5 post-market monitoring primitives (ROADMAP_V5_FOUNDATIONS.md Part 2.5, 5.4-5.5;
EU AI Act Article 72).

Three pure, deterministic building blocks for the two-loop design:

  * ``CusumDetector``  — sequential change detection on a scalar stream
    (faithfulness, or fingerprint distance). The continuous question Loop 2 asks:
    "is production still inside the regime the deep audit certified?" (Part 2.5).
  * ``JLProjector``    — Johnson-Lindenstrauss random projection of circuit
    activations to a low-dimensional sketch (Part 5.5). Distances are preserved in
    expectation, so a cheap fingerprint stands in for the full activation.
  * ``CircuitCache``   — circuit caching by task-family (Part 5.4): same
    template + model ⇒ same circuit, validated by a fingerprint match; on drift
    the cache misses and discovery re-runs.

Out of scope (torch): capturing the live activations that feed the projector and
the per-request hook path. The capture is torch; everything here operates on the
resulting vectors and is unit-tested.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Optional

import numpy as np

__all__ = ["CusumDetector", "JLProjector", "CircuitCache"]


class CusumDetector:
    """Two-sided tabular CUSUM for sequential drift detection.

    Tracks cumulative deviations of a scalar stream from a reference ``target``.
    An alarm fires when the accumulated upward or downward deviation crosses
    ``threshold``. This is the Article 72 monitor: fed a per-window faithfulness
    or fingerprint-distance statistic, it flags when production has drifted out of
    the certified regime.

    Calibrating the false-alarm rate on real high-dimensional streams is
    unresolved engineering (Part 9.2 hole #4); ``slack``/``threshold`` here are
    explicit knobs, not tuned constants.
    """

    def __init__(self, target: float, slack: float = 0.5, threshold: float = 5.0) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        self.target = float(target)
        self.slack = float(slack)
        self.threshold = float(threshold)
        self._s_hi = 0.0
        self._s_lo = 0.0
        self.n = 0

    def update(self, x: float) -> Dict[str, Any]:
        """Feed one observation; return the current state and any alarm."""
        x = float(x)
        self._s_hi = max(0.0, self._s_hi + (x - self.target - self.slack))
        self._s_lo = max(0.0, self._s_lo + (self.target - x - self.slack))
        self.n += 1
        up = self._s_hi > self.threshold
        down = self._s_lo > self.threshold
        return {
            "alarm": up or down,
            "direction": "up" if up else ("down" if down else None),
            "s_hi": round(self._s_hi, 6),
            "s_lo": round(self._s_lo, 6),
            "n": self.n,
        }

    def reset(self) -> None:
        """Reset accumulators (e.g. after an alarm triggers a Loop-1 re-audit)."""
        self._s_hi = 0.0
        self._s_lo = 0.0


class JLProjector:
    """Johnson-Lindenstrauss random projection to a fixed-size fingerprint.

    Each projected coordinate is a scaled Gaussian combination of the input, so
    E[‖project(v)‖²] = ‖v‖² — pairwise distances are preserved in expectation,
    which is what lets a 128-dim sketch monitor a high-dim activation cheaply.
    Deterministic given ``seed``.
    """

    def __init__(self, d_in: int, d_out: int = 128, seed: int = 0) -> None:
        if d_in < 1 or d_out < 1:
            raise ValueError("d_in and d_out must be >= 1")
        rng = np.random.default_rng(seed)
        self.d_in = d_in
        self.d_out = d_out
        self._mat = rng.standard_normal((d_out, d_in)) / np.sqrt(d_out)

    def project(self, vec: Any) -> np.ndarray:
        v = np.asarray(vec, dtype=float).reshape(-1)
        if v.shape[0] != self.d_in:
            raise ValueError(f"expected input dim {self.d_in}, got {v.shape[0]}")
        return self._mat @ v


class CircuitCache:
    """Circuit cache keyed by task-family, validated by fingerprint match.

    ``put`` stores a discovered circuit with the fingerprint of the run that
    produced it. ``get`` returns the cached circuit only if the current
    fingerprint is within ``fingerprint_tol`` (L2) of the stored one — otherwise
    it is a miss and discovery must re-run. This is what makes "discovery once per
    family, verification per batch" honest: the fingerprint guards the reuse.
    """

    def __init__(self, fingerprint_tol: float = 0.05) -> None:
        self.tol = float(fingerprint_tol)
        self._store: Dict[Hashable, Any] = {}
        self.hits = 0
        self.misses = 0

    def put(self, key: Hashable, circuit: Any, fingerprint: Any) -> None:
        self._store[key] = (circuit, np.asarray(fingerprint, dtype=float).reshape(-1))

    def get(self, key: Hashable, fingerprint: Any) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        circuit, stored_fp = entry
        cur = np.asarray(fingerprint, dtype=float).reshape(-1)
        if cur.shape != stored_fp.shape:
            self.misses += 1
            return None
        if float(np.linalg.norm(cur - stored_fp)) <= self.tol:
            self.hits += 1
            return circuit
        self.misses += 1  # fingerprint drifted -> rediscover
        return None

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._store)}
