"""Tests for glassbox/monitoring.py — CUSUM drift, JL fingerprint, circuit cache."""
import numpy as np
import pytest

from glassbox.monitoring import CircuitCache, CusumDetector, JLProjector


# ── CusumDetector ───────────────────────────────────────────────────────────
def test_cusum_invalid_threshold():
    with pytest.raises(ValueError):
        CusumDetector(target=0.0, threshold=0.0)


def test_cusum_stable_stream_no_alarm():
    d = CusumDetector(target=0.0, slack=0.5, threshold=3.0)
    states = [d.update(x) for x in (0.1, -0.1, 0.0, 0.2, -0.2, 0.1)]
    assert not any(s["alarm"] for s in states)


def test_cusum_detects_upward_shift():
    d = CusumDetector(target=0.0, slack=0.5, threshold=3.0)
    last = {}
    for _ in range(4):
        last = d.update(2.0)  # +1.5 to s_hi each step -> >3 by step 3
    assert last["alarm"] is True
    assert last["direction"] == "up"


def test_cusum_detects_downward_shift_and_reset():
    d = CusumDetector(target=0.0, slack=0.5, threshold=3.0)
    for _ in range(4):
        s = d.update(-2.0)
    assert s["alarm"] is True and s["direction"] == "down"
    d.reset()
    assert d.update(0.0)["alarm"] is False


# ── JLProjector ─────────────────────────────────────────────────────────────
def test_jl_shape_and_determinism():
    p1 = JLProjector(d_in=512, d_out=128, seed=7)
    p2 = JLProjector(d_in=512, d_out=128, seed=7)
    v = np.random.default_rng(0).standard_normal(512)
    out = p1.project(v)
    assert out.shape == (128,)
    assert np.allclose(out, p2.project(v))  # deterministic given seed


def test_jl_different_seed_differs():
    v = np.ones(64)
    a = JLProjector(64, 32, seed=1).project(v)
    b = JLProjector(64, 32, seed=2).project(v)
    assert not np.allclose(a, b)


def test_jl_wrong_dim_raises():
    with pytest.raises(ValueError):
        JLProjector(10, 4).project(np.ones(9))


def test_jl_invalid_dims_raise():
    with pytest.raises(ValueError):
        JLProjector(0, 4)
    with pytest.raises(ValueError):
        JLProjector(4, 0)


def test_jl_preserves_norm_in_expectation():
    # independent seeds for vector vs projector (else the matrix's first row
    # equals the vector and creates an outlier coordinate)
    v = np.random.default_rng(99).standard_normal(512)
    out = JLProjector(512, 256, seed=3).project(v)
    ratio = float(np.linalg.norm(out) / np.linalg.norm(v))
    assert 0.6 < ratio < 1.4   # JL preserves norm in expectation (loose bound)


# ── CircuitCache ────────────────────────────────────────────────────────────
def test_cache_hit_on_matching_fingerprint():
    c = CircuitCache(fingerprint_tol=0.05)
    circuit = [(9, 6), (9, 9), (10, 0)]
    c.put("credit", circuit, [0.1, 0.2, 0.3])
    assert c.get("credit", [0.1, 0.2, 0.3]) == circuit
    assert c.stats()["hits"] == 1


def test_cache_miss_on_drifted_fingerprint():
    c = CircuitCache(fingerprint_tol=0.05)
    c.put("credit", [(9, 6)], [0.1, 0.2, 0.3])
    assert c.get("credit", [5.0, 5.0, 5.0]) is None   # drifted -> rediscover
    assert c.stats()["misses"] == 1


def test_cache_miss_on_unknown_key_and_shape():
    c = CircuitCache()
    assert c.get("nope", [0.0]) is None
    c.put("k", "circ", [0.1, 0.2])
    assert c.get("k", [0.1, 0.2, 0.3]) is None  # shape mismatch
    assert c.stats()["misses"] == 2
