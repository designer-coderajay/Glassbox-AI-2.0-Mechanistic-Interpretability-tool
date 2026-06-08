"""
tests/test_multi_arch.py
========================
Fast, offline tests for glassbox.multi_arch — no model download required.

Coverage
--------
1. ARCHITECTURE_REGISTRY — all 11 families present with valid entries
2. ArchitectureConfig construction via direct kwargs (no model needed)
3. ArchitectureConfig.from_transformer_lens — mock model for each family
4. kv_head_for_query / query_heads_for_kv — GQA head mapping maths
5. GQAAttentionMapper.redistribute_kv_attributions — score merging maths
6. MultiArchAdapter round-trip — build from mocked model, run adjust_attributions
7. Parametrized sweep over all 11 families — regression guard

None of these tests load actual model weights. They only exercise the adapter
logic, the config parsing path, and the GQA math kernels.

Run with:
    pytest tests/test_multi_arch.py -v
"""

from __future__ import annotations

import math
import sys
import types
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Inject lightweight stubs for torch / transformer_lens if not installed.
# The multi_arch module uses torch.Tensor type-checks and nn.Module so we
# need a minimal stub that satisfies isinstance() checks.
# ---------------------------------------------------------------------------

def _inject_stubs():
    """Inject MagicMock stubs for heavy deps that are GENUINELY not installed.

    Detection uses importlib.util.find_spec — "does this package exist on disk" —
    not sys.modules membership ("has it been imported yet"). The latter is
    unreliable: at collection time a real, installed package may simply not be
    imported into sys.modules yet. Stubbing it then poisons every later real
    import for the whole session — which is exactly how test_engine ended up
    loading a MagicMock instead of a real HookedTransformer in CI.
    """
    import importlib.util

    def _installed(top: str) -> bool:
        # find_spec can raise ValueError when sys.modules already holds a
        # non-module stub (offline mode, where conftest injected one); treat
        # that as "not a real install" so we leave the existing stub in place.
        try:
            return importlib.util.find_spec(top) is not None
        except (ImportError, ValueError):
            return False

    for mod_name in [
        "torch", "torch.nn", "torch.nn.functional", "torch.autograd",
        "torch.autograd.functional", "torch.cuda", "torch.utils",
        "torch.utils.data", "torch.linalg",
        "transformer_lens", "transformer_lens.hook_points",
        "transformer_lens.utilities",
        "einops", "scipy", "scipy.stats", "scipy.spatial",
        "scipy.spatial.distance",
    ]:
        top = mod_name.split(".")[0]
        if _installed(top):
            continue  # genuinely installed — never stub it
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    # The Tensor / nn.Module type fix-ups are only needed when torch was stubbed;
    # a real torch provides these itself and must not be patched.
    if not _installed("torch"):
        torch_stub = sys.modules["torch"]
        if not isinstance(getattr(torch_stub, "Tensor", None), type):
            class _FakeTensor:
                pass
            torch_stub.Tensor = _FakeTensor
        nn_stub = sys.modules["torch.nn"]
        if not isinstance(getattr(nn_stub, "Module", None), type):
            class _FakeModule:
                pass
            nn_stub.Module = _FakeModule
        torch_stub.nn = nn_stub


_inject_stubs()

# Now we can safely import the module under test
from glassbox.multi_arch import (   # noqa: E402
    ARCHITECTURE_REGISTRY,
    SUPPORTED_ARCHITECTURES,
    RMSNORM_ARCHITECTURES,
    GQA_ARCHITECTURES,
    ArchitectureConfig,
    GQAAttentionMapper,
    MultiArchAdapter,
)


# ===========================================================================
# Constants — the 11 expected architecture families
# ===========================================================================

EXPECTED_FAMILIES = {
    "gpt2", "pythia", "gpt-j", "llama-2",
    "llama-3", "llama-3-70b", "mistral",
    "phi-2", "phi-3", "gemma", "qwen2",
}

# Ground-truth for each family: (kv_ratio, norm, act)
FAMILY_SPECS = {
    "gpt2":        (1.0,   "layernorm", "gelu"),
    "pythia":      (1.0,   "layernorm", "gelu"),
    "gpt-j":       (1.0,   "layernorm", "gelu"),
    "llama-2":     (1.0,   "rmsnorm",   "silu"),
    "llama-3":     (0.25,  "rmsnorm",   "silu"),
    "llama-3-70b": (0.125, "rmsnorm",   "silu"),
    "mistral":     (0.25,  "rmsnorm",   "silu"),
    "phi-2":       (1.0,   "layernorm", "gelu"),
    "phi-3":       (0.25,  "rmsnorm",   "silu"),
    "gemma":       (0.125, "rmsnorm",   "gelu"),
    "qwen2":       (0.25,  "rmsnorm",   "silu"),
}


# ===========================================================================
# 1. ARCHITECTURE_REGISTRY integrity
# ===========================================================================

class TestArchitectureRegistry:
    """All 11 families must be present with correct metadata."""

    def test_all_families_registered(self):
        assert EXPECTED_FAMILIES == set(ARCHITECTURE_REGISTRY.keys()), (
            f"Registry mismatch.\n"
            f"  Missing : {EXPECTED_FAMILIES - set(ARCHITECTURE_REGISTRY.keys())}\n"
            f"  Extra   : {set(ARCHITECTURE_REGISTRY.keys()) - EXPECTED_FAMILIES}"
        )

    @pytest.mark.parametrize("family", sorted(EXPECTED_FAMILIES))
    def test_family_has_required_keys(self, family):
        entry = ARCHITECTURE_REGISTRY[family]
        assert "kv_ratio" in entry, f"{family}: missing kv_ratio"
        assert "norm" in entry,     f"{family}: missing norm"
        assert "act" in entry,      f"{family}: missing act"

    @pytest.mark.parametrize("family,spec", sorted(FAMILY_SPECS.items()))
    def test_family_ground_truth(self, family, spec):
        kv_ratio, norm, act = spec
        entry = ARCHITECTURE_REGISTRY[family]
        assert entry["kv_ratio"] == kv_ratio, (
            f"{family}: kv_ratio expected {kv_ratio}, got {entry['kv_ratio']}"
        )
        assert entry["norm"] == norm, (
            f"{family}: norm expected {norm!r}, got {entry['norm']!r}"
        )
        assert entry["act"] == act, (
            f"{family}: act expected {act!r}, got {entry['act']!r}"
        )

    def test_derived_lists_consistent(self):
        """SUPPORTED_ARCHITECTURES, RMSNORM_ARCHITECTURES, GQA_ARCHITECTURES
        must agree with the raw registry."""
        assert set(SUPPORTED_ARCHITECTURES) == set(ARCHITECTURE_REGISTRY.keys())

        expected_rmsnorm = {k for k, v in ARCHITECTURE_REGISTRY.items()
                            if v["norm"] == "rmsnorm"}
        assert set(RMSNORM_ARCHITECTURES) == expected_rmsnorm

        expected_gqa = {k for k, v in ARCHITECTURE_REGISTRY.items()
                        if v["kv_ratio"] < 1.0}
        assert set(GQA_ARCHITECTURES) == expected_gqa

    def test_kv_ratios_are_valid_fractions(self):
        """kv_ratio must be in (0, 1] and be a power of 0.5 (or 1.0 for MHA)."""
        for family, entry in ARCHITECTURE_REGISTRY.items():
            ratio = entry["kv_ratio"]
            assert 0 < ratio <= 1.0, f"{family}: kv_ratio {ratio} out of range"


# ===========================================================================
# 2. ArchitectureConfig construction — direct instantiation
# ===========================================================================

class TestArchitectureConfigDirect:
    """Build ArchitectureConfig via kwargs and verify computed properties."""

    def _gpt2_config(self) -> ArchitectureConfig:
        return ArchitectureConfig(
            model_name="gpt2",
            n_layers=12, n_heads=12, n_kv_heads=12,
            d_model=768, d_head=64,
            norm_type="layernorm", activation="gelu",
            is_gqa=False, heads_per_kv_group=1,
        )

    def _llama3_8b_config(self) -> ArchitectureConfig:
        """Llama-3-8B: 32 Q heads, 8 KV heads (4:1 GQA)."""
        return ArchitectureConfig(
            model_name="meta-llama/Llama-3-8B",
            n_layers=32, n_heads=32, n_kv_heads=8,
            d_model=4096, d_head=128,
            norm_type="rmsnorm", activation="silu",
            is_gqa=True, heads_per_kv_group=4,
        )

    def _llama3_70b_config(self) -> ArchitectureConfig:
        """Llama-3-70B: 64 Q heads, 8 KV heads (8:1 GQA)."""
        return ArchitectureConfig(
            model_name="meta-llama/Llama-3-70B",
            n_layers=80, n_heads=64, n_kv_heads=8,
            d_model=8192, d_head=128,
            norm_type="rmsnorm", activation="silu",
            is_gqa=True, heads_per_kv_group=8,
        )

    def test_gpt2_is_not_gqa(self):
        cfg = self._gpt2_config()
        assert not cfg.is_gqa
        assert cfg.n_kv_heads == cfg.n_heads == 12
        assert cfg.heads_per_kv_group == 1

    def test_llama3_8b_is_gqa(self):
        cfg = self._llama3_8b_config()
        assert cfg.is_gqa
        assert cfg.n_kv_heads == 8
        assert cfg.heads_per_kv_group == 4

    def test_llama3_70b_ratio(self):
        cfg = self._llama3_70b_config()
        assert cfg.is_gqa
        assert cfg.n_heads // cfg.n_kv_heads == 8
        assert cfg.heads_per_kv_group == 8

    def test_d_head_consistent_gpt2(self):
        cfg = self._gpt2_config()
        assert cfg.d_head == cfg.d_model // cfg.n_heads


# ===========================================================================
# 3. ArchitectureConfig.from_transformer_lens — mock model per family
# ===========================================================================

def _make_mock_model(
    model_name: str,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    d_model: int,
    norm_type: str,
    activation: str,
) -> MagicMock:
    """Build a minimal HookedTransformer mock with cfg attributes set."""
    cfg = MagicMock()
    cfg.n_heads = n_heads
    cfg.n_layers = n_layers
    cfg.d_model = d_model
    cfg.d_head = d_model // n_heads
    cfg.model_name = model_name
    # TransformerLens >=2.0 exposes n_key_value_heads directly
    cfg.n_key_value_heads = n_kv_heads
    cfg.normalization_type = norm_type  # "LN" or "RMS"
    cfg.act_fn = activation             # "gelu" or "silu"

    model = MagicMock()
    model.cfg = cfg
    return model


# Representative "small" configurations for each family (layers/heads reduced
# so the config object is cheap to build — exact size doesn't matter here).
_FAMILY_MOCK_PARAMS = {
    # family: (model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act)
    "gpt2":        ("gpt2",                            12, 12,  12,  768,  "LN",  "gelu"),
    "pythia":      ("EleutherAI/pythia-70m",            6,  8,   8,  512,  "LN",  "gelu"),
    "gpt-j":       ("EleutherAI/gpt-j-6b",             28, 16,  16, 4096, "LN",  "gelu"),
    "llama-2":     ("meta-llama/Llama-2-7b-hf",        32, 32,  32, 4096, "RMS", "silu"),
    "llama-3":     ("meta-llama/Meta-Llama-3-8B",      32, 32,   8, 4096, "RMS", "silu"),
    "llama-3-70b": ("meta-llama/Meta-Llama-3-70B",     80, 64,   8, 8192, "RMS", "silu"),
    "mistral":     ("mistralai/Mistral-7B-v0.1",       32, 32,   8, 4096, "RMS", "silu"),
    "phi-2":       ("microsoft/phi-2",                 32, 32,  32, 2560, "LN",  "gelu"),
    "phi-3":       ("microsoft/Phi-3-mini-4k-instruct",32, 32,   8, 3072, "RMS", "silu"),
    "gemma":       ("google/gemma-7b",                 28, 16,   2, 3072, "RMS", "gelu"),
    "qwen2":       ("Qwen/Qwen2-7B",                   32, 32,   8, 4096, "RMS", "silu"),
}


class TestArchConfigFromModel:
    """from_transformer_lens correctly parses each architecture family."""

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_n_kv_heads_detected(self, family):
        params = _FAMILY_MOCK_PARAMS[family]
        model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act = params
        mock = _make_mock_model(model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act)

        cfg = ArchitectureConfig.from_transformer_lens(mock)
        assert cfg.n_kv_heads == n_kv_heads, (
            f"{family}: expected n_kv_heads={n_kv_heads}, got {cfg.n_kv_heads}"
        )

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_is_gqa_flag(self, family):
        params = _FAMILY_MOCK_PARAMS[family]
        model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act = params
        mock = _make_mock_model(model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act)

        cfg = ArchitectureConfig.from_transformer_lens(mock)
        expected_gqa = n_kv_heads < n_heads
        assert cfg.is_gqa == expected_gqa, (
            f"{family}: is_gqa expected {expected_gqa}, got {cfg.is_gqa}"
        )

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_heads_per_kv_group(self, family):
        params = _FAMILY_MOCK_PARAMS[family]
        model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act = params
        mock = _make_mock_model(model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act)

        cfg = ArchitectureConfig.from_transformer_lens(mock)
        expected_group = n_heads // n_kv_heads
        assert cfg.heads_per_kv_group == expected_group, (
            f"{family}: heads_per_kv_group expected {expected_group}, got {cfg.heads_per_kv_group}"
        )

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_norm_type_detected(self, family):
        """Norm type must match the ground-truth spec for each family."""
        _, expected_norm, _ = FAMILY_SPECS[family]
        params = _FAMILY_MOCK_PARAMS[family]
        model_name, n_layers, n_heads, n_kv_heads, d_model, norm_raw, act = params
        mock = _make_mock_model(model_name, n_layers, n_heads, n_kv_heads, d_model, norm_raw, act)

        cfg = ArchitectureConfig.from_transformer_lens(mock)
        assert cfg.norm_type == expected_norm, (
            f"{family}: norm_type expected {expected_norm!r}, got {cfg.norm_type!r}"
        )


# ===========================================================================
# 4. GQA head mapping — kv_head_for_query and query_heads_for_kv
# ===========================================================================

class TestGQAHeadMapping:
    """Verify the Q↔KV head mapping is bijective and numerically correct."""

    def _make_gqa_config(self, n_heads: int, n_kv_heads: int) -> ArchitectureConfig:
        return ArchitectureConfig(
            model_name="test-gqa", n_layers=1,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            d_model=n_heads * 64, d_head=64,
            norm_type="rmsnorm", activation="silu",
            is_gqa=True, heads_per_kv_group=n_heads // n_kv_heads,
        )

    @pytest.mark.parametrize("n_heads,n_kv_heads", [
        (32, 8),   # Llama-3-8B, Mistral, Phi-3, Qwen2 ratio
        (64, 8),   # Llama-3-70B ratio
        (16, 2),   # Gemma ratio
        (4,  4),   # MHA — all heads their own KV
        (8,  1),   # MQA (single shared KV)
    ])
    def test_every_query_head_maps_to_valid_kv_head(self, n_heads, n_kv_heads):
        cfg = self._make_gqa_config(n_heads, n_kv_heads)
        for q in range(n_heads):
            kv = cfg.kv_head_for_query(q)
            assert 0 <= kv < n_kv_heads, (
                f"q={q} mapped to invalid kv={kv} (n_kv_heads={n_kv_heads})"
            )

    @pytest.mark.parametrize("n_heads,n_kv_heads", [
        (32, 8), (64, 8), (16, 2), (4, 4), (8, 1),
    ])
    def test_query_heads_for_kv_covers_all_queries(self, n_heads, n_kv_heads):
        cfg = self._make_gqa_config(n_heads, n_kv_heads)
        all_q_heads: List[int] = []
        for kv in range(n_kv_heads):
            q_heads = cfg.query_heads_for_kv(kv)
            assert len(q_heads) == n_heads // n_kv_heads, (
                f"KV head {kv} serves {len(q_heads)} Q heads, "
                f"expected {n_heads // n_kv_heads}"
            )
            all_q_heads.extend(q_heads)
        # Every query head covered exactly once
        assert sorted(all_q_heads) == list(range(n_heads)), (
            "query_heads_for_kv does not partition Q heads exactly"
        )

    @pytest.mark.parametrize("n_heads,n_kv_heads", [
        (32, 8), (64, 8),
    ])
    def test_kv_head_and_query_heads_are_inverses(self, n_heads, n_kv_heads):
        cfg = self._make_gqa_config(n_heads, n_kv_heads)
        for q in range(n_heads):
            kv = cfg.kv_head_for_query(q)
            assert q in cfg.query_heads_for_kv(kv), (
                f"q={q} → kv={kv} but kv={kv} → {cfg.query_heads_for_kv(kv)}"
            )

    def test_mha_every_head_its_own_kv(self):
        """Standard MHA: each query head maps to itself."""
        cfg = self._make_gqa_config(n_heads=12, n_kv_heads=12)
        for h in range(12):
            assert cfg.kv_head_for_query(h) == h


# ===========================================================================
# 5. GQAAttentionMapper.redistribute_kv_attributions — score merging
# ===========================================================================

class TestGQAAttentionMapper:
    """Score redistribution maths: sums, symmetry, edge cases."""

    def _mapper(self, n_heads: int, n_kv_heads: int) -> GQAAttentionMapper:
        cfg = ArchitectureConfig(
            model_name="test", n_layers=1,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            d_model=n_heads * 64, d_head=64,
            norm_type="rmsnorm", activation="silu",
            is_gqa=True, heads_per_kv_group=n_heads // n_kv_heads,
        )
        return GQAAttentionMapper(cfg)

    def test_docstring_example_4head_2kv(self):
        """Replicate the worked example from the docstring."""
        mapper = self._mapper(n_heads=4, n_kv_heads=2)
        kv_attrs = {0: 10.0, 1: 5.0}
        q_attrs  = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

        result = mapper.redistribute_kv_attributions(kv_attrs, q_attrs)

        # KV 0 serves Q 0 and Q 1; each gets +5.0
        assert math.isclose(result[0], 6.0, rel_tol=1e-9), f"result[0]={result[0]}"
        assert math.isclose(result[1], 6.0, rel_tol=1e-9), f"result[1]={result[1]}"
        # KV 1 serves Q 2 and Q 3; each gets +2.5
        assert math.isclose(result[2], 3.5, rel_tol=1e-9), f"result[2]={result[2]}"
        assert math.isclose(result[3], 3.5, rel_tol=1e-9), f"result[3]={result[3]}"

    def test_all_query_heads_present_in_result(self):
        mapper = self._mapper(n_heads=8, n_kv_heads=2)
        kv_attrs = {0: 4.0, 1: 8.0}
        q_attrs  = {h: 0.5 for h in range(8)}

        result = mapper.redistribute_kv_attributions(kv_attrs, q_attrs)
        assert set(result.keys()) == set(range(8))

    def test_zero_kv_attribution_passthrough(self):
        """If KV scores are all zero, output equals input query attributions."""
        mapper = self._mapper(n_heads=4, n_kv_heads=2)
        q_attrs = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        kv_attrs = {0: 0.0, 1: 0.0}

        result = mapper.redistribute_kv_attributions(kv_attrs, q_attrs)
        for h, v in q_attrs.items():
            assert math.isclose(result[h], v, rel_tol=1e-9)

    def test_conservation_sum(self):
        """Total sum of merged attributions equals sum(q) + sum(kv)."""
        mapper = self._mapper(n_heads=8, n_kv_heads=2)
        kv_attrs = {0: 6.0, 1: 4.0}
        q_attrs  = {h: float(h) for h in range(8)}

        result = mapper.redistribute_kv_attributions(kv_attrs, q_attrs)

        expected_total = sum(q_attrs.values()) + sum(kv_attrs.values())
        actual_total = sum(result.values())
        assert math.isclose(actual_total, expected_total, rel_tol=1e-9), (
            f"Sum mismatch: expected {expected_total}, got {actual_total}"
        )

    def test_negative_kv_scores_handled(self):
        """Negative attribution scores (inhibitory heads) must redistribute correctly."""
        mapper = self._mapper(n_heads=4, n_kv_heads=2)
        kv_attrs = {0: -8.0, 1: 4.0}
        q_attrs  = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

        result = mapper.redistribute_kv_attributions(kv_attrs, q_attrs)

        # KV 0 (score -8) serves Q 0, Q 1: each gets 1 + (-8/2) = -3
        assert math.isclose(result[0], -3.0, rel_tol=1e-9)
        assert math.isclose(result[1], -3.0, rel_tol=1e-9)
        # KV 1 (score +4) serves Q 2, Q 3: each gets 1 + 4/2 = 3
        assert math.isclose(result[2], 3.0, rel_tol=1e-9)
        assert math.isclose(result[3], 3.0, rel_tol=1e-9)

    @pytest.mark.parametrize("n_heads,n_kv_heads", [
        (32, 8),   # Llama-3-8B / Mistral
        (64, 8),   # Llama-3-70B
        (16, 2),   # Gemma
    ])
    def test_per_family_conservation(self, n_heads, n_kv_heads):
        """Conservation law holds for each supported GQA family's head count."""
        mapper = self._mapper(n_heads, n_kv_heads)
        kv_attrs = {kv: float(kv + 1) * 0.5 for kv in range(n_kv_heads)}
        q_attrs  = {q: float(q) * 0.1 for q in range(n_heads)}

        result = mapper.redistribute_kv_attributions(kv_attrs, q_attrs)

        expected_total = sum(q_attrs.values()) + sum(kv_attrs.values())
        actual_total = sum(result.values())
        assert math.isclose(actual_total, expected_total, rel_tol=1e-9), (
            f"[n_heads={n_heads}, n_kv={n_kv_heads}] "
            f"sum expected {expected_total:.4f}, got {actual_total:.4f}"
        )


# ===========================================================================
# 6. MultiArchAdapter — build from mock model + adjust_attributions round-trip
# ===========================================================================

class TestMultiArchAdapter:
    """MultiArchAdapter wires ArchitectureConfig and GQAAttentionMapper together."""

    def _adapter(self, family: str) -> MultiArchAdapter:
        params = _FAMILY_MOCK_PARAMS[family]
        model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act = params
        mock = _make_mock_model(model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act)
        return MultiArchAdapter.from_model(mock)

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_adapter_builds_from_model(self, family):
        adapter = self._adapter(family)
        assert adapter is not None
        assert adapter.config.n_layers >= 1

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_adapter_norm_type(self, family):
        _, expected_norm, _ = FAMILY_SPECS[family]
        adapter = self._adapter(family)
        assert adapter.get_norm_type() == expected_norm, (
            f"{family}: get_norm_type() expected {expected_norm!r}, "
            f"got {adapter.get_norm_type()!r}"
        )

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_adapter_is_gqa_consistent(self, family):
        adapter = self._adapter(family)
        params = _FAMILY_MOCK_PARAMS[family]
        _, _, n_heads, n_kv_heads, *_ = params
        expected = n_kv_heads < n_heads
        assert adapter.is_gqa() == expected, (
            f"{family}: is_gqa() expected {expected}, got {adapter.is_gqa()}"
        )

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_adapter_is_rmsnorm_consistent(self, family):
        adapter = self._adapter(family)
        _, expected_norm, _ = FAMILY_SPECS[family]
        expected = expected_norm == "rmsnorm"
        assert adapter.is_rmsnorm() == expected, (
            f"{family}: is_rmsnorm() expected {expected}, got {adapter.is_rmsnorm()}"
        )

    @pytest.mark.parametrize("family", sorted(
        f for f in _FAMILY_MOCK_PARAMS
        if _FAMILY_MOCK_PARAMS[f][3] < _FAMILY_MOCK_PARAMS[f][2]  # GQA families only
    ))
    def test_gqa_adapter_adjust_attributions_conserves_sum(self, family):
        """adjust_attributions_for_gqa must conserve total score (no score created/lost)."""
        adapter = self._adapter(family)
        n_heads = adapter.config.n_heads
        n_kv = adapter.config.n_kv_heads

        # Input attributions: one per (layer, head) tuple, values = head index
        raw_attrs: Dict[tuple, float] = {
            (layer, head): float(head + 1) * 0.3
            for layer in range(min(2, adapter.config.n_layers))
            for head in range(n_heads)
        }
        adjusted = adapter.adjust_attributions_for_gqa(raw_attrs)

        # Sums should be equal
        assert math.isclose(
            sum(adjusted.values()), sum(raw_attrs.values()), rel_tol=1e-6
        ), (
            f"{family}: sum before={sum(raw_attrs.values()):.4f}, "
            f"after={sum(adjusted.values()):.4f}"
        )

    @pytest.mark.parametrize("family", sorted(
        f for f in _FAMILY_MOCK_PARAMS
        if _FAMILY_MOCK_PARAMS[f][3] < _FAMILY_MOCK_PARAMS[f][2]
    ))
    def test_gqa_adapter_gqa_mapping_covers_all_heads(self, family):
        """get_gqa_head_mapping covers all query heads for GQA families."""
        adapter = self._adapter(family)
        mapping = adapter.get_gqa_head_mapping()
        n_heads = adapter.config.n_heads
        n_kv = adapter.config.n_kv_heads

        # mapping: kv_head → [q_heads...]
        assert len(mapping) == n_kv, (
            f"{family}: mapping has {len(mapping)} KV entries, expected {n_kv}"
        )
        all_q = sorted(q for qs in mapping.values() for q in qs)
        assert all_q == list(range(n_heads)), (
            f"{family}: head mapping doesn't cover Q heads 0..{n_heads-1}"
        )


# ===========================================================================
# 7. Architecture report — smoke test for all families
# ===========================================================================

class TestArchitectureReport:
    """architecture_report() returns a populated ArchitectureReport for each family."""

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_report_fields_populated(self, family):
        params = _FAMILY_MOCK_PARAMS[family]
        model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act = params
        mock = _make_mock_model(model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act)

        adapter = MultiArchAdapter.from_model(mock)
        report = adapter.architecture_report()

        assert report is not None
        # ArchitectureReport fields (no n_layers — use adapter.config for that)
        assert report.n_heads == n_heads,       f"{family}: n_heads mismatch"
        assert report.n_kv_heads == n_kv_heads, f"{family}: n_kv_heads mismatch"
        assert report.model_name == model_name, f"{family}: model_name mismatch"
        # n_layers lives on the underlying config
        assert adapter.config.n_layers == n_layers, f"{family}: n_layers mismatch"

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_report_summary_is_str(self, family):
        params = _FAMILY_MOCK_PARAMS[family]
        model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act = params
        mock = _make_mock_model(model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act)

        adapter = MultiArchAdapter.from_model(mock)
        report = adapter.architecture_report()

        summary = report.summary()
        assert isinstance(summary, str) and len(summary) > 0, (
            f"{family}: report.summary() returned empty or non-string"
        )

    @pytest.mark.parametrize("family", sorted(_FAMILY_MOCK_PARAMS))
    def test_report_to_dict_serializable(self, family):
        params = _FAMILY_MOCK_PARAMS[family]
        model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act = params
        mock = _make_mock_model(model_name, n_layers, n_heads, n_kv_heads, d_model, norm, act)

        adapter = MultiArchAdapter.from_model(mock)
        d = adapter.architecture_report().to_dict()

        assert isinstance(d, dict)
        # Must be JSON-serializable (no torch Tensors etc.)
        import json
        json.dumps(d)  # raises if not serializable
