# SPDX-License-Identifier: MIT
"""
tests/test_acdc.py — coverage for the ACDC (Automated Circuit Discovery) data
model: ACDCEdge, ACDCCircuit, ACDCResult.

These dataclasses are pure-Python (no model needed) — repr formatting, edge/
node accounting, density, faithfulness grading, and serialization. The heavy
discover() algorithm is model-dependent and covered separately.
"""

import pytest

from glassbox.acdc import ACDCCircuit, ACDCEdge, ACDCResult

# ---------------------------------------------------------------------------
# discover() — the real ACDC algorithm on a small 2-layer model.
# Marked slow: it loads a model and tests every candidate edge. A 2-layer
# model keeps that to a few dozen edges so the run stays well under a minute.
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDiscover:
    @pytest.fixture(scope="class")
    def small_model(self):
        import sys
        _tl = sys.modules.get("transformer_lens")
        if _tl is not None and not hasattr(_tl, "__file__"):
            pytest.skip("transformer_lens is a test stub, not the real package")
        pytest.importorskip("transformer_lens")
        from transformer_lens import HookedTransformer
        # gelu-2l: 2 layers with MLPs -> a complete (and tiny) ACDC edge graph.
        return HookedTransformer.from_pretrained("gelu-2l")

    def test_discover_returns_valid_result(self, small_model):
        from glassbox.acdc import AutomatedCircuitDiscovery

        clean = small_model.to_tokens("The cat sat on the mat and looked around")
        corrupted = clean.clone()
        # swap one interior token -> same length, a valid corruption for patching
        corrupted[0, 3] = clean[0, 1]

        acdc = AutomatedCircuitDiscovery(small_model, threshold=0.5)
        result = acdc.discover(clean, corrupted)

        assert isinstance(result, ACDCResult)
        assert isinstance(result.kl_circuit, float)
        assert result.n_edges_total > 0
        assert result.circuit.n_edges() <= result.n_edges_total
        assert result.faithfulness_grade() in ("STRONG", "PARTIAL", "WEAK")
        assert "ACDC" in result.summary()
        assert "kl_circuit" in result.to_dict()


# ---------------------------------------------------------------------------
# ACDCEdge
# ---------------------------------------------------------------------------

class TestACDCEdge:
    def test_repr_attn_to_attn(self):
        e = ACDCEdge(sender=(0, "attn", 3), receiver=(2, "attn", 7))
        assert repr(e) == "L0AH3→L2AH7"

    def test_repr_mlp_to_attn(self):
        e = ACDCEdge(sender=(0, "mlp", 0), receiver=(1, "attn", 2))
        assert repr(e) == "L0MLP→L1AH2"

    def test_repr_attn_to_mlp(self):
        e = ACDCEdge(sender=(1, "attn", 5), receiver=(3, "mlp", 0))
        assert repr(e) == "L1AH5→L3MLP"

    def test_frozen_hashable(self):
        # frozen dataclass -> usable in a set
        e1 = ACDCEdge(sender=(0, "attn", 1), receiver=(1, "attn", 2))
        e2 = ACDCEdge(sender=(0, "attn", 1), receiver=(1, "attn", 2))
        assert len({e1, e2}) == 1


# ---------------------------------------------------------------------------
# ACDCCircuit
# ---------------------------------------------------------------------------

@pytest.fixture
def circuit():
    edges = {
        ACDCEdge(sender=(0, "attn", 3), receiver=(2, "attn", 7)),
        ACDCEdge(sender=(0, "mlp", 0), receiver=(1, "attn", 2)),
    }
    return ACDCCircuit(edges=edges, n_layers=12, n_heads=12)


class TestACDCCircuit:
    def test_n_edges(self, circuit):
        assert circuit.n_edges() == 2

    def test_head_nodes(self, circuit):
        # attn senders/receivers only; the mlp sender is excluded
        assert circuit.head_nodes() == {(0, 3), (2, 7), (1, 2)}

    def test_to_head_list_sorted(self, circuit):
        assert circuit.to_head_list() == [(0, 3), (1, 2), (2, 7)]

    def test_density_is_fraction(self, circuit):
        d = circuit.density()
        assert isinstance(d, float)
        assert 0.0 <= d <= 1.0

    def test_density_empty_circuit_is_zero(self):
        c = ACDCCircuit(edges=set(), n_layers=12, n_heads=12)
        assert c.density() == 0.0

    def test_density_single_layer_no_edges_possible(self):
        # n_layers=1 -> the density loop never runs -> max_edges 0 -> 0.0
        c = ACDCCircuit(edges=set(), n_layers=1, n_heads=4)
        assert c.density() == 0.0


# ---------------------------------------------------------------------------
# ACDCResult
# ---------------------------------------------------------------------------

def _result(kl, circuit):
    return ACDCResult(
        circuit=circuit,
        kl_circuit=kl,
        n_edges_total=100,
        n_edges_retained=circuit.n_edges(),
        n_edges_pruned=100 - circuit.n_edges(),
        threshold=0.10,
        faithful=(kl < 0.10),
        pruning_kl_scores={},
    )


class TestACDCResult:
    @pytest.mark.parametrize("kl,grade", [
        (0.50, "STRONG"),
        (1.20, "PARTIAL"),
        (2.00, "WEAK"),
    ])
    def test_faithfulness_grade(self, kl, grade, circuit):
        assert _result(kl, circuit).faithfulness_grade() == grade

    def test_summary_contains_key_fields(self, circuit):
        s = _result(0.5, circuit).summary()
        assert "ACDC" in s and "KL_circuit" in s and "STRONG" in s

    def test_to_dict_keys(self, circuit):
        d = _result(0.5, circuit).to_dict()
        for key in ("n_edges_circuit", "n_edges_total", "circuit_density",
                    "kl_circuit", "faithful", "faithfulness_grade", "head_list"):
            assert key in d
        assert d["n_edges_circuit"] == 2
        assert d["head_list"] == [(0, 3), (1, 2), (2, 7)]
