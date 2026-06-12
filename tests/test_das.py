# SPDX-License-Identifier: MIT
"""
tests/test_das.py — coverage for Distributed Alignment Search.

The DASResult dataclass (serialization, summary formatting) is pure-Python and
verified offline. The search() algorithm — PCA on activation differences plus
interchange interventions — is model-dependent and run on a small model.
"""

import numpy as np
import pytest

from glassbox.das import DASResult

# ---------------------------------------------------------------------------
# DASResult dataclass
# ---------------------------------------------------------------------------

def _das_result(encoded=True, score=0.82):
    return DASResult(
        concept_label="io_name_position",
        target_layer=9,
        target_position=-1,
        das_score=score,
        rotation_matrix=np.zeros((512, 4)),
        concept_dims=4,
        explained_variance=0.731,
        mean_ld_clean=3.10,
        mean_ld_intervened=0.50,
        concept_encoded=encoded,
        n_samples=20,
        pca_eigenvalues=[5.0, 3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
    )


class TestDASResult:
    def test_to_dict_keys_and_rounding(self):
        d = _das_result().to_dict()
        for key in ("concept_label", "target_layer", "das_score", "concept_dims",
                    "explained_variance", "concept_encoded", "n_samples",
                    "das_threshold", "top_eigenvalues", "rotation_shape"):
            assert key in d
        assert d["das_score"] == 0.82
        assert d["rotation_shape"] == [512, 4]

    def test_to_dict_eigenvalues_capped_at_10(self):
        d = _das_result().to_dict()
        assert len(d["top_eigenvalues"]) == 10  # 11 provided, capped at 10

    def test_summary_line_encoded(self):
        line = _das_result(encoded=True).summary_line()
        assert "ENCODED" in line and "io_name_position" in line
        assert "layer=9" in line

    def test_summary_line_not_found(self):
        line = _das_result(encoded=False).summary_line()
        assert "not found" in line


# ---------------------------------------------------------------------------
# search() — the DAS algorithm on a small 2-layer model (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSearch:
    @pytest.fixture(scope="class")
    def small_model(self):
        import sys
        _tl = sys.modules.get("transformer_lens")
        if _tl is not None and not hasattr(_tl, "__file__"):
            pytest.skip("transformer_lens is a test stub, not the real package")
        pytest.importorskip("transformer_lens")
        from transformer_lens import HookedTransformer
        return HookedTransformer.from_pretrained("gelu-2l")

    def test_search_returns_result(self, small_model):
        from glassbox.das import DistributedAlignmentSearch

        texts = [
            "The cat sat on the mat and looked",
            "A dog ran across the wide green",
            "She walked into the quiet old",
            "They opened the heavy wooden",
            "He picked up the small red",
            "We found the lost golden",
        ]
        clean = [small_model.to_tokens(t) for t in texts]
        counterfactual = []
        for tok in clean:
            cf = tok.clone()
            cf[0, 2] = tok[0, 1]  # swap one token, same length
            counterfactual.append(cf)

        target_tok = small_model.to_single_token(" the")
        distract_tok = small_model.to_single_token(" a")

        das = DistributedAlignmentSearch(small_model, concept_dims=2, n_interchange=4)
        result = das.search(
            concept_label="test_concept",
            clean_prompts_tokens=clean,
            counterfactual_tokens=counterfactual,
            target_tok=target_tok,
            distract_tok=distract_tok,
            target_layer=1,
            target_position=-1,
        )

        assert isinstance(result, DASResult)
        assert isinstance(result.das_score, float)
        assert result.concept_dims == 2
        assert result.rotation_matrix.shape[1] == 2
        assert isinstance(result.concept_encoded, bool)
        assert "test_concept" in result.summary_line()
        assert "das_score" in result.to_dict()
