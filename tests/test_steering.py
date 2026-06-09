# SPDX-License-Identifier: MIT
"""
tests/test_steering.py — coverage for activation steering.

Steering vectors are torch tensors, so these tests need real torch (they run in
the full CI job, and skip cleanly when torch is a stub). The dataclass, the
mean-diff / PCA direction helpers, exporter construction, and the export/load
round-trips need no model; extract + apply run on a small model.
"""

import pytest


@pytest.fixture(autouse=True)
def _real_torch():
    import sys
    t = sys.modules.get("torch")
    if t is not None and not hasattr(t, "__file__"):
        pytest.skip("torch is a test stub, not the real package")
    pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# SteeringVector dataclass + direction helpers (torch, no model)
# ---------------------------------------------------------------------------

class TestSteeringVector:
    def test_repr_and_norm(self):
        import torch
        from glassbox.steering import SteeringVector
        v = SteeringVector(direction=torch.randn(16), layer=2, concept_label="gender_bias")
        r = repr(v)
        assert "gender_bias" in r and "layer=2" in r
        assert isinstance(v.norm(), float)

    def test_to_dict(self):
        import torch
        from glassbox.steering import SteeringVector
        v = SteeringVector(direction=torch.randn(16), layer=3, concept_label="x", scale=-10.0)
        d = v.to_dict()
        for key in ("concept_label", "layer", "scale", "d_model", "norm", "source_info"):
            assert key in d
        assert d["layer"] == 3 and d["d_model"] == 16


class TestDirectionHelpers:
    def test_mean_diff_is_unit_norm(self):
        import torch
        from glassbox.steering import _mean_diff_direction
        pos = torch.randn(4, 8)
        neg = torch.randn(4, 8)
        d = _mean_diff_direction(pos, neg)
        assert d.shape == (8,)
        assert abs(float(d.norm()) - 1.0) < 1e-4

    def test_pca_is_unit_norm(self):
        import torch
        from glassbox.steering import _pca_direction
        pos = torch.randn(5, 8)
        neg = torch.randn(5, 8)
        d = _pca_direction(pos, neg)
        assert d.shape == (8,)
        assert abs(float(d.norm()) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Exporter construction + serialization (torch, no model)
# ---------------------------------------------------------------------------

class TestExporterConstruction:
    def test_valid_methods(self):
        from glassbox.steering import SteeringVectorExporter
        assert SteeringVectorExporter(method="mean_diff").method == "mean_diff"
        assert SteeringVectorExporter(method="pca").method == "pca"

    def test_invalid_method_raises(self):
        from glassbox.steering import SteeringVectorExporter
        with pytest.raises(ValueError):
            SteeringVectorExporter(method="nonsense")


class TestSerialization:
    def test_pt_round_trip(self, tmp_path):
        import torch
        from glassbox.steering import SteeringVector, SteeringVectorExporter
        v = SteeringVector(direction=torch.randn(16), layer=4, concept_label="tox", scale=-12.0)
        ex = SteeringVectorExporter()
        path = str(tmp_path / "v.pt")
        ex.export_pt(v, path)
        loaded = SteeringVectorExporter.load_pt(path)
        assert loaded.layer == 4 and loaded.concept_label == "tox" and loaded.scale == -12.0

    def test_numpy_export_writes_files(self, tmp_path):
        import os
        import torch
        from glassbox.steering import SteeringVector, SteeringVectorExporter
        v = SteeringVector(direction=torch.randn(16), layer=1, concept_label="c")
        path = str(tmp_path / "v.npy")
        SteeringVectorExporter().export_numpy(v, path)
        assert os.path.exists(path)
        assert os.path.exists(str(tmp_path / "v_meta.json"))


# ---------------------------------------------------------------------------
# extract + apply on a small model (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestExtractApply:
    @pytest.fixture(scope="class")
    def small_model(self):
        from transformer_lens import HookedTransformer
        return HookedTransformer.from_pretrained("gelu-2l")

    def test_extract_mean_diff_and_apply(self, small_model):
        from glassbox.steering import SteeringVectorExporter
        ex = SteeringVectorExporter(method="mean_diff")
        pos = ["I love this", "This is wonderful", "A truly great day"]
        neg = ["I hate this", "This is terrible", "A truly awful day"]
        sv = ex.extract_mean_diff(small_model, pos, neg, layer=1, concept_label="sentiment")
        assert sv.layer == 1
        assert sv.concept_label == "sentiment"
        assert sv.direction.shape[0] == small_model.cfg.d_model

        out = ex.apply(small_model, "The weather today is", sv)
        assert isinstance(out, str)

    def test_extract_pca_method(self, small_model):
        from glassbox.steering import SteeringVectorExporter
        ex = SteeringVectorExporter(method="pca")
        sv = ex.extract_mean_diff(
            small_model, ["good", "great", "nice"], ["bad", "awful", "poor"],
            layer=1, concept_label="sent_pca")
        assert sv.source_info["extraction_method"] == "pca"
