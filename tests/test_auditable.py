"""Tests for glassbox/auditable.py — the Auditable Interface + conformance suite."""
from contextlib import contextmanager, nullcontext

from glassbox.auditable import AuditableModel, UnitSpec, run_conformance

_TOKENS = [1, 2, 3]
_LOGITS = [[0.1, 0.2, 0.3, 0.4]]


class MockAdapter:
    """A minimal conformant adapter (deterministic, no-op patch, exact recon)."""

    config = {"n_units": 2, "d_model": 4}

    def forward(self, tokens):
        return [[0.1, 0.2, 0.3, 0.4]]

    def units(self):
        return [UnitSpec("u0", 0, "head", 0), UnitSpec("u1", 1, "head", 1)]

    def read(self, unit, tokens):
        return [0.0, 0.0, 0.0, 0.0]

    def patch(self, unit, value):
        return nullcontext()

    def contributions(self, tokens):
        # two unit contributions that sum exactly to forward() logits
        return {"u0": [[0.1, 0.1, 0.1, 0.1]], "u1": [[0.0, 0.1, 0.2, 0.3]]}


def test_protocol_runtime_checkable():
    assert isinstance(MockAdapter(), AuditableModel)


def test_conformant_adapter_passes_all():
    report = run_conformance(MockAdapter(), _TOKENS)
    assert report.passed is True
    names = {c.name for c in report.checks}
    assert names == {"determinism", "patch_identity", "reconstruction"}
    assert "PASS" in report.summary_line()


def test_black_box_adapter_no_units_still_runs():
    class BlackBox(MockAdapter):
        def units(self):
            return []  # black-box tier
    report = run_conformance(BlackBox(), _TOKENS)
    # patch_identity is vacuously true with no units; determinism + recon hold
    assert report.passed is True


def test_nondeterminism_is_caught():
    class NonDet(MockAdapter):
        _n = 0
        def forward(self, tokens):
            NonDet._n += 1
            return [[float(NonDet._n), 0.0, 0.0, 0.0]]
    report = run_conformance(NonDet(), _TOKENS)
    det = next(c for c in report.checks if c.name == "determinism")
    assert det.passed is False
    assert report.passed is False


def test_bad_patch_is_caught():
    class BadPatch(MockAdapter):
        _patched = False
        def patch(self, unit, value):
            outer = self
            @contextmanager
            def ctx():
                outer._patched = True
                try:
                    yield
                finally:
                    outer._patched = False
            return ctx()
        def forward(self, tokens):
            return [[9.9, 9.9, 9.9, 9.9]] if self._patched else [[0.1, 0.2, 0.3, 0.4]]
    report = run_conformance(BadPatch(), _TOKENS)
    pi = next(c for c in report.checks if c.name == "patch_identity")
    assert pi.passed is False
    assert "u0" in pi.detail


def test_bad_reconstruction_is_caught():
    class BadRecon(MockAdapter):
        def contributions(self, tokens):
            return {"u0": [[1.0, 1.0, 1.0, 1.0]]}  # does not sum to logits
    report = run_conformance(BadRecon(), _TOKENS)
    recon = next(c for c in report.checks if c.name == "reconstruction")
    assert recon.passed is False


def test_adapter_without_contributions_skips_reconstruction():
    class NoContrib:
        def forward(self, tokens): return [[0.1, 0.2]]
        def units(self): return []
        def read(self, unit, tokens): return [0.0]
        def patch(self, unit, value): return nullcontext()
    report = run_conformance(NoContrib(), _TOKENS)
    names = {c.name for c in report.checks}
    assert "reconstruction" not in names
    assert report.passed is True
