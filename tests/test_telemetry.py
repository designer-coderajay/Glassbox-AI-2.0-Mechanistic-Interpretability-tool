# SPDX-License-Identifier: MIT
"""
tests/test_telemetry.py — coverage for the OpenTelemetry integration.

telemetry.py is pure-Python with OTel as an optional, lazily-imported dep.
These tests exercise the config object, the no-op fallback paths (telemetry
disabled), the trace_span context-manager/decorator, and the
instrument_glassbox wrapper — none of which require opentelemetry installed.
"""

import pytest

from glassbox import telemetry
from glassbox.telemetry import (
    TelemetryConfig,
    is_telemetry_enabled,
    setup_telemetry,
    teardown_telemetry,
    trace_span,
)


@pytest.fixture(autouse=True)
def _no_tracer(monkeypatch):
    """Force telemetry-disabled state for deterministic no-op tests."""
    monkeypatch.setattr(telemetry, "_tracer", None)


# ---------------------------------------------------------------------------
# _check_otel / TelemetryConfig
# ---------------------------------------------------------------------------

class TestBasics:
    def test_check_otel_returns_bool(self):
        assert isinstance(telemetry._check_otel(), bool)

    def test_config_defaults(self):
        c = TelemetryConfig()
        assert c.service_name == "glassbox"
        assert c.endpoint.startswith("http")
        assert c.headers == {}
        assert c.insecure is True
        assert c.export_interval_ms == 5000

    def test_config_custom(self):
        c = TelemetryConfig(service_name="prod", endpoint="http://x:4317",
                            headers={"k": "v"}, insecure=False, export_interval_ms=1000)
        assert c.service_name == "prod" and c.headers == {"k": "v"}
        assert c.insecure is False and c.export_interval_ms == 1000


# ---------------------------------------------------------------------------
# setup / teardown / enabled — no-op paths
# ---------------------------------------------------------------------------

class TestSetupTeardown:
    def test_setup_returns_false_without_otel(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_otel_available", False)
        assert setup_telemetry(endpoint="http://localhost:4317") is False

    def test_setup_returns_false_without_endpoint(self, monkeypatch):
        # Pretend otel is available, but give no endpoint and clear the env var.
        monkeypatch.setattr(telemetry, "_check_otel", lambda: True)
        monkeypatch.delenv("GLASSBOX_OTEL_ENDPOINT", raising=False)
        assert setup_telemetry(endpoint=None) is False

    def test_is_telemetry_enabled_false(self):
        assert is_telemetry_enabled() is False

    def test_teardown_noop_without_otel(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_otel_available", False)
        teardown_telemetry()  # must not raise


# ---------------------------------------------------------------------------
# trace_span — no-op context manager + decorator
# ---------------------------------------------------------------------------

class TestTraceSpan:
    def test_context_manager_noop(self):
        with trace_span("glassbox.test", {"model": "gpt2", "n": 3}) as span:
            assert span is not None
        # no exception = pass

    def test_context_manager_no_attrs(self):
        with trace_span("glassbox.test"):
            pass

    def test_decorator_runs_and_returns(self):
        @trace_span("glassbox.fn", {"a": 1})
        def add(x, y):
            return x + y
        assert add(2, 3) == 5

    def test_set_attribute_noop(self):
        span = trace_span("glassbox.x")
        with span:
            span.set_attribute("key", "value")  # no tracer -> no-op, no error

    def test_exception_propagates_through_span(self):
        with pytest.raises(ValueError):
            with trace_span("glassbox.boom"):
                raise ValueError("x")


# ---------------------------------------------------------------------------
# instrument_glassbox — wraps .analyze and emits (no-op) span
# ---------------------------------------------------------------------------

class _Cfg:
    model_name = "gpt2"


class _Model:
    cfg = _Cfg()


class _FakeGB:
    def __init__(self, result, with_model=True):
        self.model = _Model() if with_model else None
        self._result = result

    def analyze(self, prompt, correct, incorrect, **kwargs):
        return self._result


def _res(f1):
    return {"faithfulness": {"sufficiency": 0.9, "comprehensiveness": 0.7, "f1": f1},
            "n_heads": 4}


class TestInstrument:
    def test_wraps_and_returns_result(self):
        gb = _FakeGB(_res(0.85))
        telemetry.instrument_glassbox(gb)
        out = gb.analyze("Prompt", " Yes", " No")
        assert out["faithfulness"]["f1"] == 0.85

    def test_works_without_model_attr(self):
        gb = _FakeGB(_res(0.7), with_model=False)
        telemetry.instrument_glassbox(gb)
        assert gb.analyze("p", "c", "i")["n_heads"] == 4

    @pytest.mark.parametrize("f1", [0.95, 0.70, 0.55, 0.20])
    def test_grade_branches_all_exercised(self, f1):
        gb = _FakeGB(_res(f1))
        telemetry.instrument_glassbox(gb)
        # Each f1 band drives a different grade branch inside traced_analyze.
        assert gb.analyze("p", "c", "i")["faithfulness"]["f1"] == f1

    def test_passes_method_kwarg(self):
        gb = _FakeGB(_res(0.9))
        telemetry.instrument_glassbox(gb)
        assert gb.analyze("p", "c", "i", method="integrated_gradients")["n_heads"] == 4
