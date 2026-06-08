# SPDX-License-Identifier: MIT
"""
tests/test_telemetry.py — coverage for the OpenTelemetry integration.

telemetry.py is pure-Python with OTel as an optional, lazily-imported dep.
These tests exercise the config object, the no-op fallback paths (telemetry
disabled), the trace_span context-manager/decorator, and the
instrument_glassbox wrapper — none of which require opentelemetry installed.
"""

import sys
from unittest.mock import MagicMock

import pytest

from glassbox import telemetry
from glassbox.telemetry import (
    TelemetryConfig,
    is_telemetry_enabled,
    setup_telemetry,
    teardown_telemetry,
    trace_span,
)

# Module paths that setup_telemetry imports lazily. Stubbing these with
# MagicMocks lets the real setup/teardown/span bodies run without OpenTelemetry
# actually installed — deterministic in every environment (sandbox + CI).
_OTEL_MODS = [
    "opentelemetry", "opentelemetry.trace", "opentelemetry.sdk",
    "opentelemetry.sdk.resources", "opentelemetry.sdk.trace",
    "opentelemetry.sdk.trace.export",
    "opentelemetry.exporter", "opentelemetry.exporter.otlp",
    "opentelemetry.exporter.otlp.proto",
    "opentelemetry.exporter.otlp.proto.grpc",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
]


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


# ---------------------------------------------------------------------------
# OTel-active paths, driven with mocked opentelemetry modules
# ---------------------------------------------------------------------------

@pytest.fixture
def otel(monkeypatch):
    """Stub the opentelemetry module tree so the real setup/span bodies run."""
    for m in _OTEL_MODS:
        monkeypatch.setitem(sys.modules, m, MagicMock())
    monkeypatch.setattr(telemetry, "_otel_available", True)
    monkeypatch.setattr(telemetry, "_tracer", None)
    monkeypatch.setattr(telemetry, "_config", None)
    return sys.modules


class TestCheckOtelBranches:
    def test_true_when_importable(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_otel_available", None)
        monkeypatch.setitem(sys.modules, "opentelemetry", MagicMock())
        assert telemetry._check_otel() is True

    def test_false_when_absent(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_otel_available", None)
        # None in sys.modules forces `import opentelemetry` to raise ImportError
        # regardless of whether the package is actually installed.
        monkeypatch.setitem(sys.modules, "opentelemetry", None)
        assert telemetry._check_otel() is False


class TestSetupActive:
    def test_setup_success(self, otel):
        assert setup_telemetry(endpoint="http://localhost:4317") is True
        assert is_telemetry_enabled() is True

    def test_setup_parses_env_headers(self, otel, monkeypatch):
        monkeypatch.setenv("GLASSBOX_OTEL_HEADERS", "x-key=val,nopair,x-two=v2")
        assert setup_telemetry(endpoint="http://x:4317", headers=None) is True

    def test_setup_body_exception_returns_false(self, otel):
        otel["opentelemetry.sdk.resources"].Resource.side_effect = RuntimeError("boom")
        assert setup_telemetry(endpoint="http://x:4317") is False

    def test_teardown_real_path(self, otel):
        setup_telemetry(endpoint="http://x:4317")
        teardown_telemetry()
        assert is_telemetry_enabled() is False


class TestTraceSpanActive:
    def test_span_sets_attributes(self, otel):
        setup_telemetry(endpoint="http://x:4317")
        with trace_span("op", {"s": "v", "n": 3, "flag": True, "lst": [1, 2]}):
            pass

    def test_span_exception_sets_status(self, otel):
        setup_telemetry(endpoint="http://x:4317")
        with pytest.raises(ValueError):
            with trace_span("op"):
                raise ValueError("x")

    def test_set_attribute_method_active(self, otel):
        setup_telemetry(endpoint="http://x:4317")
        sp = trace_span("op")
        with sp:
            sp.set_attribute("k", "v")

    def test_decorator_active(self, otel):
        setup_telemetry(endpoint="http://x:4317")

        @trace_span("fn", {"a": 1})
        def double(x):
            return x * 2

        assert double(5) == 10

    def test_instrument_with_active_tracer(self, otel):
        setup_telemetry(endpoint="http://x:4317")
        gb = _FakeGB(_res(0.9))
        telemetry.instrument_glassbox(gb)
        # now traced_analyze runs the ACTIVE span path (set_attribute on real span)
        assert gb.analyze("p", "c", "i")["faithfulness"]["f1"] == 0.9


class TestActiveErrorHandling:
    """The defensive `except: pass` handlers must swallow otel errors silently."""

    def test_teardown_swallows_shutdown_error(self, otel):
        sys.modules["opentelemetry"].trace.get_tracer_provider.return_value.shutdown.side_effect = RuntimeError("x")
        setup_telemetry(endpoint="http://x")
        teardown_telemetry()  # must not raise

    def test_enter_swallows_tracer_error(self, otel):
        setup_telemetry(endpoint="http://x")
        telemetry._tracer.start_as_current_span.side_effect = RuntimeError("x")
        with trace_span("op", {"a": 1}):
            pass  # __enter__ swallows the error, no raise

    def test_exit_swallows_cm_error(self, otel):
        setup_telemetry(endpoint="http://x")
        telemetry._tracer.start_as_current_span.return_value.__exit__.side_effect = RuntimeError("x")
        with trace_span("op"):
            pass  # __exit__ swallows the error

    def test_set_attribute_swallows_error(self, otel):
        setup_telemetry(endpoint="http://x")
        span = telemetry._tracer.start_as_current_span.return_value.__enter__.return_value
        span.set_attribute.side_effect = RuntimeError("x")
        sp = trace_span("op")  # no attrs, so __enter__ doesn't call set_attribute
        with sp:
            sp.set_attribute("k", "v")  # the method swallows the error
