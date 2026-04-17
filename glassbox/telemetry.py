"""
glassbox.telemetry
==================
OpenTelemetry tracing integration for self-hosted Glassbox deployments.

Exports OTLP traces from every Glassbox analysis call so your existing
observability stack (Datadog, Honeycomb, Jaeger, Grafana Tempo, etc.)
can capture:

  - Per-analysis span with prompt hash, model name, grade, F1, circuit size
  - Per-method child spans: attribution_patching, minimum_faithful_circuit,
    _comp (exact comprehensiveness), _suff_exact (exact sufficiency)
  - Compliance report generation spans
  - Risk register ingestion spans

Why this matters for compliance teams (Article 72)
---------------------------------------------------
EU AI Act Article 72 requires post-market monitoring. OTel traces give you
a continuous, machine-readable audit trail of model behaviour over time
with sub-millisecond latency overhead. Self-host → traces never leave your
infrastructure.

Usage — one-time setup
-----------------------
    from glassbox.telemetry import setup_telemetry

    # Jaeger / OTLP
    setup_telemetry(
        service_name = "glassbox-prod",
        endpoint     = "http://localhost:4317",   # OTLP gRPC
    )

    # Honeycomb
    setup_telemetry(
        service_name = "glassbox-prod",
        endpoint     = "https://api.honeycomb.io",
        headers      = {"x-honeycomb-team": "YOUR_API_KEY"},
    )

    # After setup, ALL GlassboxV2.analyze() calls emit traces automatically.
    from glassbox import GlassboxV2
    gb = GlassboxV2(model)
    result = gb.analyze(...)   # → span: "glassbox.analyze"

No-op mode
----------
If opentelemetry-sdk is not installed, all telemetry calls are silently
no-ops — Glassbox works normally with zero telemetry overhead.

    pip install 'glassbox-mech-interp[telemetry]'
    # or manually:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp

Environment variable configuration (alternative to code setup)
---------------------------------------------------------------
    GLASSBOX_OTEL_ENDPOINT   = "http://localhost:4317"
    GLASSBOX_OTEL_SERVICE    = "glassbox"
    GLASSBOX_OTEL_HEADERS    = "x-key=val,x-other=val2"
    GLASSBOX_OTEL_ENABLED    = "true"   (default: true if endpoint is set)

References
----------
OpenTelemetry Python SDK: https://opentelemetry.io/docs/instrumentation/python/
OTLP exporter: https://opentelemetry-python-contrib.readthedocs.io/en/latest/
EU AI Act Article 72 — Post-market monitoring obligations for high-risk AI.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "setup_telemetry",
    "teardown_telemetry",
    "trace_span",
    "is_telemetry_enabled",
    "TelemetryConfig",
]

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

_tracer         = None     # opentelemetry.trace.Tracer | None
_otel_available = None     # bool | None  (lazy check)
_config: Optional["TelemetryConfig"] = None


def _check_otel() -> bool:
    global _otel_available
    if _otel_available is None:
        try:
            import opentelemetry  # noqa: F401
            _otel_available = True
        except ImportError:
            _otel_available = False
    return _otel_available


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryConfig:
    """
    Telemetry configuration for a Glassbox deployment.

    Parameters
    ----------
    service_name : str
        OTel service.name attribute (default "glassbox").
    endpoint     : str
        OTLP gRPC or HTTP endpoint (e.g. "http://localhost:4317").
    headers      : dict, optional
        Additional HTTP headers (e.g. API key for managed services).
    insecure     : bool
        Use insecure gRPC channel (default True for localhost).
    export_interval_ms : int
        Batch export interval in milliseconds (default 5000).
    """

    def __init__(
        self,
        service_name:        str  = "glassbox",
        endpoint:            str  = "http://localhost:4317",
        headers:             Optional[Dict[str, str]] = None,
        insecure:            bool = True,
        export_interval_ms:  int  = 5000,
    ) -> None:
        self.service_name       = service_name
        self.endpoint           = endpoint
        self.headers            = headers or {}
        self.insecure           = insecure
        self.export_interval_ms = export_interval_ms


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def setup_telemetry(
    service_name:        str  = "glassbox",
    endpoint:            Optional[str] = None,
    headers:             Optional[Dict[str, str]] = None,
    insecure:            bool = True,
    export_interval_ms:  int  = 5000,
) -> bool:
    """
    Initialise OpenTelemetry tracing for Glassbox.

    Call once at application startup. All subsequent GlassboxV2.analyze()
    calls will emit traces to the configured endpoint.

    Parameters
    ----------
    service_name       : OTel service.name (default "glassbox").
    endpoint           : OTLP endpoint URL.
                         Falls back to GLASSBOX_OTEL_ENDPOINT env var.
    headers            : HTTP headers for authentication.
                         Falls back to GLASSBOX_OTEL_HEADERS env var.
    insecure           : Disable TLS for gRPC (default True for localhost).
    export_interval_ms : Batch export interval in milliseconds.

    Returns
    -------
    bool : True if telemetry was successfully initialised, False otherwise.

    Examples
    --------
    >>> setup_telemetry(service_name="glassbox-prod",
    ...                 endpoint="http://localhost:4317")
    True
    """
    global _tracer, _config

    if not _check_otel():
        logger.warning(
            "OpenTelemetry SDK not installed. Telemetry is disabled.\n"
            "Install with:  pip install 'glassbox-mech-interp[telemetry]'"
        )
        return False

    # Resolve endpoint from env if not given
    resolved_endpoint = (
        endpoint
        or os.environ.get("GLASSBOX_OTEL_ENDPOINT")
    )
    if not resolved_endpoint:
        logger.warning(
            "setup_telemetry: no endpoint provided and GLASSBOX_OTEL_ENDPOINT "
            "env var is not set. Telemetry disabled."
        )
        return False

    resolved_service = (
        service_name
        or os.environ.get("GLASSBOX_OTEL_SERVICE", "glassbox")
    )

    # Parse headers from env if not given
    resolved_headers = headers or {}
    env_headers = os.environ.get("GLASSBOX_OTEL_HEADERS", "")
    if env_headers and not resolved_headers:
        for pair in env_headers.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                resolved_headers[k.strip()] = v.strip()

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(
                endpoint=resolved_endpoint,
                headers=resolved_headers,
                insecure=insecure,
            )
        except ImportError:
            # Fallback to HTTP/protobuf exporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(
                endpoint=resolved_endpoint,
                headers=resolved_headers,
            )

        resource = Resource(attributes={SERVICE_NAME: resolved_service})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                export_timeout_millis=export_interval_ms,
            )
        )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("glassbox", "4.2.6")
        _config = TelemetryConfig(
            service_name       = resolved_service,
            endpoint           = resolved_endpoint,
            headers            = resolved_headers,
            insecure           = insecure,
            export_interval_ms = export_interval_ms,
        )
        logger.info(
            "Glassbox telemetry initialised: service=%s endpoint=%s",
            resolved_service, resolved_endpoint,
        )
        return True

    except Exception as exc:
        logger.error("setup_telemetry failed: %s", exc)
        return False


def teardown_telemetry() -> None:
    """Flush pending spans and shut down the tracer provider."""
    global _tracer, _config
    if not _check_otel():
        return
    try:
        from opentelemetry import trace
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception as exc:
        logger.warning("teardown_telemetry error: %s", exc)
    _tracer = None
    _config = None


def is_telemetry_enabled() -> bool:
    """Return True if telemetry has been successfully initialised."""
    return _tracer is not None


def trace_span(
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Context manager / decorator that wraps a code block in an OTel span.

    No-op if telemetry is not enabled.

    Usage as context manager
    ------------------------
    >>> with trace_span("glassbox.my_operation", {"model": "gpt2"}):
    ...     do_work()

    Usage as decorator
    ------------------
    >>> @trace_span("glassbox.my_function")
    ... def my_function(...):
    ...     ...
    """
    return _TraceSpan(span_name, attributes or {})


class _TraceSpan:
    """Internal context manager for OTel spans with graceful no-op fallback."""

    def __init__(self, name: str, attrs: Dict[str, Any]) -> None:
        self._name  = name
        self._attrs = attrs
        self._span  = None
        self._cm    = None

    def __enter__(self):
        if _tracer is None:
            return self
        try:
            from opentelemetry import trace
            self._cm   = _tracer.start_as_current_span(self._name)
            self._span = self._cm.__enter__()
            if self._attrs and self._span is not None:
                for k, v in self._attrs.items():
                    self._span.set_attribute(k, str(v) if not isinstance(v, (bool, int, float, str)) else v)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cm is not None:
            try:
                if exc_type is not None and self._span:
                    from opentelemetry.trace import StatusCode
                    self._span.set_status(StatusCode.ERROR, str(exc_val))
                self._cm.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
        return False   # do not suppress exceptions

    def __call__(self, func: Callable) -> Callable:
        """Support @trace_span("name") as a decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _TraceSpan(self._name, self._attrs):
                return func(*args, **kwargs)
        return wrapper

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute (no-op if telemetry disabled)."""
        if self._span is not None:
            try:
                self._span.set_attribute(key, value)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: instrument a GlassboxV2 instance
# ─────────────────────────────────────────────────────────────────────────────

def instrument_glassbox(gb) -> None:
    """
    Monkey-patch a GlassboxV2 instance to auto-emit OTel spans.

    Call after setup_telemetry(). Each analyze() call will emit a span
    with attributes: model_name, n_circuit_heads, f1, grade, method.

    Parameters
    ----------
    gb : GlassboxV2
        The instance to instrument.

    Example
    -------
    >>> setup_telemetry(endpoint="http://localhost:4317")
    >>> gb = GlassboxV2(model)
    >>> instrument_glassbox(gb)
    >>> result = gb.analyze(...)   # → span emitted automatically
    """
    original_analyze = gb.analyze

    @functools.wraps(original_analyze)
    def traced_analyze(prompt, correct, incorrect, **kwargs):
        prompt_hash = hashlib.sha256(
            (prompt + correct + incorrect).encode()
        ).hexdigest()[:12]

        model_name = getattr(
            getattr(gb, "model", None),
            "cfg",
            type("_", (), {"model_name": "unknown"})()
        ).model_name

        span_attrs = {
            "glassbox.prompt_hash":  prompt_hash,
            "glassbox.model":        str(model_name),
            "glassbox.method":       kwargs.get("method", "taylor"),
        }

        with _TraceSpan("glassbox.analyze", span_attrs) as span:
            t0     = time.perf_counter()
            result = original_analyze(prompt, correct, incorrect, **kwargs)
            elapsed = time.perf_counter() - t0

            faith = result.get("faithfulness", {})
            span.set_attribute("glassbox.f1",           round(faith.get("f1",    0.0), 4))
            span.set_attribute("glassbox.sufficiency",  round(faith.get("sufficiency", 0.0), 4))
            span.set_attribute("glassbox.comprehensiveness", round(faith.get("comprehensiveness", 0.0), 4))
            span.set_attribute("glassbox.circuit_heads", result.get("n_heads", 0))
            span.set_attribute("glassbox.duration_ms",  round(elapsed * 1000, 1))

            f1 = faith.get("f1", 0.0)
            grade = "A" if f1 >= 0.80 else "B" if f1 >= 0.65 else "C" if f1 >= 0.50 else "D"
            span.set_attribute("glassbox.grade", grade)

        return result

    gb.analyze = traced_analyze
    logger.info("GlassboxV2 instance instrumented for OTel tracing.")
