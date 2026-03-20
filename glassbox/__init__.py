"""
Glassbox 2.0 — Causal Mechanistic Interpretability Engine
==========================================================

Quick start
-----------
    from transformer_lens import HookedTransformer
    from glassbox import GlassboxV2

    model = HookedTransformer.from_pretrained("gpt2")
    gb    = GlassboxV2(model)

    result = gb.analyze(
        prompt    = "When Mary and John went to the store, John gave a drink to",
        correct   = " Mary",
        incorrect = " John",
    )
    print(result["faithfulness"])
    # {'sufficiency': 0.80, 'comprehensiveness': 0.37, 'f1': 0.49,
    #  'category': 'backup_mechanisms', 'suff_is_approx': True}

    # Token-level saliency map
    tokens = model.to_tokens("When Mary and John went to the store, John gave a drink to")
    tok_attr = gb.token_attribution(tokens, model.to_single_token(" Mary"),
                                            model.to_single_token(" John"))
    print(tok_attr["top_tokens"])

    # SAE feature attribution (requires: pip install sae-lens)
    from glassbox import SAEFeatureAttributor
    sfa    = SAEFeatureAttributor(model)
    feats  = sfa.attribute(tokens, " Mary", " John", layers=[9, 10, 11])
    print(feats["top_features"][:5])

    # Head composition scores (Elhage et al. 2021)
    from glassbox import HeadCompositionAnalyzer
    comp   = HeadCompositionAnalyzer(model)
    scores = comp.all_composition_scores(result["circuit"])
    print(scores["combined_edges"][:5])

    # Tamper-evident audit log (v2.9.0)
    from glassbox import AuditLog
    log = AuditLog("glassbox_audit.jsonl")
    log.append_from_result(result, auditor="ajay@example.com")
    print(log.summary())          # grade distribution, compliance rate, avg F1

    # Jupyter widget (requires: pip install glassbox-mech-interp[jupyter])
    from glassbox.widget import CircuitWidget
    w = CircuitWidget.from_prompt(gb, "When Mary and John ...", " Mary", " John")
    w.show()                      # renders inline in a notebook cell

Package layout
--------------
glassbox/
  __init__.py           ← you are here — re-exports the public API
  core.py               ← GlassboxV2 class: attribution patching, MFC, FCAS,
                          bootstrap, logit lens, EAP, attribution stability,
                          token attribution, attention patterns
  sae_attribution.py    ← SAEFeatureAttributor: sparse feature decomposition
                          of circuit components via sae-lens SAEs
  composition.py        ← HeadCompositionAnalyzer: Q/K/V composition scores
                          between attention heads (Elhage et al. 2021)
  audit_log.py          ← AuditLog: append-only JSONL audit log with SHA-256
                          hash chain for tamper detection (v2.9.0)
  widget.py             ← CircuitWidget / HeatmapWidget: Jupyter notebook
                          widgets with attribution heatmap (v2.9.0)
  cli.py                ← glassbox-ai CLI entry point
  alignment.py          ← DEPRECATED: thin shim kept for back-compat
  utils.py              ← shared utilities
"""

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "2.9.0"
__author__  = "Ajay Pravin Mahale"
__email__   = "mahale.ajay01@gmail.com"

# ---------------------------------------------------------------------------
# Core engine — always available
# ---------------------------------------------------------------------------
from glassbox.core import GlassboxV2          # primary analysis class

# ---------------------------------------------------------------------------
# SAE Feature Attribution — requires sae-lens (optional)
# ---------------------------------------------------------------------------
try:
    from glassbox.sae_attribution import SAEFeatureAttributor
    _SAE_AVAILABLE = True
except ImportError:
    # sae-lens not installed.  Expose a stub so `from glassbox import
    # SAEFeatureAttributor` succeeds silently; the class raises a clear
    # ImportError only when the user tries to *instantiate* it.
    class SAEFeatureAttributor:  # type: ignore[no-redef]
        """Stub raised when sae-lens is not installed.

        Install the optional dependency::

            pip install 'glassbox-mech-interp[sae]'
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SAEFeatureAttributor requires sae-lens. "
                "Install it with:  pip install 'glassbox-mech-interp[sae]'"
            )
    _SAE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Head Composition Analysis — always available (no extra deps)
# ---------------------------------------------------------------------------
from glassbox.composition import HeadCompositionAnalyzer

# ---------------------------------------------------------------------------
# Public type aliases and constants
# ---------------------------------------------------------------------------
from glassbox.types import (
    HeadTuple,
    CircuitList,
    AttributionDict,
    PromptTuple,
    VALID_HEAD_TYPES,
    FAITHFULNESS_CATEGORIES,
    ATTRIBUTION_METHODS,
)

# ---------------------------------------------------------------------------
# Utility helpers — exposed for power users and extension authors
# ---------------------------------------------------------------------------
from glassbox.utils import (
    stable_api,
    deprecated,
    format_head_label,
    parse_head_label,
    estimate_forward_pass_memory_mb,
)

# ---------------------------------------------------------------------------
# EU AI Act Annex IV Compliance Report Generator — core product
# ---------------------------------------------------------------------------
from glassbox.compliance import (
    AnnexIVReport,             # .to_model_card() and .save_model_card() added in v2.8.0
    DeploymentContext,
    RiskClassification,
    ExplainabilityGrade,
    ComplianceStatus,
)

# ---------------------------------------------------------------------------
# Black-Box Audit Mode — any model via API (no TransformerLens needed)
# ---------------------------------------------------------------------------
from glassbox.audit import (
    BlackBoxAuditor,
    ModelProvider,
    BlackBoxResult,
    from_env as black_box_from_env,
)

# ---------------------------------------------------------------------------
# Tamper-evident Audit Log — SHA-256 hash chain, JSONL persistence (v2.9.0)
# ---------------------------------------------------------------------------
from glassbox.audit_log import AuditLog, AuditRecord

# ---------------------------------------------------------------------------
# Jupyter Notebook Widgets — CircuitWidget, HeatmapWidget (v2.9.0)
# ---------------------------------------------------------------------------
try:
    from glassbox.widget import CircuitWidget, HeatmapWidget
    _WIDGETS_AVAILABLE = True
except ImportError:
    # ipywidgets not installed; stubs so `from glassbox import CircuitWidget`
    # succeeds with a clear message at instantiation time.
    class CircuitWidget:  # type: ignore[no-redef]
        """Stub: install ipywidgets first.

        Run::

            pip install 'glassbox-mech-interp[jupyter]'
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CircuitWidget requires ipywidgets. "
                "Install with:  pip install 'glassbox-mech-interp[jupyter]'"
            )

    class HeatmapWidget:  # type: ignore[no-redef]
        """Stub: install ipywidgets first.

        Run::

            pip install 'glassbox-mech-interp[jupyter]'
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "HeatmapWidget requires ipywidgets. "
                "Install with:  pip install 'glassbox-mech-interp[jupyter]'"
            )
    _WIDGETS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------
GlassboxEngine = GlassboxV2   # deprecated — use GlassboxV2

__all__ = [
    # Primary classes
    "GlassboxV2",
    "SAEFeatureAttributor",          # requires sae-lens
    "HeadCompositionAnalyzer",
    # Compliance — EU AI Act Annex IV
    "AnnexIVReport",
    "DeploymentContext",
    "RiskClassification",
    "ExplainabilityGrade",
    "ComplianceStatus",
    # Black-box audit — any model via API
    "BlackBoxAuditor",
    "ModelProvider",
    "BlackBoxResult",
    "black_box_from_env",
    # Audit log — tamper-evident, hash-chained (v2.9.0)
    "AuditLog",
    "AuditRecord",
    # Jupyter widgets (v2.9.0; requires ipywidgets)
    "CircuitWidget",
    "HeatmapWidget",
    # Type aliases
    "HeadTuple",
    "CircuitList",
    "AttributionDict",
    "PromptTuple",
    # Constants
    "VALID_HEAD_TYPES",
    "FAITHFULNESS_CATEGORIES",
    "ATTRIBUTION_METHODS",
    # Utilities
    "stable_api",
    "deprecated",
    "format_head_label",
    "parse_head_label",
    "estimate_forward_pass_memory_mb",
    # Meta
    "__version__",
    # Deprecated
    "GlassboxEngine",
]
