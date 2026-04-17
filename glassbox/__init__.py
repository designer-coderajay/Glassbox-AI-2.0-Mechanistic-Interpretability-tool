"""
Glassbox AI — Causal Mechanistic Interpretability Engine
=========================================================

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

    # Multi-agent causal handoff tracing (v3.4.0 — Article 9 system-level risk)
    from glassbox import MultiAgentAudit, AgentCall
    audit  = MultiAgentAudit()
    report = audit.audit_chain([
        AgentCall("router",  "gpt2", "Classify this job application", output_text),
        AgentCall("scorer",  "gpt2", output_text, final_text),
    ])
    print(report.chain_risk_level)

    # Steering vector export (v3.4.0 — Article 9(2)(b) risk mitigation)
    from glassbox import SteeringVectorExporter
    exporter = SteeringVectorExporter()
    sv = exporter.extract_mean_diff(model, pos_prompts, neg_prompts, layer=8,
                                    concept_label="gender_bias")
    exporter.export_pt(sv, "steering/gender_bias.pt")

    # Annex IV Evidence Vault (v3.4.0 — full documentation package)
    from glassbox import build_annex_iv_vault
    vault = build_annex_iv_vault(
        gb_result=result, model_name="gpt2", provider="Acme Corp",
        steering_vectors={"gender_bias": sv},
        output_json="reports/annex-iv.json",
        output_html="reports/annex-iv.html",
    )
    print(vault.to_dict()["compliance_summary"])

Package layout
--------------
glassbox/
  __init__.py           <- you are here — re-exports the public API
  core.py               <- GlassboxV2 class: attribution patching, MFC, FCAS,
                          bootstrap (exact_suff=True), logit lens, EAP,
                          attribution stability, token attribution,
                          attention patterns, _suff_exact (v3.1.0)
  circuit_diff.py       <- CircuitDiff: mechanistic diff between model versions
                          (v3.1.0 — Article 72 post-market monitoring)
  sae_attribution.py    <- SAEFeatureAttributor: sparse feature decomposition
                          via sae-lens hub SAEs or custom .pt checkpoints (v3.1.0)
  telemetry.py          <- OpenTelemetry tracing for self-hosted deployments
                          (v3.1.0 — setup_telemetry, instrument_glassbox)
  composition.py        <- HeadCompositionAnalyzer: Q/K/V composition scores
                          between attention heads (Elhage et al. 2021)
  audit_log.py          <- AuditLog: append-only JSONL audit log with SHA-256
                          hash chain for tamper detection (v2.9.0)
  bias.py               <- BiasAnalyzer: demographic parity, counterfactual
                          fairness, token bias probing (v3.0.0)
  risk_register.py      <- RiskRegister: persistent cross-audit risk tracking
                          (v3.0.0 — Article 9 EU AI Act)
  widget.py             <- CircuitWidget / HeatmapWidget: Jupyter notebook
                          widgets with attribution heatmap (v2.9.0)
  multiagent.py         <- MultiAgentAudit: causal handoff tracing for
                          multi-agent chains (v3.4.0 — Article 9 system risk)
  steering.py           <- SteeringVectorExporter: representation engineering
                          vectors for concept suppression (v3.4.0 — Article 9(2)(b))
  evidence_vault.py     <- AnnexIVEvidenceVault: full Annex IV documentation
                          package builder (v3.4.0 — Article 11)
  cli.py                <- glassbox-ai CLI entry point
  alignment.py          <- DEPRECATED: thin shim kept for back-compat
  utils.py              <- shared utilities
  corruption.py         <- MultiCorruptionPipeline: 4 corruption strategies
                          (name_swap, random_token, gaussian_noise, mean_ablation)
                          + RobustnessReport (v3.7.0 — perturbation sensitivity)
  validation.py         <- SampleSizeGate (n<20 block, n<50 warn) +
                          HeldOutValidator (50/50 split, gap < 0.10) (v3.7.0)
  layernorm_correction.py <- FoldedLayerNorm: fold LN scale γ into W_Q/K/V,
                          compute per-head bias Δα(h), flag |bias/α|>0.15 (v4.0.0)
  fdr.py                <- BenjaminiHochberg: FDR-corrected head significance,
                          E[FDR]≤α, alongside Bonferroni; HeadSignificance,
                          FDRReport (v4.0.0)
  polysemanticity.py    <- PolysemanticityScorerSAE: H(p(feature|head_h)) via
                          SAE features or PCA participation ratio fallback (v4.0.0)
  hessian.py            <- HessianErrorBounds: Pearlmutter HVP for second-order
                          Taylor error bounds; flags |ε(h)/α(h)|>0.20 (v4.1.0)
  causal_scrubbing.py   <- CausalScrubbing: CircuitHypothesis dataclass, resample
                          non-hypothesised activations, CS score 0-1 (v4.1.0)
  das.py                <- DistributedAlignmentSearch: PCA subspace encoding
                          concept via interchange interventions, DAS score (v4.1.0)
  acdc.py               <- AutomatedCircuitDiscovery: edge-level ACDC algorithm
                          (Conmy et al. NeurIPS 2023); exact KL-divergence pruning,
                          ACDCEdge, ACDCCircuit, ACDCResult (v4.2.0)
  multi_arch.py         <- MultiArchAdapter: architecture-aware adapter for GQA
                          (Llama-3, Mistral, Phi-3) and RMSNorm models; GQAAttentionMapper,
                          RMSNormFolding, ArchitectureConfig (v4.2.0)
  cross_model.py        <- CrossModelComparison: run all frameworks across model
                          families; Jaccard circuit similarity, attribution correlation,
                          consensus head detection (v4.2.0)
"""

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "4.2.6"
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
# Bias Analysis — demographic parity, counterfactual fairness (v3.0.0)
# ---------------------------------------------------------------------------
from glassbox.bias import (
    BiasAnalyzer,
    BiasReport,
    CounterfactualFairnessResult,
    DemographicParityResult,
    TokenBiasResult,
)

# ---------------------------------------------------------------------------
# Risk Register — persistent cross-audit risk tracking (v3.0.0)
# ---------------------------------------------------------------------------
from glassbox.risk_register import RiskEntry, RiskRegister

# ---------------------------------------------------------------------------
# Circuit Diff — mechanistic diff between model versions (v3.1.0)
# ---------------------------------------------------------------------------
from glassbox.circuit_diff import CircuitDiff, CircuitDiffResult

# ---------------------------------------------------------------------------
# Cross-Model Circuit Comparison — robustness across architectures (v4.2.0)
# ---------------------------------------------------------------------------
from glassbox.cross_model import (
    CrossModelComparison,
    CrossModelReport,
    CrossModelSimilarity,
    SingleModelResult,
    ModelAnalysisConfig,
    compare_models,
)

# ---------------------------------------------------------------------------
# OpenTelemetry Tracing — self-hosted deployments (v3.1.0)
# ---------------------------------------------------------------------------
from glassbox.telemetry import (
    setup_telemetry,
    teardown_telemetry,
    trace_span,
    is_telemetry_enabled,
    instrument_glassbox,
    TelemetryConfig,
)

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
# Natural Language Explainer — plain English for compliance officers (v3.3.0)
# ---------------------------------------------------------------------------
from glassbox.explain import NaturalLanguageExplainer, explain as explain_result

# ---------------------------------------------------------------------------
# HuggingFace Hub Integration — load_from_hub, HuggingFaceModelCard (v3.3.0)
# ---------------------------------------------------------------------------
from glassbox.hf_integration import load_from_hub, HuggingFaceModelCard

# ---------------------------------------------------------------------------
# MLflow Integration — log_glassbox_run, GlassboxMLflowCallback (v3.3.0)
# ---------------------------------------------------------------------------
from glassbox.mlflow_integration import (
    log_glassbox_run,
    register_compliance_artifact,
    GlassboxMLflowCallback,
)

# ---------------------------------------------------------------------------
# Slack / Teams Alerting — SlackNotifier, TeamsNotifier, AlertConfig (v3.3.0)
# ---------------------------------------------------------------------------
from glassbox.notify import SlackNotifier, TeamsNotifier, AlertConfig

# ---------------------------------------------------------------------------
# Multi-Agent Causal Handoff Tracing (v3.4.0 — Article 9 system-level risk)
# ---------------------------------------------------------------------------
from glassbox.multiagent import (
    MultiAgentAudit,
    AgentCall,
    LiabilityReport,
    AgentLiabilityScore,
    HandoffAnalysis,
    BiasSignals,
)

# ---------------------------------------------------------------------------
# Steering Vector Export (v3.4.0 — Article 9(2)(b) risk mitigation)
# ---------------------------------------------------------------------------
from glassbox.steering import (
    SteeringVector,
    SteeringVectorExporter,
    extract_steering_vector,
)

# ---------------------------------------------------------------------------
# Annex IV Evidence Vault (v3.4.0 — Article 11 full documentation package)
# ---------------------------------------------------------------------------
from glassbox.evidence_vault import (
    AnnexIVEvidenceVault,
    VaultEntry,
    build_annex_iv_vault,
)

# ---------------------------------------------------------------------------
# Multi-Corruption Pipeline — 4 corruption strategies + robustness (v3.7.0)
# ---------------------------------------------------------------------------
from glassbox.corruption import (
    CorruptionStrategy,
    CorruptionResult,
    RobustnessReport,
    MultiCorruptionPipeline,
    ROBUSTNESS_DELTA,
)

# ---------------------------------------------------------------------------
# Statistical Validation Gates — SampleSizeGate + HeldOutValidator (v3.7.0)
# ---------------------------------------------------------------------------
from glassbox.validation import (
    SampleSizeGate,
    SampleSizeError,
    SampleSizeWarning,
    HeldOutValidator,
    HeldOutValidationResult,
    N_HARD_MINIMUM,
    N_SOFT_MINIMUM,
    GENERALIZATION_GAP_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Folded LayerNorm Correction — unbiased attribution patching (v4.0.0)
# ---------------------------------------------------------------------------
from glassbox.layernorm_correction import (
    FoldedLayerNorm,
    LayerNormBiasReport,
    LAYERNORM_BIAS_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR Control — multiple testing correction (v4.0.0)
# ---------------------------------------------------------------------------
from glassbox.fdr import (
    BenjaminiHochberg,
    FDRReport,
    HeadSignificance,
    apply_fdr_correction,
    attribution_to_pvalue,
    bootstrap_se,
)

# ---------------------------------------------------------------------------
# SAE Polysemanticity Score — entropy-based head analysis (v4.0.0)
# ---------------------------------------------------------------------------
from glassbox.polysemanticity import (
    PolysemanticityScorerSAE,
    PolysemanticitySummary,
    HeadPolysemanticity,
)

# ---------------------------------------------------------------------------
# Hessian Error Bounds — second-order Taylor bounds via Pearlmutter HVP (v4.1.0)
# ---------------------------------------------------------------------------
from glassbox.hessian import (
    HessianErrorBounds,
    HessianBoundsReport,
    HeadHessianBound,
    HESSIAN_ERROR_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Causal Scrubbing — CircuitHypothesis + CS score (v4.1.0)
# ---------------------------------------------------------------------------
from glassbox.causal_scrubbing import (
    CausalScrubbing,
    CircuitHypothesis,
    CausalScrubbingResult,
    CS_STRONG_THRESHOLD,
    CS_PARTIAL_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Distributed Alignment Search — linear concept subspace (v4.1.0)
# ---------------------------------------------------------------------------
from glassbox.das import (
    DistributedAlignmentSearch,
    DASResult,
    DAS_SCORE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Automated Circuit Discovery (ACDC) — edge-level KL pruning (v4.2.0)
# ---------------------------------------------------------------------------
from glassbox.acdc import (
    AutomatedCircuitDiscovery,
    ACDCEdge,
    ACDCCircuit,
    ACDCResult,
    ACDC_KL_THRESHOLD,
    ACDC_FAITHFULNESS_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Multi-Architecture Adapter — GQA + RMSNorm support (v4.2.0)
# ---------------------------------------------------------------------------
from glassbox.multi_arch import (
    MultiArchAdapter,
    ArchitectureConfig,
    ArchitectureReport,
    GQAAttentionMapper,
    RMSNormFolding,
    SUPPORTED_ARCHITECTURES,
    RMSNORM_ARCHITECTURES,
    GQA_ARCHITECTURES,
)

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
    # Bias analysis — demographic parity, counterfactual fairness (v3.0.0)
    "BiasAnalyzer",
    "BiasReport",
    "CounterfactualFairnessResult",
    "DemographicParityResult",
    "TokenBiasResult",
    # Risk register — persistent cross-audit risk tracking (v3.0.0)
    "RiskEntry",
    "RiskRegister",
    # Circuit diff — model version comparison (v3.1.0)
    "CircuitDiff",
    "CircuitDiffResult",
    # Cross-model circuit comparison — robustness across architectures (v4.2.0)
    "CrossModelComparison",
    "CrossModelReport",
    "CrossModelSimilarity",
    "SingleModelResult",
    "ModelAnalysisConfig",
    "compare_models",
    # OpenTelemetry tracing (v3.1.0)
    "setup_telemetry",
    "teardown_telemetry",
    "trace_span",
    "is_telemetry_enabled",
    "instrument_glassbox",
    "TelemetryConfig",
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
    # Natural Language Explainer (v3.3.0)
    "NaturalLanguageExplainer",
    "explain_result",
    # HuggingFace Hub Integration (v3.3.0)
    "load_from_hub",
    "HuggingFaceModelCard",
    # MLflow Integration (v3.3.0)
    "log_glassbox_run",
    "register_compliance_artifact",
    "GlassboxMLflowCallback",
    # Slack / Teams Alerting (v3.3.0)
    "SlackNotifier",
    "TeamsNotifier",
    "AlertConfig",
    # Multi-Agent Causal Handoff Tracing (v3.4.0)
    "MultiAgentAudit",
    "AgentCall",
    "LiabilityReport",
    "AgentLiabilityScore",
    "HandoffAnalysis",
    "BiasSignals",
    # Steering Vector Export (v3.4.0)
    "SteeringVector",
    "SteeringVectorExporter",
    "extract_steering_vector",
    # Annex IV Evidence Vault (v3.4.0)
    "AnnexIVEvidenceVault",
    "VaultEntry",
    "build_annex_iv_vault",
    # Multi-Corruption Pipeline (v3.7.0)
    "CorruptionStrategy",
    "CorruptionResult",
    "RobustnessReport",
    "MultiCorruptionPipeline",
    "ROBUSTNESS_DELTA",
    # Statistical Validation Gates (v3.7.0)
    "SampleSizeGate",
    "SampleSizeError",
    "SampleSizeWarning",
    "HeldOutValidator",
    "HeldOutValidationResult",
    "N_HARD_MINIMUM",
    "N_SOFT_MINIMUM",
    "GENERALIZATION_GAP_THRESHOLD",
    # Folded LayerNorm Correction (v4.0.0)
    "FoldedLayerNorm",
    "LayerNormBiasReport",
    "LAYERNORM_BIAS_THRESHOLD",
    # Benjamini-Hochberg FDR (v4.0.0)
    "BenjaminiHochberg",
    "FDRReport",
    "HeadSignificance",
    "apply_fdr_correction",
    "attribution_to_pvalue",
    "bootstrap_se",
    # SAE Polysemanticity Score (v4.0.0)
    "PolysemanticityScorerSAE",
    "PolysemanticitySummary",
    "HeadPolysemanticity",
    # Hessian Error Bounds (v4.1.0)
    "HessianErrorBounds",
    "HessianBoundsReport",
    "HeadHessianBound",
    "HESSIAN_ERROR_THRESHOLD",
    # Causal Scrubbing (v4.1.0)
    "CausalScrubbing",
    "CircuitHypothesis",
    "CausalScrubbingResult",
    "CS_STRONG_THRESHOLD",
    "CS_PARTIAL_THRESHOLD",
    # Distributed Alignment Search (v4.1.0)
    "DistributedAlignmentSearch",
    "DASResult",
    "DAS_SCORE_THRESHOLD",
    # Automated Circuit Discovery — ACDC (v4.2.0)
    "AutomatedCircuitDiscovery",
    "ACDCEdge",
    "ACDCCircuit",
    "ACDCResult",
    "ACDC_KL_THRESHOLD",
    "ACDC_FAITHFULNESS_THRESHOLD",
    # Multi-Architecture Adapter — GQA + RMSNorm (v4.2.0)
    "MultiArchAdapter",
    "ArchitectureConfig",
    "ArchitectureReport",
    "GQAAttentionMapper",
    "RMSNormFolding",
    "SUPPORTED_ARCHITECTURES",
    "RMSNORM_ARCHITECTURES",
    "GQA_ARCHITECTURES",
    # Meta
    "__version__",
    # Deprecated
    "GlassboxEngine",
]
