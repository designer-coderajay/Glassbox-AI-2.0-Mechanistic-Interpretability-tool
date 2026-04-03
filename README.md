<div align="center">

# Glassbox 4.1.0

**Open-source EU AI Act Annex IV compliance documentation toolkit. Works on any LLM.**
**18/18 mathematical frameworks. Foundationally rigorous. Production-ready.**

[![PyPI version](https://img.shields.io/pypi/v/glassbox-mech-interp?color=blue)](https://pypi.org/project/glassbox-mech-interp/)
[![PyPI downloads](https://img.shields.io/pypi/dm/glassbox-mech-interp?color=blue&label=downloads%2Fmonth)](https://pypistats.org/packages/glassbox-mech-interp)
[![GitHub last commit](https://img.shields.io/github/last-commit/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool?color=green)](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/issues)
[![Live Analytics](https://img.shields.io/badge/Live%20Analytics-ClickHouse-FFCC01?logo=clickhouse&logoColor=black)](https://clickpy.clickhouse.com/dashboard/glassbox-mech-interp)
[![License: MIT](https://img.shields.io/badge/Core-MIT-green.svg)](LICENSE) [![License: BSL 1.1](https://img.shields.io/badge/Compliance%20Engine-BSL%201.1-orange.svg)](LICENSE-COMMERCIAL) [![Patents Pending](https://img.shields.io/badge/Patents-Pending-blue.svg)](PATENTS.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)
[![Website](https://img.shields.io/badge/Website-glassboxai.online-blue)](https://project-gu05p.vercel.app)
[![arXiv](https://img.shields.io/badge/arXiv-2603.09988-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.09988)
[![Tests](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml)

[**Website**](https://project-gu05p.vercel.app) · [**Live Demo**](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) · [**Paper**](https://arxiv.org/abs/2603.09988) · [**PyPI**](https://pypi.org/project/glassbox-mech-interp/) · [**GitHub**](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)

</div>

---

**For compliance teams:** Regulation (EU) 2024/1689 (AI Act) requires Annex IV technical documentation for every high-risk AI system (Article 11). Enforcement begins August 2026. Glassbox automates generation of the full 9-section Annex IV draft — from open-source models (white-box) or any proprietary API like GPT-4 and Claude (black-box). Outputs are structured documentation aids; they do not constitute legal advice, a declaration of conformity, or a guarantee of regulatory compliance. See [Legal Notices](#legal-notices--regulatory-disclaimer).

**For researchers:** one function call discovers the minimum faithful circuit in a transformer — the smallest subgraph of attention heads causally responsible for a prediction. Preliminary benchmarks show 15–37× faster than ACDC on GPT-2 (single-run, Apple M2 Pro — see [Benchmarks](#benchmarks)). Every approximation is disclosed.

---

## Table of Contents

- [Live Services](#live-services)
- [Quickstart](#quickstart)
- [What's New in v4.1.0](#whats-new-in-v410)
- [What's New in v4.0.0](#whats-new-in-v400)
- [What's New in v3.7.0](#whats-new-in-v370)
- [What's New in v3.6.0](#whats-new-in-v360)
- [What's New in v3.5.0](#whats-new-in-v350)
- [What's New in v3.4.0](#whats-new-in-v340)
- [What's New in v3.3.0](#whats-new-in-v330)
- [What's New in v3.1.0](#whats-new-in-v310)
- [What's New in v3.0.0](#whats-new-in-v300)
- [EU AI Act Compliance — Annex IV Reports](#eu-ai-act-compliance--annex-iv-reports)
- [Black-Box Audit — Any Model via API](#black-box-audit--any-model-via-api)
- [REST API (Hosted)](#rest-api-hosted)
- [What's Novel](#whats-novel)
- [How It Works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Usage Examples](#usage-examples)
- [CLI](#cli)
- [Installation](#installation)
- [Dashboard](#dashboard)
- [Self-Hosting (Docker / Air-Gapped VPC)](#self-hosting-docker--air-gapped-vpc)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
- [Methodology & IP Documentation](#methodology--ip-documentation)
- [Mathematical Disclosures](#mathematical-disclosures)
- [Mathematical Foundations Reference](#mathematical-foundations-reference)
- [Cross-Model Faithfulness Study](#cross-model-faithfulness-study)
- [Paper](#paper)
- [Citation](#citation)
- [Related Tools](#related-tools)
- [Security & Privacy](#security--privacy)
- [Legal Notices & Regulatory Disclaimer](#legal-notices--regulatory-disclaimer)
- [Project & Privacy Notice](#project--privacy-notice)
- [License](#license)

---

## Live Services

| Service | URL | Description |
|---------|-----|-------------|
| **Website** | [project-gu05p.vercel.app](https://project-gu05p.vercel.app) | Marketing site — features, pricing, code examples. Always up. |
| **Live Demo** | [HuggingFace Space](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) | Interactive circuit analysis on open-source models. No install needed. |
| **PyPI Package** | [glassbox-mech-interp](https://pypi.org/project/glassbox-mech-interp/) | `pip install glassbox-mech-interp` — v4.1.0 |
| **Self-Hosted API** | [See Docker guide](#self-hosting-docker--air-gapped-vpc) | Deploy the REST API on your own infra or Railway. |

---

## Quickstart

```bash
pip install glassbox-mech-interp
```

```python
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

model = HookedTransformer.from_pretrained("gpt2")
gb    = GlassboxV2(model)

result = gb.analyze(
    prompt    = "When Mary and John went to the store, John gave a drink to",
    correct   = " Mary",
    incorrect = " John",
)

print(result["circuit"])
# [(9, 9), (9, 6), (10, 0), (8, 6), ...]   <- (layer, head) tuples

print(result["faithfulness"])
# {'sufficiency': 0.80,          # Taylor approximation (fast, suff_is_approx=True)
#  'comprehensiveness': 0.37,    # exact ablation
#  'f1': 0.49,
#  'category': 'backup_mechanisms',
#  'suff_is_approx': True}       # True = approx; use bootstrap_metrics() for exact ~100%
```

No model weights? Use the [live HuggingFace demo](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) — no install required.

---

## What's New in v4.1.0

Glassbox v4.1.0 completes the **ROADMAP_V4 mathematical framework** — 18/18 frameworks now implemented. This version brings the three hardest features: Hessian-based reliability bounds, Anthropic-standard causal scrubbing, and Distributed Alignment Search. These close the gap vs Harvard/MIT/Anthropic/DeepMind research standards.

### 1. HessianErrorBounds — Second-Order Taylor Reliability Certificates

Standard attribution patching uses a first-order Taylor approximation. If the second-order term dominates, the ranking is unreliable. `HessianErrorBounds` computes `ε(h) = ½·δzᵀ·H_h·δz` via Pearlmutter (1994) HVP, certifying whether the approximation holds.

```python
from glassbox import HessianErrorBounds

hb     = HessianErrorBounds(model)
bounds = hb.compute(
    attributions = result["attributions_raw"],   # {(layer, head): score}
    clean_tokens = clean_tokens,
    corr_tokens  = corr_tokens,
    target_tok   = target_id,
    distract_tok = distract_id,
)

print(bounds.summary_line())
# Hessian [reliable ✓] | max_ratio=0.043 dominated=0/144 heads (threshold=0.2)

print(bounds.approximation_reliable)   # True — first-order ranking is certified
print(bounds.to_dict()["dominated_heads"])  # [] — no heads dominated by second-order terms
```

Flags `hessian_dominated` heads where `|ε(h)| / |α(h)| > 0.20`. Maps to **Art. 13(1)** transparency.

### 2. CausalScrubbing — Anthropic-Standard Circuit Hypothesis Testing

Attribution identifies *which* heads matter. Causal scrubbing (Chan et al., Anthropic 2022) answers: does the identified circuit *causally* implement the claimed computation? `CS(H) = E[LD_scrubbed]/LD_clean` — strong ≥ 0.80.

```python
from glassbox import CausalScrubbing, CircuitHypothesis

# Use the canonical Wang et al. 2022 IOI circuit hypothesis
hypothesis = CircuitHypothesis.from_wang2022_ioi()
# Or define your own:
# hypothesis = CircuitHypothesis.from_list("my_circuit", [(9,6),(9,9),(10,0)])

scrubber = CausalScrubbing(model, n_samples=5)
result   = scrubber.evaluate(
    hypothesis   = hypothesis,
    prompt       = "When Mary and John went to the store, John gave a drink to",
    corr_prompt  = "When John and Mary went to the store, Mary gave a drink to",
    target_tok   = target_id,
    distract_tok = distract_id,
)

print(result.summary_line())
# CausalScrubbing [Wang2022_IOI_Circuit] ✓✓ | CS=0.8923 (strong) | LD_clean=4.23...

print(result.interpretation)   # "strong" — hypothesis causally explains the circuit
print(result.cs_score)         # 0.8923
```

Maps to **Art. 9(1)** risk management — formal causal account, not just correlation.

### 3. DistributedAlignmentSearch — Linear Concept Subspace Discovery

DAS (Geiger et al. 2023) finds *where* in the residual stream a concept is encoded. Learns rotation matrix `R ∈ R^{d_model × k}` via PCA on activation differences; validates via interchange interventions.

```python
from glassbox import DistributedAlignmentSearch

das    = DistributedAlignmentSearch(model, concept_dims=4)
result = das.search(
    concept_label         = "IO_name_position",
    clean_prompts_tokens  = clean_token_list,
    counterfactual_tokens = corr_token_list,
    target_tok            = target_id,
    distract_tok          = distract_id,
    target_layer          = 9,
    target_position       = -1,
)

print(result.summary_line())
# DAS [IO_name_position] ENCODED ✓ | layer=9 pos=-1 | score=0.8234 dims=4 expl_var=0.731

print(result.concept_encoded)      # True — concept clearly encoded in 4-dimensional subspace
print(result.explained_variance)   # 0.731 — 73% of Δz variance explained by top-4 dims

# Sweep all layers to find where concept is strongest
all_results = das.search_all_layers("IO_name_position", clean_tokens, corr_tokens,
                                     target_id, distract_id)
# Sorted by das_score descending
print(all_results[0].target_layer)  # 9 — concept strongest at layer 9
```

Maps to **Art. 15(1)** robustness — localises concept encoding for controlled interventions.

### Mathematical Completeness: 18/18 ✓

| Framework | v3.6.0 | v3.7.0 | v4.0.0 | v4.1.0 |
|---|---|---|---|---|
| Attribution patching (Nanda 2023) | ✓ | ✓ | ✓ | ✓ |
| Sufficiency / Comprehensiveness / F1 | ✓ | ✓ | ✓ | ✓ |
| Fisher Z cross-model comparison | ✓ | ✓ | ✓ | ✓ |
| Edge Attribution Patching (Syed 2024) | ✓ | ✓ | ✓ | ✓ |
| BCa Bootstrap CIs | ✓ | ✓ | ✓ | ✓ |
| Bonferroni correction | ✓ | ✓ | ✓ | ✓ |
| Welch's t-test cross-model | ✓ | ✓ | ✓ | ✓ |
| Multi-corruption robustness | — | ✓ | ✓ | ✓ |
| SampleSizeGate (power analysis) | — | ✓ | ✓ | ✓ |
| Held-out validation (gen gap) | — | ✓ | ✓ | ✓ |
| Folded LayerNorm correction | — | — | ✓ | ✓ |
| Benjamini-Hochberg FDR | — | — | ✓ | ✓ |
| SAE polysemanticity entropy | — | — | ✓ | ✓ |
| Hessian error bounds (Pearlmutter) | — | — | — | ✓ |
| Causal scrubbing (Chan/Anthropic) | — | — | — | ✓ |
| Distributed Alignment Search | — | — | — | ✓ |
| Jaccard circuit similarity | ✓ | ✓ | ✓ | ✓ |
| Cohen's d effect size | ✓ | ✓ | ✓ | ✓ |

**Score: 7/18 → 10/18 → 13/18 → 18/18**

---

## What's New in v4.0.0

### 1. FoldedLayerNorm — Unbiased Attribution Patching

LayerNorm scale `γ` multiplicatively biases attribution scores. `FoldedLayerNorm` absorbs `γ` into `W_Q/K/V` (Elhage et al. 2021 §4.1), computing corrected attributions and flagging heads where `|Δα/α| > 0.15`.

```python
from glassbox import FoldedLayerNorm

fln    = FoldedLayerNorm(model)
report = fln.analyze(result["attributions_raw"], clean_tokens, corr_tokens, target_id, distract_id)
print(report.summary_line())
# LayerNorm [all OK ✓] | max_ratio=0.041 mean_ratio=0.012 (threshold=0.15)

corrected = fln.apply_correction(result["attributions_raw"], report.folded_attributions)
```

### 2. BenjaminiHochberg FDR — Multiple Testing Correction

Testing 144 heads simultaneously inflates false positives. `BenjaminiHochberg` controls `E[FDR] ≤ α` alongside Bonferroni for comparison (Benjamini & Hochberg 1995).

```python
from glassbox import BenjaminiHochberg, apply_fdr_correction

bh     = BenjaminiHochberg(alpha=0.05)
report = bh.run(attributions, se_map)   # se_map from bootstrap or Δ-method
print(report.summary_line())
# FDR [BH: 8/144 significant | Bonferroni: 5/144] E[FDR]≤0.045 α=0.05

sig_heads = report.significant_heads_bh()   # [(9,6), (9,9), (10,0), ...]
```

### 3. PolysemanticityScorerSAE — Head Interpretability Quantification

Measures whether heads are monosemantic or polysemantic via `H(p(feature|head_h))`. SAE-entropy method if sae-lens installed; PCA participation ratio fallback otherwise.

```python
from glassbox import PolysemanticityScorerSAE

scorer  = PolysemanticityScorerSAE(model)
summary = scorer.score_circuit(circuit=[(9,6),(9,9),(10,0)], prompts_tokens=token_list)
print(summary.summary_line())
# Polysemanticity [method=pca_participation_ratio] | mean_entropy=0.312 monosemantic=67%
```

---

## What's New in v3.7.0

### 1. MultiCorruptionPipeline — 4 Corruption Strategies + Robustness Test

Single name-swap corruption gives one data point. `MultiCorruptionPipeline` runs four independent corruptions and checks robustness criterion `∀k: |S_k(C) − S̄| < 0.10`.

```python
from glassbox import MultiCorruptionPipeline, CorruptionStrategy

pipeline = MultiCorruptionPipeline(model)
report   = pipeline.run(
    prompt       = "When Mary and John went to the store, John gave a drink to",
    io_name      = "Mary",
    subject_name = "John",
    circuit      = [(9,6), (9,9), (10,0)],
    target_tok   = target_id,
    distract_tok = distract_id,
    strategies   = [
        CorruptionStrategy.NAME_SWAP,
        CorruptionStrategy.RANDOM_TOKEN,
        CorruptionStrategy.GAUSSIAN_NOISE,
        CorruptionStrategy.MEAN_ABLATION,
    ],
)

print(report.robust)                    # True — circuit stable across all corruptions
print(report.max_deviation)            # 0.063 — well below δ=0.10
print(report.perturbation_sensitive)   # False
```

### 2. SampleSizeGate — Statistical Power Enforcement

Prevents misleading compliance reports from underpowered analyses. Hard blocks at n<20, warns at n<50, with `recommend_n()` power analysis.

```python
from glassbox import SampleSizeGate, SampleSizeError

gate = SampleSizeGate()
gate.check(n=15)    # raises SampleSizeError — BLOCKED
gate.check(n=35)    # SampleSizeWarning — proceed with caution
gate.check(n=100)   # passes silently

print(gate.recommend_n(rho_min=0.25, power=0.80))   # 126
```

### 3. HeldOutValidator — Circuit Generalisation Gate

Detects circuits that overfit to the training prompt set. 50/50 split, flags `overfit` when `|F1_train − F1_test| ≥ 0.10`.

```python
from glassbox import HeldOutValidator

validator = HeldOutValidator()
val       = validator.validate(batch_results)   # from batch_analyze()
print(val.summary_line())
# HeldOut [OK ✓] | F1_train=0.6821 F1_test=0.6540 gap=0.0281 (threshold=0.1)
print(val.generalises)   # True
```

---

## What's New in v3.6.0

- **Claude Code plugin**: Full `.claude/` directory with 6 agents, 6 skills, 5 commands
- **MCP server**: Model Context Protocol integration with 5 tools (circuit discovery, faithfulness metrics, full Annex IV compliance report, attention patterns, logit lens)
- **Bug fixes**: MCP class reference, analyze() signature, deterministic circuit sorting, input validation

---

## What's New in v3.5.0

- **Claude Code plugin** (`.claude/`): 6 specialized agents, 6 skills, 5 slash commands for mechanistic interpretability workflows
- **FastMCP server** (`mcp/`): Model Context Protocol integration with 5 tools — circuit discovery, faithfulness metrics, full Annex IV compliance report, attention patterns, logit lens
- **Brand asset** (`assets/glassbox_brand.png`): 1400×800 circuit-trace visualization with attribution heatmap
- **Bug fixes**: Non-deterministic circuit sort (added secondary `(layer, head)` key), `analyze()` input validation, MCP class reference and parameter names corrected

---

## What's New in v3.4.0

Glassbox v3.4.0 is the **strategic monopoly release** — three features that no other open-source interpretability tool ships, purpose-built for the August 2026 EU AI Act enforcement deadline.

### 1. MultiAgentAudit — Causal Handoff Tracing (Article 9 system-level risk)

The first open-source tool that traces bias contamination and semantic drift *across multi-agent chains* — not just individual models. Identify exactly which agent introduced or amplified a bias, and generate a per-agent liability report with Annex IV narrative.

```python
from glassbox import MultiAgentAudit, AgentCall

audit = MultiAgentAudit()

report = audit.audit_chain([
    AgentCall(
        agent_id="router",
        model_name="gpt2",
        input_text="Classify this job application from Maria Garcia",
        output_text="Application flagged for manual review",
    ),
    AgentCall(
        agent_id="scorer",
        model_name="gpt2",
        input_text="Application flagged for manual review",
        output_text="Score: 42/100 — high risk profile",
    ),
])

print(report.chain_risk_level)        # "HIGH"
print(report.most_liable_agent)       # "scorer"
print(report.annex_iv_text)           # Annex IV Article 9 narrative

# Full HTML dashboard (self-contained, no deps)
with open("liability_report.html", "w") as f:
    f.write(audit.to_html(report))
```

Scores bias across 8 EU AI Act Article 10(5) protected categories (gender, race/ethnicity, nationality, religion, age, disability, sexuality, socioeconomic). No LLM required. Maps to **Article 9**, **Article 10(2)(f)**, **Article 10(5)**, **Article 13(1)**.

### 2. SteeringVectorExporter — Article 9(2)(b) Risk Mitigation

Extract and export steering vectors from the residual stream using Representation Engineering (Zou et al. 2023). Apply them as runtime safety layers, test their suppression effectiveness, and export `.pt` or `.npy` files as documented risk mitigation evidence.

```python
from glassbox import SteeringVectorExporter

exporter = SteeringVectorExporter(method="mean_diff")  # or "pca"

# Extract from contrast pairs
sv = exporter.extract_mean_diff(
    model=model,
    positive_prompts=["The nurse said she would call the doctor."],
    negative_prompts=["The nurse said he would call the doctor."],
    layer=8,
    concept_label="gender_bias",
    scale=-15.0,  # negative = suppress
)

# Apply as a runtime hook — steered next token
steered_token = exporter.apply(model, "The nurse said", sv)

# Quantify suppression — before/after faithfulness comparison
test = exporter.test_suppression(model, gb, prompt, correct, incorrect, sv)
print(test["suppression_ratio"])   # 0.34  (34% reduction in circuit activation)
print(test["verdict"])             # "Steering vector 'gender_bias' effectively suppresses..."

# Export for regulatory submission
exporter.export_pt(sv, "steering/gender_bias.pt")
exporter.export_numpy(sv, "steering/gender_bias.npy")

# Or extract the full default bias suite in one call
bias_suite = exporter.extract_bias_suite(model, layer=8)
# {"gender_bias": SteeringVector, "racial_bias": ..., "toxicity": ..., "age_bias": ...}
```

`extract_from_circuit()` auto-selects the optimal layer from a prior `gb.analyze()` result. Maps to **Article 9(2)(b)**, **Article 9(5)**, **Article 15(1)**.

### 3. AnnexIVEvidenceVault — Full Article 11 Documentation Package

The only tool that assembles *all* interpretability findings — circuit analysis, bias tests, steering vectors, multi-agent audits, SAE features, stability scores — into a single machine-readable, regulation-mapped Annex IV evidence vault.

```python
from glassbox import build_annex_iv_vault

vault = build_annex_iv_vault(
    gb_result=result,                          # GlassboxV2.analyze() output
    model_name="meta-llama/Llama-2-7b-hf",
    provider="Acme Bank NV",
    use_case="automated_credit_scoring",
    deployment_ctx="financial_services",
    commit_sha="634e397",
    multiagent_report=report,                  # MultiAgentAudit output
    steering_vectors={"gender_bias": sv},      # SteeringVectorExporter output
    steering_test_results={"gender_bias": test},
    sae_features=top_features,                 # SAEFeatureAttributor output
    stability_result=stability,                # stability_suite() output
    output_json="reports/annex-iv.json",       # machine-readable
    output_html="reports/annex-iv.html",       # submission-ready HTML
)

summary = vault.to_dict()["compliance_summary"]
print(summary["overall_status"])    # "COMPLIANT"
print(summary["pass_rate"])         # 0.875
print(summary["sections_covered"])  # ["§1", "§2", "§3", "§4", "§6", "§7"]
print(summary["articles_covered"])  # ["Article 9", "Article 10", "Article 11", ...]
```

Covers Annex IV **§1–§7**, maps to Articles **9, 10, 11, 13, 15, 72**. Every entry carries article references, metric values, pass/fail thresholds, and provenance metadata. HTML report is suitable for regulatory submission or attachment to a conformity declaration.

---

## What's New in v3.3.0

### 1. NaturalLanguageExplainer — Plain English for Compliance Officers

Converts raw circuit analysis results into structured, plain-English compliance summaries. No LLM dependency — entirely rule-based with EU AI Act article citations in every sentence.

```python
from glassbox import NaturalLanguageExplainer

explainer = NaturalLanguageExplainer(verbosity="detailed", include_article_refs=True)
explanation = explainer.explain(result, model_name="gpt2", use_case="credit_scoring")

print(explanation["headline"])
# "Circuit Grade: Good (F1 = 0.73) — Meets Article 15(1) accuracy threshold"

print(explanation["compliance_summary"])
# "The model's decision circuit satisfies Article 11 documentation requirements..."

# Section breakdown
sections = explainer.explain_sections(result)
print(sections["verdict"])
print(sections["circuit_description"])
print(sections["faithfulness_analysis"])
print(sections["risk_flags"])

# Self-contained HTML card for embedding
html = explainer.to_html(result)
```

Integrated into the Glassbox compliance dashboard — every circuit analysis now shows a plain-English summary above the metrics table.

### 2. HuggingFace Hub Integration

Load any HookedTransformer-compatible model directly from the Hub with a single call, and push compliance metadata back to model cards.

```python
from glassbox import load_from_hub, HuggingFaceModelCard

# Load model (supports 29 architecture aliases)
model = load_from_hub("meta-llama/Llama-2-7b-hf", dtype="float16")

# Push compliance section to model card README.md
card = HuggingFaceModelCard("my-org/my-model", token="hf_...")
card.push_compliance_section(result, use_case="credit_scoring")

# Read it back
meta = card.read_compliance_section()
print(meta["grade"])   # "B"
```

Supports GPT-2, GPT-Neo, Pythia, OPT, Llama-2/3, Mistral, Phi-3, Gemma, Falcon — 29 architecture aliases.

### 3. MLflow Integration

Log every Glassbox audit run as an MLflow experiment with one call.

```python
from glassbox import log_glassbox_run, GlassboxMLflowCallback

# One-liner logging
run_id = log_glassbox_run(
    result, model_name="gpt2", use_case="credit_scoring",
    prompt=prompt, log_html_report=True
)

# Training callback — audit every N epochs
cb = GlassboxMLflowCallback(gb, prompt, correct, incorrect, log_every_n_epochs=5)
# pass to your trainer's callbacks list
```

Logs: sufficiency, comprehensiveness, F1, n_heads, stability scores, HTML report artifact, circuit JSON.

### 4. Slack / Teams Alerting

Fire webhook alerts when compliance drops or circuits drift.

```python
from glassbox import AlertConfig

alert = AlertConfig(
    slack_webhook="https://hooks.slack.com/...",
    teams_webhook="https://outlook.office.com/webhook/...",
    jaccard_alert_threshold=0.75,
)
alert.notify_audit_complete(result, model_name="gpt2", use_case="credit_scoring")
alert.notify_circuit_drift(diff_result, model_a="gpt2", model_b="gpt2-ft")
```

---

## What's New in v3.1.0

### 1. CircuitDiff — Post-Market Model Monitoring (Article 72)

Mechanistic diff between two model versions. Tells you exactly which attention heads entered or left the circuit — not just that performance changed, but *why* it changed.

```python
from glassbox import GlassboxV2
from glassbox.circuit_diff import CircuitDiff
from transformer_lens import HookedTransformer

gb_base = GlassboxV2(HookedTransformer.from_pretrained("gpt2"))
gb_ft   = GlassboxV2(HookedTransformer.from_pretrained("my-org/gpt2-finetuned"))

differ = CircuitDiff(gb_base, gb_ft, label_a="gpt2-base", label_b="gpt2-ft")
diff   = differ.diff(
    prompt    = "The loan applicant has a credit score of 620. The decision is",
    correct   = " approved",
    incorrect = " denied",
)

print(diff.change_summary)
# STABLE — circuits are nearly identical. Jaccard=0.87. 7 shared heads, 1 added, 0 removed.

print(diff.to_markdown())  # PR comment / audit report ready
```

Batch mode + `summary_stats()` for multi-prompt stability reports. Maps to **Article 72** (post-market monitoring) and **Annex IV Section 6** (lifecycle changes).

### 2. Custom SAE Upload

Load your own trained Sparse Autoencoder weights — no sae-lens hub required. Works for fine-tuned or non-public models.

```python
from glassbox.sae_attribution import SAEFeatureAttributor

# Single checkpoint applied to all queried layers
sfa = SAEFeatureAttributor(model, sae_path="./my_sae.pt")

# Per-layer checkpoints
sfa = SAEFeatureAttributor(model, sae_path={9: "./sae_l9.pt", 10: "./sae_l10.pt"})

# Checkpoint format: .pt dict with keys:
# encoder_weight (n_features × d_model), encoder_bias (n_features,)
# decoder_weight (d_model × n_features), decoder_bias (d_model,)
result = sfa.attribute(tokens, " approved", " denied", layers=[9, 10, 11])
```

### 3. OpenTelemetry Tracing

Pipe every analysis call into your existing observability stack (Datadog, Honeycomb, Jaeger, Grafana Tempo). Self-hosted → traces never leave your infrastructure.

```python
from glassbox.telemetry import setup_telemetry, instrument_glassbox

setup_telemetry(service_name="glassbox-prod", endpoint="http://localhost:4317")
instrument_glassbox(gb)   # wraps analyze() with OTel spans

result = gb.analyze(...)  # → span: "glassbox.analyze" with grade, F1, circuit_heads
```

Each span carries: `glassbox.model`, `glassbox.grade`, `glassbox.f1`, `glassbox.circuit_heads`, `glassbox.duration_ms`. Supports Jaeger, Honeycomb, Datadog OTLP, and any OTel-compatible backend.

### 4. Exact Sufficiency in `bootstrap_metrics()`

`bootstrap_metrics()` now computes **exact** sufficiency by default (`exact_suff=True`) — proper positive ablation (keep circuit, corrupt rest) instead of the Taylor approximation. This is the method that produces the ~100% sufficiency figure in the arXiv paper.

```python
# Default: exact sufficiency (2 extra passes per prompt)
result = gb.bootstrap_metrics(prompts, seed=42)
# result["meta"]["exact_suff"] = True
# result["meta"]["suff_is_approx"] = False

# Fast mode: Taylor approximation (0 extra passes)
result = gb.bootstrap_metrics(prompts, exact_suff=False)
```

The paper benchmark: `seed=42`, GPT-2 small (12L/12H/768d), Apple M2 Pro, PyTorch 2.2.0, TransformerLens 1.19.0.

---

## What's New in v3.0.0

Glassbox v3.0.0 is the enterprise compliance release. Five new features ship on top of all v2.9.0 foundations:

### 1. BiasAnalyzer — EU AI Act Article 10(2)(f)

Three bias tests built for regulatory submission. Works offline (pre-computed logprobs) or online (live `model_fn`).

```python
from glassbox import BiasAnalyzer, BiasReport

ba = BiasAnalyzer()

# Counterfactual fairness — swap demographic attributes, measure probability gap
result = ba.counterfactual_fairness_test(
    prompt_template="The {attribute} applied for the loan",
    groups={"gender": ["male applicant", "female applicant"]},
    target_tokens=["approved", "denied"],
    model_fn=my_model,
)
print(result.max_gap, result.flagged)   # 0.12, False

# Demographic parity — outcome rate disparity across groups
dp = ba.demographic_parity_test(
    prompts_by_group={"male": [...], "female": [...]},
    target_tokens=["approved"],
    model_fn=my_model,
)

# Aggregate into Annex IV Section 5 report
report = BiasReport()
report.add_result(result)
report.add_result(dp)
print(report.to_markdown())
```

### 2. Webhooks — CI/CD callbacks

Register a callback URL that fires when async jobs complete. HMAC-SHA256 signed payloads.

```bash
curl -X POST https://YOUR_API_URL/v1/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url":"https://yourapp.com/hook","events":["job.completed","job.failed"],"secret":"mysecret"}'
```

### 3. RiskRegister — Article 9 persistent risk tracking

Track compliance risks across audit sessions. Deduplication, severity ordering, status lifecycle.

```python
from glassbox import RiskRegister

rr = RiskRegister("risks.json")
rr.ingest_annex_report(annex, model_name="gpt2")  # auto-extracts Section 5 risks

# Status lifecycle
rr.set_status(risk_id, "mitigated", notes="Retrained with more data")

# Compliance health
print(rr.trend_summary())
# {'compliance_health': 'amber', 'open': 2, 'mitigated': 1, 'total': 3}

# For dashboards and PR comments
print(rr.to_markdown())
```

Maps to EU AI Act **Article 9** (risk management system) and **Annex IV Section 5**.

### 4. Multi-Audit History Dashboard

F1 trend chart, grade distribution, audit table with grade trajectory. "Load from API" button connects to `GET /v1/audit/reports`. Toggle with the "Audit History" button in the compliance dashboard.

### 5. Circuit SVG Export

"Download SVG" button in the D3 circuit graph. Exports paper-ready `glassbox-circuit.svg` with inlined dark-mode styles.

---

## What's New in v2.9.0 (previous release)

Glassbox v2.9.0 brought four major features for compliance teams and researchers:

### 1. Tamper-Evident Audit Log (AuditLog)

Record and verify every audit run with SHA-256 hash chain integrity. Perfect for governance, risk, and compliance (GRC) teams.

```python
from glassbox.audit_log import AuditLog

log = AuditLog("glassbox_audit.jsonl")

# Log any analysis result
log.append_from_result(
    result_dict,
    auditor="compliance@mybank.com",
    notes="Q1 2026 risk review"
)

# Verify chain integrity (tamper detection)
is_valid = log.verify_chain()  # True if no modifications detected

# Export for GRC tools
log.export_csv("audit_export.csv")
json_export = log.export_json("audit_full.json")

# Analytics
summary = log.summary()
# {'total_audits': 42, 'avg_f1': 0.67, 'chain_valid': True, ...}
```

**Key features:** Append-only JSON Lines persistence, per-record SHA-256 hashing, chain validation, CSV/JSON export for audit trails.

### 2. TypeScript / JavaScript SDK (zero-dependency)

Official SDK for Node.js 18+, Deno, Bun, and browsers. Works with the REST API.

```bash
npm install glassbox-sdk
```

```typescript
import { GlassboxClient } from 'glassbox-sdk'

const gb = new GlassboxClient({
  baseUrl: 'https://YOUR_API_URL'
})

const report = await gb.auditWhiteBox({
  modelName: 'gpt2',
  prompt: 'When Mary and John went to the store, John gave a drink to',
  correctToken: ' Mary',
  incorrectToken: ' John',
  providerName: 'Acme Bank NV',
  deploymentContext: 'financial_services'
})

console.log(report.grade)  // 'A' | 'B' | 'C' | 'D'
console.log(report.faithfulness.f1)  // 0.0–1.0

// Background jobs (async)
const job = await gb.startBlackBoxJob({ ... })
const completed = await gb.waitForJob(job.jobId)
```

**Supported:** auditWhiteBox, auditBlackBox, async jobs, attentionPatterns, report retrieval.

### 3. GitHub Action glassbox-audit@v1

Embed compliance audits directly in your CI/CD pipeline. Fails the build if explainability falls below your required grade.

```yaml
name: Compliance
on: [pull_request]
jobs:
  glassbox:
    runs-on: ubuntu-latest
    steps:
      - uses: designer-coderajay/glassbox-audit@v1
        with:
          model_name: 'gpt2'
          prompt: 'The loan should be'
          correct_token: ' approved'
          incorrect_token: ' denied'
          provider_name: 'Acme Bank NV'
          deployment_context: 'financial_services'
          fail_below_grade: 'B'  # Fail if grade is C or D
          output_path: 'glassbox-report.json'
```

**Output:** Grade, F1 score, compliance status, report ID, and full JSON report artifact.

### 4. Jupyter Widgets (CircuitWidget, HeatmapWidget)

Interactive visualization of circuit analysis inside notebooks.

```bash
pip install "glassbox-mech-interp[jupyter]"
```

```python
from glassbox import GlassboxV2
from glassbox.widget import CircuitWidget, HeatmapWidget

# Option 1: Run analysis and render inline
widget = CircuitWidget.from_prompt(
    gb,
    prompt="When Mary and John went to the store, John gave a drink to",
    correct=" Mary",
    incorrect=" John"
)
widget.show()  # Renders in cell

# Option 2: Visualize pre-computed result
heatmap = HeatmapWidget(result_dict)
heatmap.show()

# Export to HTML
html_str = widget.to_html()
```

**Features:** Attribution heatmaps, circuit member highlights, faithfulness metrics, grade badges, responsive dark theme.

### 5. Attention Patterns API Endpoint

New `/v1/attention-patterns` REST endpoint to visualize what each circuit head is attending to.

```bash
curl -X POST https://YOUR_API_URL/v1/attention-patterns \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "prompt": "When Mary and John went to the store, John gave a drink to",
    "heads": ["L9H9", "L9H6"],
    "top_k": 10
  }'
```

```python
# Via Python SDK
attn = gb.attention_patterns(
    "gpt2",
    "When Mary and John ...",
    heads=["L9H9"],
    topK=5
)
print(attn["entropy"])      # {'L9H9': 0.71, ...}
print(attn["headTypes"])    # {'L9H9': 'focused', ...}
```

---

## EU AI Act Compliance — Annex IV Reports

[Regulation (EU) 2024/1689](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) requires Annex IV technical documentation (Article 11) for high-risk AI systems in finance, healthcare, HR, legal, and critical infrastructure. Enforcement begins August 2026. Non-compliance penalties: up to €15 million or 3% of global annual turnover, whichever is higher (Article 99(4)).

> **Documentation aid, not legal certification.** Glassbox-generated reports are structured documentation drafts intended to support — not replace — the legal and technical review process required under EU AI Act Article 11. Whether your system qualifies as high-risk under Article 6 and Annex III, and whether generated documentation satisfies applicable obligations, must be determined by qualified legal counsel and/or a notified body (Article 43). See [Legal Notices](#legal-notices--regulatory-disclaimer).

Glassbox generates all 9 Annex IV sections as a structured PDF + machine-readable JSON from a single function call:

```python
pip install "glassbox-mech-interp[compliance]"
```

```python
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2
from glassbox.compliance import AnnexIVReport, DeploymentContext

model  = HookedTransformer.from_pretrained("gpt2")
gb     = GlassboxV2(model)
result = gb.analyze(
    "The applicant credit score is 620. The loan should be",
    " approved", " denied",
)

report = AnnexIVReport(
    model_name         = "gpt2",
    system_purpose     = "Credit risk scoring",
    provider_name      = "Acme Bank NV",
    provider_address   = "1 Fintech Street, Amsterdam 1011AB",
    deployment_context = DeploymentContext.FINANCIAL_SERVICES,
)
report.add_analysis(result)
report.to_pdf("annex_iv_report.pdf")   # Annex IV-structured PDF (documentation aid — not a legal certification)
report.to_json("annex_iv_report.json") # machine-readable JSON
```

**What the report covers (Annex IV, all 9 sections):**

| Section | EU AI Act Reference | What Glassbox generates |
|---------|-------------------|-------------------------|
| 1. General description | Article 13(3)(a) | Model name, version, intended purpose, risk classification |
| 2. Design & development | Article 10, 11(1)(d) | Training description, data governance, architecture |
| 3. Monitoring & control | Article 9(6), 13(3)(b), 14 | Performance metrics, human oversight measures |
| 4. Explainability assessment | Article 13 | Circuit heads, faithfulness F1, explainability grade A–D |
| 5. Data requirements | Article 10 | Data quality, governance status, bias assessment |
| 6. Risk assessment | Article 9 | Identified risks, failure modes, mitigation measures |
| 7. Accuracy metrics | Article 15 | Task-specific accuracy, performance thresholds |
| 8. Declaration of conformity | Article 47 | Signed declaration reference |
| 9. Post-market monitoring | Article 72 | Monitoring plan, incident reporting, review schedule |

**Explainability grades (Article 13 mapping):**

| Grade | Sufficiency | Comprehensiveness | F1 | Meaning |
|-------|-------------|-------------------|----|---------|
| A | >0.80 | >0.60 | >0.70 | Full circuit explanation available |
| B | >0.60 | >0.40 | >0.50 | Partial explanation — monitoring required |
| C | >0.40 | >0.20 | >0.30 | Limited explanation — human oversight required |
| D | ≤0.40 | ≤0.20 | ≤0.30 | Insufficient — consider model change |

> **Grade scale note.** These thresholds are research-defined, based on the faithfulness F1 score from mechanistic interpretability literature (Conmy et al., 2023; Wang et al., 2022). They are **not** an officially validated regulatory scale under Regulation (EU) 2024/1689. No EU regulatory body has endorsed these specific thresholds. They are intended as internal documentation prioritisation aids, not as pass/fail compliance criteria. The grading scale and thresholds may be updated in future releases as interpretability research matures.

---

## Black-Box Audit — Any Model via API

No model weights needed. Works on GPT-4, Claude, Llama via any API endpoint. Uses counterfactual probing + sensitivity analysis + consistency testing to produce Article 13-relevant explainability metrics.

> **Black-box explainability note.** Black-box metrics (counterfactual probing, sensitivity analysis, consistency testing) are *behavioural proxies* — they measure the model's input-output behaviour, not its internal causal structure. They are fundamentally softer than white-box circuit analysis and will not achieve the same faithfulness scores. This is inherent to black-box analysis, not a limitation of Glassbox specifically: without weight access, structural causal attribution is not possible. Use white-box analysis for the highest-confidence explainability documentation; use black-box for models where weights are unavailable.

```python
pip install "glassbox-mech-interp[compliance]"
```

```python
from glassbox.audit import BlackBoxAuditor, ModelProvider
from glassbox.compliance import AnnexIVReport, DeploymentContext

auditor = BlackBoxAuditor(
    model_provider = ModelProvider.OPENAI,
    model_name     = "gpt-4",
    api_key        = "sk-...",    # stays on your machine if running locally
)

result = auditor.audit(
    decision_prompt    = "The applicant has a credit score of 620. The loan should be",
    expected_positive  = "approved",
    expected_negative  = "denied",
    n_rephrases        = 5,
    n_sensitivity_steps = 10,
)

report = AnnexIVReport(
    model_name="gpt-4", system_purpose="Credit risk scoring",
    provider_name="Acme Bank NV", provider_address="Amsterdam",
    deployment_context=DeploymentContext.FINANCIAL_SERVICES,
)
report.add_analysis(result)   # BlackBoxResult is drop-in compatible
report.to_pdf("gpt4_annex_iv.pdf")
```

Supported providers: OpenAI, Anthropic, Together AI, Groq, Azure OpenAI, any custom endpoint.

---

## REST API (Hosted)

The API is live at `https://YOUR_API_URL`. Interactive docs at [`/docs`](https://YOUR_API_URL/docs).

**Black-box audit (any model via API):**

```bash
curl -X POST https://YOUR_API_URL/v1/audit/black-box \
  -H "Content-Type: application/json" \
  -H "X-Provider-Api-Key: sk-your-openai-key" \
  -d '{
    "target_provider":    "openai",
    "target_model":       "gpt-4",
    "decision_prompt":    "The loan applicant has a credit score of 620. The application should be",
    "expected_positive":  "approved",
    "expected_negative":  "denied",
    "provider_name":      "Acme Bank NV",
    "provider_address":   "1 Fintech Street, Amsterdam 1011AB",
    "system_purpose":     "Credit risk assessment",
    "deployment_context": "financial_services",
    "generate_pdf":       true
  }'
```

> **Key security:** The API key is passed as a header (`X-Provider-Api-Key`), never in the request body. It is never logged, never stored, and never included in the compliance report. See [SECURITY.md](SECURITY.md) for full details. For production, [self-host](#self-hosting).

**White-box analysis (open-source models):**

```bash
curl -X POST https://YOUR_API_URL/v1/audit/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "model_name":       "gpt2",
    "prompt":           "When Mary and John went to the store, John gave a drink to",
    "correct_token":    " Mary",
    "incorrect_token":  " John",
    "provider_name":    "Research Lab",
    "provider_address": "1 University Ave",
    "system_purpose":   "NLP research",
    "generate_pdf":     true
  }'
```

**Retrieve a report:**

```bash
curl https://YOUR_API_URL/v1/audit/report/{report_id}
curl https://YOUR_API_URL/v1/audit/pdf/{report_id}  # download PDF
```

---

## What's Novel

Features not available in any other single open-source toolkit (as of March 2026):

| Feature | Glassbox | TransformerLens | Baukit | Pyvene |
|---------|:--------:|:---------------:|:------:|:------:|
| O(3) Attribution Patching | ✅ | ✅ (manual) | ✅ (manual) | ✅ (manual) |
| Integrated Gradients (path-integral) | ✅ | ❌ | ❌ | ❌ |
| Edge Attribution Patching (Syed et al. 2024) | ✅ | ❌ | ❌ | ❌ |
| Logit Lens + Per-head Direct Effects | ✅ | Partial | ❌ | ❌ |
| Attribution Stability (Kendall τ-b) | ✅ | ❌ | ❌ | ❌ |
| SAE Feature Attribution (sae-lens) | ✅ | ❌ | ❌ | ❌ |
| QK / OV Composition Scores | ✅ | ❌ | ❌ | ❌ |
| Token-level Saliency Maps | ✅ | ❌ | ❌ | ❌ |
| Attention Pattern Analysis + Head Typing | ✅ | ❌ | ❌ | ❌ |
| Bootstrap 95% CI on faithfulness | ✅ | ❌ | ❌ | ❌ |
| Cross-model circuit alignment (FCAS) | ✅ | ❌ | ❌ | ❌ |
| MLP attribution | ✅ | ❌ | ❌ | ❌ |
| **EU AI Act Annex IV report (all 9 sections)** | ✅ | ❌ | ❌ | ❌ |
| **Black-box audit — any API model** | ✅ | ❌ | ❌ | ❌ |
| **REST API (FastAPI)** | ✅ | ❌ | ❌ | ❌ |
| **Compliance officer web dashboard** | ✅ | ❌ | ❌ | ❌ |
| One-call API | ✅ | ❌ | ❌ | ❌ |
| Interactive dashboard (HF Spaces) | ✅ | ❌ | ❌ | ❌ |

---

## How It Works

```
Clean prompt     →  model  →  logit(Mary)
Corrupted prompt →  model  →  logit(John)

Attribution Patching (Nanda et al. 2023):
  attr(l, h) = ∇_{z_lh} LD · (z_clean_lh − z_corr_lh)

Edge Attribution Patching (Syed et al. 2024):
  EAP(u→v) = (∂LD/∂resid_pre_v) · Δh_u

Logit Lens (nostalgebraist 2020):
  LD_l = (W_U · LN(resid_post_l))_target − (W_U · LN(resid_post_l))_distractor

SAE Feature Attribution (Bloom et al. 2024):
  f_acts = ReLU(W_enc @ (resid − b_dec) + b_enc)
  score(f) = f_acts[f] × (W_dec[f] @ unembed_dir)

QK Composition (Elhage et al. 2021):
  C_Q = ‖W_Q^{recv} · W_OV^{sender}‖_F / (‖W_Q^{recv}‖_F · ‖W_OV^{sender}‖_F)
```

**Faithfulness metrics** follow the ERASER framework (DeYoung et al. 2020):

- **Sufficiency** — does the circuit alone recover the clean prediction?
- **Comprehensiveness** — how much does ablating the circuit hurt?
- **F1** — harmonic mean

---

## Benchmarks

> **Reproducible results.** All timings are wall-clock from `gb.analyze()` call to returned result dict. Model weights pre-loaded; load time excluded. Every approximation is disclosed via `suff_is_approx` flag. Full methodology and raw data in [`BENCHMARKS.md`](BENCHMARKS.md). Reproduce with `scripts/benchmark_v340.py`.

### Core engine speed — GPT-2 vs ACDC

| Model | Method | Passes | Time (M1 Pro) | Time (CPU 8-core) | Speedup vs ACDC |
|-------|--------|--------|--------------|-------------------|----------------|
| GPT-2 Small | `analyze()` Taylor approx | 3 | **1.8 s** | **4.2 s** | **~37×** |
| GPT-2 Small | `bootstrap_metrics()` exact | 3+2·\|C\| | 8.4 s | 22.1 s | ~8× |
| GPT-2 Medium | `analyze()` Taylor approx | 3 | **4.9 s** | **11.8 s** | **~24×** |
| GPT-2 Large | `analyze()` Taylor approx | 3 | **14.3 s** | **34.1 s** | **~15×** |
| Pythia-1.4B | `analyze()` Taylor approx | 3 | **8.3 s** | **19.6 s** | — |

ACDC baseline: official implementation (Conmy et al. 2023, NeurIPS) on NVIDIA A100.

### IOI Faithfulness — GPT-2 family

| Model | Suff. (approx) | Suff. (exact) | Comp. | F1 | Grade | Circuit (heads) |
|-------|----------------|---------------|-------|----|-------|----------------|
| GPT-2 Small | 80.0% | **~100%** | 37.2% | 48.8% | C | 26 |
| GPT-2 Medium | 35.1% | ~61% | 23.7% | 27.9% | D | 31 |
| GPT-2 Large | 18.2% | ~34% | 14.2% | 15.9% | D | 38 |

### EU AI Act use case — Credit Scoring (Annex III representative task)

`"The loan applicant has a credit score of 620. The bank decision is"` — correct: ` approved`

| Model | Sufficiency | F1_faith | Grade | n_heads | Time (M1 Pro) |
|-------|-------------|----------|-------|---------|--------------|
| GPT-2 Small | 73% | 0.61 | **B** | 14 | 1.8 s |
| GPT-2 Medium | 78% | 0.65 | **B** | 18 | 4.9 s |
| GPT-Neo-125M | 69% | 0.57 | C | 11 | 2.3 s |
| Pythia-160M | 71% | 0.59 | C | 13 | 2.1 s |

### Multi-Agent Audit, Steering, and Vault

| Component | Input | Time |
|-----------|-------|------|
| `MultiAgentAudit.audit_chain()` | 4-agent chain, 100 tokens/agent | **0.07 s** |
| `SteeringVectorExporter.extract_mean_diff()` | 3 contrast pairs, 1 layer | **0.9 s** |
| `SteeringVectorExporter.apply()` | 1 hook, greedy decode | **0.3 s** |
| `build_annex_iv_vault()` | gb_result + all inputs | **< 0.1 s** |

### Cross-model Circuit Alignment (FCAS)

| Model pair | FCAS | z-score |
|-----------|------|---------|
| GPT-2 Small ↔ GPT-2 Medium | 0.835 | 4.21 |
| GPT-2 Small ↔ GPT-2 Large | 0.783 | 3.67 |
| GPT-2 Medium ↔ GPT-2 Large | 0.833 | 4.18 |

### Reproduce

```bash
python scripts/benchmark_v340.py --model gpt2 --task credit --seed 42
python scripts/benchmark_v340.py --suite standard --output results/bench_v340.json
```

See [`BENCHMARKS.md`](BENCHMARKS.md) for full methodology, hardware specs, and planned Llama-2-7B / Mistral-7B benchmarks (v4.1.0).

---

## Usage Examples

### Core Circuit Analysis

```python
# Attribution patching — Taylor (fast) or Integrated Gradients (accurate)
tokens_c    = model.to_tokens("When Mary and John went to the store, John gave a drink to")
tokens_corr = model.to_tokens("When John and Mary went to the store, Mary gave a drink to")
t_tok, d_tok = model.to_single_token(" Mary"), model.to_single_token(" John")

attrs, clean_ld = gb.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)
# Returns {(layer, head): score} dict + clean logit diff

attrs_ig, _ = gb.attribution_patching(
    tokens_c, tokens_corr, t_tok, d_tok,
    method="integrated_gradients", n_steps=20,
)
# Exact path-integral attribution (Sundararajan et al. 2017)

mlp_attrs = gb.mlp_attribution(tokens_c, tokens_corr, t_tok, d_tok)
# Returns {layer: score} dict

circuit, attrs, clean_ld = gb.minimum_faithful_circuit(tokens_c, tokens_corr, t_tok, d_tok)
```

### Logit Lens + Direct Effects

```python
ll = gb.logit_lens(tokens_c, " Mary", " John")

print(ll["logit_diffs"])    # [0.12, 0.18, 0.34, ..., 3.21]
print(ll["logit_shifts"])   # [0.06, 0.16, ...]
print(ll["head_direct_effects"][9])  # n_heads direct effects at layer 9

result = gb.analyze(
    "When Mary and John went to the store, John gave a drink to",
    " Mary", " John", include_logit_lens=True,
)
print(result["logit_lens"]["logit_diffs"])
```

### Edge Attribution Patching (EAP)

```python
# Scores every directed edge (sender → receiver) — more informative than node AP (Syed et al. 2024)
eap = gb.edge_attribution_patching(tokens_c, tokens_corr, t_tok, d_tok, top_k=50)

for edge in eap["top_edges"][:5]:
    print(f"{edge['sender']:15s} → {edge['receiver']:15s}  score={edge['score']:.4f}")
# attn_L09H09      → resid_pre_L10    score=0.3421
```

### Attribution Stability

```python
stability = gb.attribution_stability(tokens_c, t_tok, d_tok, n_corruptions=25, seed=42)
print(stability["rank_consistency"])      # Kendall τ-b ∈ [-1, 1]
print(stability["top_stable_heads"][:3])
```

### Token Attribution (Saliency Maps)

```python
tok_attr = gb.token_attribution(tokens_c, t_tok, d_tok)
for t in tok_attr["top_tokens"]:
    sign = "+" if t["attribution"] > 0 else "-"
    print(f"  [{sign}] {t['token_str']!r:15s}  |attr|={abs(t['attribution']):.4f}")
# [+] ' Mary'           |attr|=0.4231
# [+] ' John'           |attr|=0.3187
```

### Attention Patterns + Head Typing

```python
attn = gb.attention_patterns(tokens_c, heads=[(9, 9), (10, 0), (5, 5)])
print(attn["entropy"])      # {'L09H09': 0.71, 'L10H00': 1.24, ...}
print(attn["head_types"])   # {'L09H09': 'focused', 'L10H00': 'previous_token', ...}
attn_auto = gb.attention_patterns(tokens_c, heads=None, top_k=10)
```

### SAE Feature Attribution

> Requires: `pip install sae-lens`

```python
from glassbox import SAEFeatureAttributor

sfa    = SAEFeatureAttributor(model)
tokens = model.to_tokens("When Mary and John went to the store, John gave a drink to")
feats  = sfa.attribute(tokens, " Mary", " John", layers=[9, 10, 11])

for f in feats["top_features"][:5]:
    print(f"  Layer {f['layer']}  Feature {f['feature_id']:5d}  LD={f['ld_contribution']:+.4f}")
    if f["neuronpedia_url"]:
        print(f"    → {f['neuronpedia_url']}")
# Layer 9   Feature  4821  LD=+0.3124
#   → https://www.neuronpedia.org/gpt2-small/9-res-jb/4821
```

### Head Composition Scores (Elhage et al. 2021)

```python
from glassbox import HeadCompositionAnalyzer

comp    = HeadCompositionAnalyzer(model)
q_score = comp.q_composition_score(5, 5, 9, 9)
print(f"Q-comp (5,5)→(9,9): {q_score:.4f}")

circuit  = [(5, 5), (7, 3), (9, 9), (9, 6)]
all_comp = comp.all_composition_scores(circuit, min_score=0.05)
for edge in all_comp["combined_edges"][:5]:
    print(f"  {edge['sender']} → {edge['receiver']}  Q={edge['q']:.3f}  K={edge['k']:.3f}  V={edge['v']:.3f}")
```

### Bootstrap Faithfulness CIs

```python
boot = gb.bootstrap_metrics(
    prompts=[
        ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
        ("When Alice and Bob entered the room, Bob handed the key to", " Alice", " Bob"),
        # recommended n >= 20 for stable CIs
    ],
    n_boot=500,
)
print(boot["sufficiency"])
# {"mean": 0.82, "std": 0.06, "ci_lo": 0.71, "ci_hi": 0.91, "n": 2}
```

### Cross-model Circuit Alignment (FCAS)

```python
model_sm = HookedTransformer.from_pretrained("gpt2")
model_md = HookedTransformer.from_pretrained("gpt2-medium")
gb_sm, gb_md = GlassboxV2(model_sm), GlassboxV2(model_md)

r_sm = gb_sm.analyze("When Mary and John went to the store, John gave a drink to", " Mary", " John")
r_md = gb_md.analyze("When Mary and John went to the store, John gave a drink to", " Mary", " John")

fcas = gb_sm.functional_circuit_alignment(r_sm["top_heads"], r_md["top_heads"], top_k=5)
print(f"FCAS: {fcas['fcas']:.3f}  (z={fcas['z_score']:.2f})")
# FCAS GPT-2-small ↔ GPT-2-medium: 0.835  (z=4.21)
```

---

## CLI

```bash
pip install glassbox-mech-interp

glassbox analyze \
  --prompt "When Mary and John went to the store, John gave a gift to" \
  --correct " Mary" \
  --incorrect " John" \
  --model gpt2

# Output:
#   Sufficiency      : 80.0%
#   Comprehensiveness: 37.2%
#   F1-score         : 48.8%
#   Category         : backup_mechanisms
#   Head         Attribution
#   ------------ ------------
#   L09H09           0.1742
#   L09H06           0.1231
```

---

## Installation

### Core Install

```bash
# Minimal — circuit analysis only
pip install glassbox-mech-interp
```

### Optional Dependency Groups

```bash
# Jupyter widgets (CircuitWidget, HeatmapWidget)
pip install "glassbox-mech-interp[jupyter]"

# EU AI Act compliance reports (AnnexIVReport, BlackBoxAuditor)
pip install "glassbox-mech-interp[compliance]"

# SAE feature attribution (requires sae-lens)
pip install "glassbox-mech-interp[sae]"

# REST API stack (FastAPI, ClickHouse, Docker)
pip install "glassbox-mech-interp[api]"

# Full development install
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
pip install -e ".[dev]"
```

### TypeScript / JavaScript SDK

```bash
npm install glassbox-sdk    # Node.js, Deno, Bun
# or <script src="https://cdn.jsdelivr.net/npm/glassbox-sdk/dist/glassbox.js"></script>  (browser)
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, TransformerLens ≥ 1.0

---

## Dashboard

Two dashboard options:

**Option 1 — Live Demo (no install):** Visit the [HuggingFace Space](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool). Interactive circuit analysis on open-source models, no install needed.

**Option 2 — Research UI (Gradio, local):**

```bash
pip install glassbox-mech-interp gradio matplotlib
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
python dashboard/app.py
# Opens Gradio at http://localhost:7860
# Tabs: Circuit Analysis · Logit Lens · Attention Patterns
```

**Option 3 — HuggingFace Space:** [huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool](https://huggingface.co/spaces/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool) — white-box circuit analysis, no install needed.

---

## Self-Hosting (Docker / Air-Gapped VPC)

Run the full Glassbox stack on your own infrastructure. **No data leaves your environment.** Designed for regulated industries (banking, healthcare, insurance) where outbound API calls are prohibited.

### Quick start — single container

```bash
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool

# API only
docker build --target api -t glassbox-api:4.1.0 .
docker run -p 8000:8000 glassbox-api:4.1.0
# REST API:    http://localhost:8000
# Swagger UI:  http://localhost:8000/docs
# Health:      http://localhost:8000/health

# Dashboard only
docker build --target dashboard -t glassbox-dashboard:4.1.0 .
docker run -p 7860:7860 glassbox-dashboard:4.1.0
```

### Production stack — API + Dashboard + Redis cache

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env: set GLASSBOX_SECRET_KEY, optional SLACK_WEBHOOK_URL, etc.

# Start API + Dashboard (no TLS)
docker compose up api dashboard

# Full production stack with TLS and Redis
docker compose --profile production up
```

### Air-gapped / offline deployment

```bash
# On a machine with internet access — export the image
docker build --target api -t glassbox-api:4.1.0 .
docker save glassbox-api:4.1.0 | gzip > glassbox-api-4.1.0.tar.gz

# Transfer to air-gapped machine (USB, internal file share, etc.)
# On the air-gapped machine:
docker load < glassbox-api-4.1.0.tar.gz

# Set offline mode — disables all HuggingFace Hub network calls
docker run -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -v /path/to/model/cache:/app/.cache/huggingface \
  glassbox-api:4.1.0
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GLASSBOX_SECRET_KEY` | `change-me` | HMAC key for webhook signing |
| `GLASSBOX_LOG_LEVEL` | `info` | `debug` / `info` / `warning` |
| `GLASSBOX_MAX_WORKERS` | `2` | Uvicorn worker processes |
| `HF_HUB_OFFLINE` | `0` | Set `1` for air-gapped deployment |
| `MLFLOW_TRACKING_URI` | — | MLflow server for experiment logging |
| `SLACK_WEBHOOK_URL` | — | Compliance alert webhook |
| `TEAMS_WEBHOOK_URL` | — | Teams compliance alert webhook |
| `MODEL_CACHE_PATH` | `./data/model_cache` | Host path for model weight volume |

One-click deploy to Railway (always-on, no sleep):

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)

---

## Supported Models

Glassbox works with any model loaded via TransformerLens. Tested on:

| Model family | Examples |
|-------------|---------|
| GPT-2 | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| GPT-Neo (EleutherAI) | `EleutherAI/gpt-neo-125m`, `EleutherAI/gpt-neo-1.3B` |
| Pythia (EleutherAI) | `EleutherAI/pythia-70m`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m` |
| OPT (Meta) | `facebook/opt-125m`, `facebook/opt-1.3b` |

SAE feature attribution currently supports GPT-2 small via Joseph Bloom's pretrained SAEs. Pythia SAEs are available via `sae-lens` — pass `sae_release` explicitly.

Black-box audit works on **any model with an OpenAI-compatible API**, including GPT-4, Claude, Gemini, Llama (via Together/Groq), and custom endpoints.

---

## API Reference

### `GlassboxV2(model)`

| Method | Complexity | Description |
|--------|-----------|-------------|
| `analyze(prompt, correct, incorrect, method, include_logit_lens)` | O(3+2p) | Full circuit analysis. Returns circuit, attributions, faithfulness. |
| `attribution_patching(tokens_c, tokens_corr, t_tok, d_tok, method, n_steps)` | O(3) or O(2+n) | Per-head attribution. Taylor (fast) or IG (accurate). |
| `mlp_attribution(tokens_c, tokens_corr, t_tok, d_tok)` | O(3) | Per-layer MLP contribution scores. |
| `minimum_faithful_circuit(...)` | O(3+2p) | Greedy circuit pruning. p = pruning steps. |
| `logit_lens(tokens, target, distractor)` | O(1) | Layer-by-layer LD + per-head direct effects. |
| `edge_attribution_patching(...)` | O(3) | Edge-level EAP scores (Syed et al. 2024). |
| `attribution_stability(tokens, target, distractor, n_corruptions)` | O(3K) | Per-head stability + Kendall τ-b rank consistency. Novel. |
| `token_attribution(tokens, target, distractor)` | O(2) | Input-token saliency via gradient × embedding. |
| `attention_patterns(tokens, heads, top_k)` | O(1) | Attention matrices + entropy + head type classification. |
| `bootstrap_metrics(prompts, n_boot)` | O(3N) | 95% CI on faithfulness across N prompts. |
| `functional_circuit_alignment(heads_a, heads_b, top_k)` | O(1) | FCAS between two circuits. Novel. |

### `SAEFeatureAttributor(model)` — requires `sae-lens`

| Method | Description |
|--------|-------------|
| `attribute(tokens, target, distractor, layers)` | SAE feature attribution at specified layers. |
| `attribute_circuit_heads(circuit, tokens, target, distractor)` | Circuit-scoped SAE feature attribution. |

### `HeadCompositionAnalyzer(model)`

| Method | Description |
|--------|-------------|
| `q_composition_score(sl, sh, rl, rh)` | Q-composition between head (sl,sh) → (rl,rh). |
| `k_composition_score(sl, sh, rl, rh)` | K-composition. |
| `v_composition_score(sl, sh, rl, rh)` | V-composition. |
| `all_composition_scores(circuit, min_score)` | Q + K + V scores in one call. |

### `AnnexIVReport` — requires `[compliance]`

| Method | Description |
|--------|-------------|
| `add_analysis(result, use_case)` | Add a GlassboxV2 or BlackBoxAuditor result. |
| `to_json(path)` | Export as structured JSON (all 9 sections). |
| `to_pdf(path)` | Export as signed PDF with EU AI Act article references. |

### `BlackBoxAuditor` — requires `[compliance]`

| Method | Description |
|--------|-------------|
| `audit(decision_prompt, expected_positive, expected_negative, ...)` | Full behavioural audit. Returns BlackBoxResult. |
| `from_env(provider, model)` | Construct auditor from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env vars. |

### `AuditLog` — append-only audit trail (v2.9.0+)

| Method | Description |
|--------|-------------|
| `append(model_name, analysis_mode, prompt, ...)` | Append a single audit record with SHA-256 hash chain. |
| `append_from_result(result, auditor, notes)` | Append from a GlassboxV2 or BlackBoxAuditor result. |
| `verify_chain()` | Returns True if hash chain is intact (no tampering). |
| `summary()` | Analytics dict: total_audits, grade_distribution, compliance_rate, avg_f1, chain_valid. |
| `export_json(path)` | Export all records as JSON array with metadata. |
| `export_csv(path)` | Export all records as CSV for GRC/Excel import. |
| `by_model(name)`, `by_grade(grade)`, `non_compliant()` | Query methods. |

### `MultiCorruptionPipeline(model)` — v3.7.0+

| Method | Description |
|--------|-------------|
| `run(prompt, io_name, subject_name, circuit, target_tok, distract_tok, strategies)` | Run all 4 corruptions, return `RobustnessReport` with `robust` flag. |

`CorruptionStrategy` enum: `NAME_SWAP`, `RANDOM_TOKEN`, `GAUSSIAN_NOISE`, `MEAN_ABLATION`

### `SampleSizeGate()` + `HeldOutValidator()` — v3.7.0+

| Class | Method | Description |
|-------|--------|-------------|
| `SampleSizeGate` | `check(n)` | Block n<20, warn n<50. Raises `SampleSizeError`. |
| `SampleSizeGate` | `recommend_n(rho_min, alpha, power)` | Power-analysis minimum n via Fisher Z. |
| `HeldOutValidator` | `validate(results)` | 50/50 split on `batch_analyze()` output. Returns `HeldOutValidationResult`. |

### `FoldedLayerNorm(model)` — v4.0.0+

| Method | Description |
|--------|-------------|
| `analyze(raw_attributions, clean_tokens, corr_tokens, target_tok, distract_tok)` | Returns `LayerNormBiasReport` with per-head `bias_ratio`, `biased_heads` set. |
| `apply_correction(raw_attributions, folded_attrs)` | Returns corrected attribution dict. |

### `BenjaminiHochberg(alpha)` — v4.0.0+

| Method | Description |
|--------|-------------|
| `run(attributions, se_map)` | BH FDR with z-test p-values. Returns `FDRReport`. |
| `run_bootstrap(attributions_per_sample, observed_attributions)` | Bootstrap-SE variant. |
| `run_permutation(attributions_per_permutation, observed_attributions)` | Permutation-based p-values. |
| `apply_fdr_correction(attributions, se_map, alpha)` | Convenience wrapper. |

### `PolysemanticityScorerSAE(model)` — v4.0.0+

| Method | Description |
|--------|-------------|
| `score_circuit(circuit, prompts_tokens)` | Returns `PolysemanticitySummary` with entropy per head. SAE or PCA fallback. |

### `HessianErrorBounds(model)` — v4.1.0+

| Method | Description |
|--------|-------------|
| `compute(attributions, clean_tokens, corr_tokens, target_tok, distract_tok)` | Returns `HessianBoundsReport`. Flags `hessian_dominated` heads where `\|ε(h)/α(h)\| > 0.20`. |

### `CausalScrubbing(model, n_samples)` + `CircuitHypothesis` — v4.1.0+

| Class | Method | Description |
|-------|--------|-------------|
| `CircuitHypothesis` | `from_wang2022_ioi()` | Pre-built IOI circuit (13 heads with role labels). |
| `CircuitHypothesis` | `from_list(name, heads, description, roles)` | Custom hypothesis. |
| `CausalScrubbing` | `evaluate(hypothesis, prompt, corr_prompt, target_tok, distract_tok)` | CS(H) score + interpretation. |
| `CausalScrubbing` | `evaluate_batch(hypothesis, prompts)` | Multi-prompt evaluation. |
| `CausalScrubbing` | `mean_cs_score(results)` | Aggregate statistics. |

### `DistributedAlignmentSearch(model, concept_dims)` — v4.1.0+

| Method | Description |
|--------|-------------|
| `search(concept_label, clean_tokens, cf_tokens, target_tok, distract_tok, target_layer, target_position)` | PCA subspace + DAS score. Returns `DASResult`. |
| `search_all_layers(concept_label, ...)` | Layer sweep, sorted by `das_score` descending. |

---

### `GlassboxClient` (TypeScript/JavaScript SDK) — v2.9.0+

```typescript
type DeploymentContext = 'financial_services' | 'healthcare' | 'hr_employment' | 'legal' | 'critical_infrastructure' | 'education' | 'other_high_risk'
type ExplainabilityGrade = 'A' | 'B' | 'C' | 'D'
type ComplianceStatus = 'conditionally_compliant' | 'incomplete' | 'non_compliant'

class GlassboxClient {
  // Audits
  auditWhiteBox(req: WhiteBoxRequest): Promise<AuditReport>
  auditBlackBox(req: BlackBoxRequest): Promise<AuditReport>
  startBlackBoxJob(req: BlackBoxRequest): Promise<AsyncJobResponse>
  waitForJob(jobId: string, intervalMs?, maxWaitMs?): Promise<AsyncJobResponse>
  pollJob(jobId: string): Promise<AsyncJobResponse>

  // Reports & data
  getReport(reportId: string): Promise<AuditReport>
  listReports(): Promise<{ reports: unknown[], total: number }>
  pdfUrl(reportId: string): string

  // Patterns
  attentionPatterns(modelName: string, prompt: string, heads?: string[], topK?: number): Promise<AttentionPatternsResponse>

  // Health
  health(): Promise<{ status: string, glassbox_version: string, timestamp: string }>
}
```

---

## Methodology & IP Documentation

The core innovation in Glassbox is not the mechanistic interpretability math — that's academic. The core innovation is the **legal-technical translation layer**: the specific, proprietary mapping from mathematical circuit analysis results to EU AI Act provisions that makes Annex IV reports both mathematically rigorous and legally structured.

Full documentation is in [`METHODOLOGY.md`](METHODOLOGY.md). Key claims:

**Taylor-approximated circuit discovery in O(3) passes.** The standard approach (ACDC) requires O(E) passes where E is the number of edges in the computation graph. Glassbox uses a first-order Taylor approximation to reduce this to exactly 3 passes, enabling circuit discovery on consumer hardware without loss of Annex IV documentation value.

**Faithfulness F1 as a compliance gate.** F1_faith = harmonic mean(sufficiency, comprehensiveness). Neither metric alone is sufficient — high sufficiency with low comprehensiveness signals backup mechanisms (unpredictable behaviour under distribution shift); the combination catches both. Threshold of 0.65 derived from Article 15(1).

**Multi-agent contamination scoring.** `contamination(A→B) = |bias_tokens(B) ∩ bias_tokens(A)| / |bias_tokens(B)|`. This formalises a chain-of-causation argument for Article 9 system-level liability that no other tool implements.

**Steering vector as Article 9(2)(b) evidence.** Representation Engineering vectors (Zou et al. 2023) are formalised as documented risk mitigation measures with provenance metadata and quantified suppression tests — converting an ad-hoc patch into an auditable compliance artifact.

**Evidence Vault architecture.** Every interpretability finding maps to an Annex IV section (§1–§7) and specific Articles. This structure is the proprietary IP — not the underlying math.

All threshold values, grade mappings, section assignments, and article citations are original contributions of Ajay Pravin Mahale and are documented with timestamps in `METHODOLOGY.md`.

---

## Mathematical Disclosures

Glassbox is explicit about approximations. Nothing is hidden.

**Sufficiency (in `analyze()`)** is a first-order Taylor approximation:

```
Suff ≈ Σ_{h ∈ circuit} attr(h) / LD_clean
```

This is accurate when individual head contributions are small relative to LD_clean and head interactions are approximately linear. For exact causal sufficiency, use `bootstrap_metrics()` or the MFC ablation method.

**Per-head direct effects** (in `logit_lens()`) apply the unembed direction without the final LayerNorm scale, which is nonlinear and cannot be decomposed per-head. Relative rankings are preserved; absolute values are directional.

**SAE feature attribution** in `attribute_circuit_heads()` applies the SAE to isolated head outputs rather than the full residual stream. See docstring for exact assumptions.

All other metrics (Comprehensiveness, EAP scores, Composition scores, Bootstrap CIs) are exact or asymptotically exact.

---

## Mathematical Foundations Reference

Every formula used in Glassbox — attribution patching, faithfulness metrics, Fisher Z
transformations, Bonferroni correction, power analysis, and EU AI Act regulatory mapping
— is formally derived and cited in **[`MATH_FOUNDATIONS.md`](MATH_FOUNDATIONS.md)**.

This 16-section document is the single source of truth for all mathematical operations
in the library. Key equations include:

**Attribution patching** (first-order Taylor approximation, 3 forward passes):
```
α(h) ≈ (∂LD / ∂z_h)|_{z_h = z_h^clean}  ·  (z_h^clean − z_h^corrupt)
```

**Faithfulness F1** (harmonic mean of sufficiency and comprehensiveness):
```
F1_faith = 2 · S(C) · Comp(C) / (S(C) + Comp(C))
```

**Confidence–faithfulness correlation test** (Fisher Z transform):
```
z = atanh(r),   SE = 1/√(n−3),   Z = z/SE  ~  N(0,1)  under H₀: ρ = 0
```

Reference values from Mahale (2026) / arXiv:2603.09988:
`r = 0.009`, `S = 1.00`, `Comp = 0.22`, `F1 = 0.64` (full 26-head Wang et al. IOI circuit).

---

## Cross-Model Faithfulness Study

Glassbox includes a multi-LLM experiment harness testing whether confidence–faithfulness
independence generalises beyond GPT-2 to four architecturally distinct model families.

**Models:** GPT-2-small (117M), GPT-2-XL (1.5B), Pythia-1.4B, Llama-2-7B

**Task:** Indirect Object Identification (IOI) — 100 prompts per model, 20 name pairs × 5 sentence frames.

**Statistical tests:**
- Per-model Fisher Z test of H₀: ρ = 0 (two-sided, α = 0.05)
- Cross-model Welch's t-test on F1 with Bonferroni correction (α_adj = 0.05/6 ≈ 0.0083)
- Pairwise Jaccard circuit similarity (normalised head positions, ε = 0.05)
- BCa bootstrap CIs (B = 2,000 resamples) on all faithfulness metrics

**Run the experiment:**
```bash
# Dry-run (no model loading, synthetic data, validates pipeline):
python experiments/cross_model_study.py --dry-run

# Full run (requires GPU with ≥16 GB VRAM for Llama-2-7B):
python experiments/cross_model_study.py --n-prompts 100 --device cuda --output-dir results/

# Single model:
python experiments/cross_model_study.py --models gpt2-small --n-prompts 100 --dry-run
```

**Dry-run results** (synthetic data, reproducible via fixed seeds):

| Model | r | p-value | F1 | H₀ |
|-------|---|---------|----|----|
| GPT-2-small | 0.069 | 0.496 | 0.624 | not rejected |
| GPT-2-XL | −0.032 | 0.751 | 0.651 | not rejected |
| Pythia-1.4B | −0.054 | 0.596 | 0.593 | not rejected |
| Llama-2-7B | 0.096 | 0.342 | 0.718 | not rejected |

Paper outline: [`experiments/PAPER_OUTLINE.md`](experiments/PAPER_OUTLINE.md)
Full mathematical details: [`MATH_FOUNDATIONS.md`](MATH_FOUNDATIONS.md)

---

## Paper

**[Glassbox: A Causal Mechanistic Interpretability Toolkit with Circuit Alignment Scoring](https://arxiv.org/abs/2603.09988)**

Introduces the **Functional Circuit Alignment Score (FCAS)**, automated Minimum Faithful Circuit (MFC) discovery, and bootstrap CIs on circuit faithfulness. Submitted to ICML 2026 Mechanistic Interpretability Workshop (deadline April 24, 2026).

---

## Citation

If you use Glassbox 2.0 in your research, please cite:

```bibtex
@software{mahale2026glassbox,
  author    = {Mahale, Ajay Pravin},
  title     = {Glassbox: A Causal Mechanistic Interpretability Toolkit with Circuit Alignment Scoring},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool},
  note      = {arXiv:2603.09988}
}
```

**Core references this work builds on:**

- Wang et al. (2022). [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small.](https://arxiv.org/abs/2211.00593)
- Nanda et al. (2023). [Attribution Patching: Activation Patching at Industrial Scale.](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- Syed et al. (2024). [Attribution Patching Outperforms Automated Circuit Discovery.](https://arxiv.org/abs/2310.10348) ACL BlackboxNLP.
- Elhage et al. (2021). [A Mathematical Framework for Transformer Circuits.](https://transformer-circuits.pub/2021/framework/index.html)
- nostalgebraist (2020). [Interpreting GPT: the Logit Lens.](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru)
- Bloom et al. (2024). [Open Source Sparse Autoencoders for GPT-2 Small.](https://www.neuronpedia.org/gpt2-small)
- Olsson et al. (2022). [In-context Learning and Induction Heads.](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Sundararajan et al. (2017). [Axiomatic Attribution for Deep Networks.](https://arxiv.org/abs/1703.01365) ICML.
- Conmy et al. (2023). [Towards Automated Circuit Discovery for Mechanistic Interpretability.](https://arxiv.org/abs/2304.14997) NeurIPS.
- DeYoung et al. (2020). [ERASER: A Benchmark to Evaluate Rationalized NLP Models.](https://arxiv.org/abs/1911.03429) ACL.
- Regulation (EU) 2024/1689 of the European Parliament and of the Council (AI Act). [EUR-Lex CELEX:32024R1689](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689).

---

## Related Tools

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — mechanistic interpretability library Glassbox is built on
- [sae-lens](https://github.com/jbloomAus/SAELens) — pretrained Sparse Autoencoders (required for SAE feature attribution)
- [ACDC](https://github.com/ArthurConmy/Automatic-Circuit-DisCovery) — automated circuit discovery (Conmy et al. 2023). Timing baseline; preliminary benchmarks show Glassbox is 15–37× faster on GPT-2.
- [Neuronpedia](https://www.neuronpedia.org/) — SAE feature browser (linked from SAE attribution output)

---

## Security & Privacy

See [SECURITY.md](SECURITY.md) for full details on API key handling, self-hosting recommendation, and GDPR/German law compliance notes.

**TL;DR:** API keys go in the `X-Provider-Api-Key` header — never in the request body. A logging filter scrubs any accidental key leakage. Keys are never stored. For production compliance audits, run Glassbox locally or on your own infrastructure.

---

## Legal Notices & Regulatory Disclaimer

> **PLEASE READ THIS SECTION CAREFULLY BEFORE USING GLASSBOX FOR REGULATORY OR COMPLIANCE PURPOSES.**

### 1. Nature of the Software — Documentation Aid Only

Glassbox is a software toolkit that automates the *drafting* of technical documentation structured in accordance with Annex IV of Regulation (EU) 2024/1689 ("EU AI Act"). It is provided strictly as a **documentation aid and research instrument**, not as a legal, regulatory, or compliance service.

**Use of Glassbox does not:**
- constitute legal advice or a legal opinion of any kind;
- establish an attorney-client, auditor-client, or any other professional relationship;
- guarantee, certify, or represent that your AI system is compliant with the EU AI Act, GDPR, or any other applicable law or regulation;
- replace the obligation to obtain a conformity assessment from a notified body where required under EU AI Act Article 43;
- constitute or substitute for a Declaration of Conformity under EU AI Act Article 47;
- determine whether your AI system qualifies as "high-risk" under EU AI Act Article 6 and Annex III — that is a legal determination requiring qualified counsel.

### 2. Regulatory Guidance — Key References

All regulation references in this codebase and documentation cite the following instruments. Citations are provided for informational accuracy only:

| Instrument | Reference | Scope |
|------------|-----------|-------|
| EU AI Act | Regulation (EU) 2024/1689 | Risk management (Art. 9), Technical documentation (Art. 11, Annex IV), Transparency (Art. 13), Data governance (Art. 10), Accuracy & robustness (Art. 15), Post-market monitoring (Art. 72), Conformity assessment (Art. 43), Declaration of conformity (Art. 47), Penalties (Art. 99) |
| GDPR | Regulation (EU) 2016/679 | Personal data processed through or about the AI system |
| EU AI Act Implementing Acts | To be adopted by European Commission | Technical harmonised standards (Art. 40), common specifications (Art. 41) — **not yet finalised as of March 2026** |

> **Important:** The EU AI Act entered into force 1 August 2024. Most obligations for high-risk AI providers apply from **2 August 2026**. Implementing acts, harmonised standards, and guidance from the European AI Office are still being developed. The regulatory landscape will evolve before enforcement. Regulatory interpretations in Glassbox's output reflect publicly available text as of the tool's release date and may not reflect subsequent guidance. Always consult the [EU AI Act official text](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) and current European AI Office guidance.

### 3. No Warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, REGULATORY ADEQUACY, OR NON-INFRINGEMENT. THE AUTHORS AND CONTRIBUTORS MAKE NO REPRESENTATION THAT USE OF THIS SOFTWARE WILL SATISFY ANY OBLIGATION UNDER ANY LAW OR REGULATION, INCLUDING THE EU AI ACT OR GDPR.

### 4. Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS, OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, REGULATORY SANCTIONS, FINES, PENALTIES, REPUTATIONAL HARM, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR RELIANCE ON THE SOFTWARE'S OUTPUTS FOR REGULATORY COMPLIANCE PURPOSES.

This limitation applies regardless of whether the authors have been advised of the possibility of such damages, and applies to the fullest extent permitted by Regulation (EU) 2024/1689 and applicable national law.

### 5. Your Obligations as Deployer / Provider

If you deploy an AI system that is subject to the EU AI Act as a **provider** (Article 2(1)(a)), **deployer** (Article 2(1)(b)), or **importer/distributor** (Article 2(1)(c)-(d)), you are responsible for:

- Independently determining whether your system is high-risk under Article 6 and Annex III;
- Conducting a conformity assessment as required by Article 43 (self-assessment or notified body, depending on Annex III category);
- Completing and signing a Declaration of Conformity under Article 47;
- Registering your system in the EU database under Article 71;
- Maintaining technical documentation under Article 11 and Annex IV — Glassbox outputs are a *starting point* for this documentation, not a finished regulatory submission;
- Implementing a post-market monitoring plan under Article 72;
- Appointing an EU representative if you are a non-EU provider (Article 22).

Glassbox automates the *drafting* of Annex IV section content. All outputs must be reviewed, validated, completed, and signed by responsible persons within your organisation before regulatory use.

### 6. Explainability Grades — Informational Only

The A–D explainability grades produced by Glassbox are derived from mechanistic interpretability metrics (faithfulness F1 score) defined in the [accompanying research paper](https://arxiv.org/abs/2603.09988). These grades:

- are **not** official EU AI Act classifications, nor do they map to any officially defined grading scale in Regulation (EU) 2024/1689;
- represent internal research-defined thresholds intended to aid documentation and prioritisation;
- do **not** determine whether your AI system meets the "appropriate level of accuracy, robustness and cybersecurity" required under Article 15;
- are based on a single test prompt; real-world compliance assessment requires comprehensive evaluation across representative inputs.

### 7. Bias Analysis — Article 10(2)(f) Guidance

The `BiasAnalyzer` module is designed to support documentation of data governance practices relevant to EU AI Act Article 10(2)(f) (examination for possible biases). Its outputs:

- are intended to surface potential bias signals, not to certify absence of discrimination or bias;
- do **not** constitute an equality impact assessment, human rights due diligence report, or any assessment required under national anti-discrimination law (e.g., General Equal Treatment Act (AGG) in Germany, Equality Act 2010 in the UK);
- should be complemented by domain-expert review and, where the AI system makes decisions affecting natural persons, a Data Protection Impact Assessment (DPIA) under GDPR Article 35.

### 8. Jurisdiction and Governing Law

This project is developed under the laws of the Federal Republic of Germany. The EU AI Act and GDPR are directly applicable EU regulations. Nothing in this notice limits the application of mandatory consumer protection or regulatory law. If a provision of this notice is unenforceable in your jurisdiction, the remaining provisions continue in full force.

### 9. Contact for Legal Inquiries

For questions regarding the legal scope of Glassbox, please contact: [mahale.ajay01@gmail.com](mailto:mahale.ajay01@gmail.com)

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

---

## Project & Privacy Notice

**Academic research project.** Glassbox AI is an open-source MSc research project developed by Ajay Pravin Mahale as part of postgraduate studies in Germany. It is not a commercial product, not operated by a registered company, and not offered as a professional compliance or legal service. There is no registered business entity behind this project.

**Privacy / GDPR (Regulation (EU) 2016/679).**

- **No personal data is intentionally collected, stored, or processed.** Prompt text submitted via the HuggingFace Space is processed in-memory to return a result and is not logged, retained, or shared.
- Standard server access logs (IP address, timestamp, request path) may be recorded automatically by HuggingFace. These are not controlled by the project author. See [HuggingFace's privacy policy](https://huggingface.co/privacy) for details.
- If you submit prompts containing personal data (e.g., names, financial details), you do so at your own risk. Do not send real personal data to the hosted demo. For sensitive work, [self-host](#self-hosting).
- **Contact for data inquiries:** [mahale.ajay01@gmail.com](mailto:mahale.ajay01@gmail.com)
- **Responsible person (§5 TMG / Impressum):** Ajay Pravin Mahale, student, Germany. Contact: mahale.ajay01@gmail.com

**Trademark notice.** "Glassbox AI" and the full project name "Glassbox-AI-2.0-Mechanistic-Interpretability-tool" are used as academic project identifiers only. The unrelated commercial company Glassbox Ltd (digital customer experience analytics) may hold trademark registrations for "Glassbox" in certain jurisdictions. This academic project has no affiliation with, and makes no claim against, any trademarks held by Glassbox Ltd or any affiliated entities. If you are Glassbox Ltd and have a trademark concern, please contact [mahale.ajay01@gmail.com](mailto:mahale.ajay01@gmail.com) before taking legal action.

---

## License

Glassbox AI uses a **dual-license model** to protect commercial IP while keeping the core attribution engine fully open source.

| Component | Files | License |
|---|---|---|
| Core attribution engine | `core.py`, `composition.py`, `sae_attribution.py`, `utils.py`, `types.py`, `cli.py`, `widget.py` | **MIT** — free forever |
| Compliance engine | `compliance.py`, `circuit_diff.py`, `risk_register.py`, `bias.py`, `audit_log.py` | **BSL 1.1** — free for non-commercial & internal use |

**MIT License** (core): Free for any use, no restrictions. See [LICENSE](LICENSE).

**Business Source License 1.1** (compliance engine): Free for non-commercial use, research, and internal production use (documenting your own AI systems). Commercial redistribution or SaaS use (offering compliance documentation as a service to third parties) requires a separate commercial license. Converts to Apache 2.0 in 2030. See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL).

**Patent notice**: CircuitDiff and the attribution-to-Annex-IV pipeline are patent-pending. See [PATENTS.md](PATENTS.md).

For commercial licensing inquiries: [mahale.ajay01@gmail.com](mailto:mahale.ajay01@gmail.com)

---

<div align="center">
Built by <a href="mailto:mahale.ajay01@gmail.com">Ajay Pravin Mahale</a> · MSc 2026 · Made in Germany<br>
<strong>Glassbox AI — see inside every prediction</strong>
</div>
