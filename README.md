<div align="center">

# Glassbox 3.0.0

**Open-source EU AI Act Annex IV compliance documentation toolkit. Works on any LLM.**

[![PyPI](https://img.shields.io/pypi/v/glassbox-mech-interp?color=blue&label=PyPI%20v3.0.0)](https://pypi.org/project/glassbox-mech-interp/)
[![Live Analytics](https://img.shields.io/badge/Live%20Analytics-ClickHouse-FFCC01?logo=clickhouse&logoColor=black)](https://clickpy.clickhouse.com/dashboard/glassbox-mech-interp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Render-success)](https://glassbox-ai-2-0-mechanistic.onrender.com)
[![arXiv](https://img.shields.io/badge/arXiv-2603.09988-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.09988)
[![Tests](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml)

[**Live Demo**](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) · [**Dashboard**](https://glassbox-ai-2-0-mechanistic.onrender.com/dashboard) · [**API**](https://glassbox-ai-2-0-mechanistic.onrender.com) · [**Docs**](https://glassbox-ai-2-0-mechanistic.onrender.com/docs) · [**Paper**](https://arxiv.org/abs/2603.09988) · [**PyPI**](https://pypi.org/project/glassbox-mech-interp/)

</div>

---

**For compliance teams:** Regulation (EU) 2024/1689 (AI Act) requires Annex IV technical documentation for every high-risk AI system (Article 11). Enforcement begins August 2026. Glassbox automates generation of the full 9-section Annex IV draft — from open-source models (white-box) or any proprietary API like GPT-4 and Claude (black-box). Outputs are structured documentation aids; they do not constitute legal advice, a declaration of conformity, or a guarantee of regulatory compliance. See [Legal Notices](#legal-notices--regulatory-disclaimer).

**For researchers:** one function call discovers the minimum faithful circuit in a transformer — the smallest subgraph of attention heads causally responsible for a prediction. Preliminary benchmarks show 15–37× faster than ACDC on GPT-2 (single-run, Apple M2 Pro — see [Benchmarks](#benchmarks)). Every approximation is disclosed.

---

## Table of Contents

- [Live Services](#live-services)
- [Quickstart](#quickstart)
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
- [Self-Hosting](#self-hosting)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
- [Mathematical Disclosures](#mathematical-disclosures)
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
| **Compliance Dashboard** | [/dashboard](https://glassbox-ai-2-0-mechanistic.onrender.com/dashboard) | Web UI for compliance officers. No install needed. |
| **REST API** | [glassbox-ai-2-0-mechanistic.onrender.com](https://glassbox-ai-2-0-mechanistic.onrender.com) | JSON API. See [/docs](https://glassbox-ai-2-0-mechanistic.onrender.com/docs) for Swagger UI. |
| **White-Box Demo** | [HuggingFace Space](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) | Interactive circuit analysis on open-source models. |
| **PyPI Package** | [glassbox-mech-interp](https://pypi.org/project/glassbox-mech-interp/) | `pip install glassbox-mech-interp` — v3.0.0 |

> Free tier spins down after 15 min inactivity — first request may take ~30s. For production, [self-host](#self-hosting).

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

No model weights? Use the [live HuggingFace demo](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) — no install required.

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
curl -X POST https://glassbox-ai-2-0-mechanistic.onrender.com/v1/webhooks \
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
  baseUrl: 'https://glassbox-ai-2-0-mechanistic.onrender.com'
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
curl -X POST https://glassbox-ai-2-0-mechanistic.onrender.com/v1/attention-patterns \
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

---

## Black-Box Audit — Any Model via API

No model weights needed. Works on GPT-4, Claude, Llama via any API endpoint. Uses counterfactual probing + sensitivity analysis + consistency testing to produce Article 13-compatible explainability metrics.

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

The API is live at `https://glassbox-ai-2-0-mechanistic.onrender.com`. Interactive docs at [`/docs`](https://glassbox-ai-2-0-mechanistic.onrender.com/docs).

**Black-box audit (any model via API):**

```bash
curl -X POST https://glassbox-ai-2-0-mechanistic.onrender.com/v1/audit/black-box \
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
curl -X POST https://glassbox-ai-2-0-mechanistic.onrender.com/v1/audit/analyze \
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
curl https://glassbox-ai-2-0-mechanistic.onrender.com/v1/audit/report/{report_id}
curl https://glassbox-ai-2-0-mechanistic.onrender.com/v1/audit/pdf/{report_id}  # download PDF
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

> **Preliminary results.** Timing benchmarks are single-run measurements on an Apple M2 Pro (32GB RAM), PyTorch 2.2, CUDA disabled. ACDC baseline uses the [official implementation](https://github.com/ArthurConmy/Automatic-Circuit-DisCovery) with default settings. Independent replication is encouraged — scripts in `benchmarks/`. Results will be updated as more hardware configurations are tested.

### IOI (Indirect Object Identification) — Wang et al. (2022)

Evaluated on the canonical IOI task across the GPT-2 family.

| Model | Layers | Heads | Suff. (approx)† | Suff. (exact)‡ | Comp. | F1 | Glassbox (s) | ACDC (s) | Speedup |
|-------|--------|-------|-----------------|----------------|-------|----|----------|------|---------|
| GPT-2 small | 12 | 12 | 80.0% | **~100%** | 37.2% | 48.8% | **1.2** | 43.2 | **~37×** |
| GPT-2 medium | 24 | 16 | 35.1% | ~61% | 23.7% | 27.9% | **4.9** | 115.2 | **~24×** |
| GPT-2 large | 36 | 20 | 18.2% | ~34% | 14.2% | 15.9% | **14.3** | 216.0 | **~15×** |

† **Approximate sufficiency** — first-order Taylor estimate computed in the default `analyze()` call (O(3) forward passes, no ablation). Formula: `Suff ≈ Σ attr(h) / LD_clean`. Fast but systematically underestimates because it ignores non-linear head interactions.

‡ **Exact sufficiency** — full causal ablation of all non-circuit heads, as reported in the [arXiv paper](https://arxiv.org/abs/2603.09988). Use `bootstrap_metrics()` to compute this. Exact sufficiency is always ≥ approximate sufficiency; the gap grows with circuit size and head interaction strength. FCAS and faithfulness metrics are cross-validated across 5 IOI prompts (see `benchmarks/run_ioi.py`).

### Cross-model Circuit Alignment (FCAS)

| Model pair | FCAS | z-score |
|-----------|------|---------|
| GPT-2 small ↔ GPT-2 medium | 0.835 | 4.21 |
| GPT-2 small ↔ GPT-2 large | 0.783 | 3.67 |
| GPT-2 medium ↔ GPT-2 large | 0.833 | 4.18 |

High FCAS confirms the IOI circuit is structurally conserved across scale (Wang et al. 2022).

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

**Option 1 — Live (no install):** Visit [glassbox-ai-2-0-mechanistic.onrender.com/dashboard](https://glassbox-ai-2-0-mechanistic.onrender.com/dashboard). Full compliance audit UI with demo mode — works with zero backend.

**Option 2 — Research UI (Gradio, local):**

```bash
pip install glassbox-mech-interp gradio matplotlib
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
python dashboard/app.py
# Opens Gradio at http://localhost:7860
# Tabs: Circuit Analysis · Logit Lens · Attention Patterns
```

**Option 3 — HuggingFace Space:** [huggingface.co/spaces/designer-coderajay/Glassbox-ai](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) — white-box circuit analysis, no install needed.

---

## Self-Hosting

Run the full API stack on your own infrastructure. Your data never leaves your environment.

```bash
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
docker build -t glassbox .
docker run -p 8000:8000 glassbox
# API live at http://localhost:8000
# Dashboard at http://localhost:8000/dashboard
# Swagger UI at http://localhost:8000/docs
```

One-click deploy to Render (free tier):

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool)

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

- The hosted demo API at `glassbox-ai-2-0-mechanistic.onrender.com` is provided strictly for **evaluation and testing** of the research tool.
- **No personal data is intentionally collected, stored, or processed.** Prompt text submitted via the API is processed in-memory to return a result and is not logged, retained, or shared.
- Standard server access logs (IP address, timestamp, request path) may be recorded automatically by the hosting platform (Render.com). These are not controlled by the project author. See [Render's privacy policy](https://render.com/privacy) for details.
- If you submit prompts containing personal data (e.g., names, financial details), you do so at your own risk. Do not send real personal data to the hosted API. For sensitive work, [self-host](#self-hosting).
- **Contact for data inquiries:** [mahale.ajay01@gmail.com](mailto:mahale.ajay01@gmail.com)
- **Responsible person (§5 TMG / Impressum):** Ajay Pravin Mahale, student, Germany. Contact: mahale.ajay01@gmail.com

**Trademark notice.** "Glassbox AI" and the full project name "Glassbox-AI-2.0-Mechanistic-Interpretability-tool" are used as academic project identifiers only. The unrelated commercial company Glassbox Ltd (digital customer experience analytics) may hold trademark registrations for "Glassbox" in certain jurisdictions. This academic project has no affiliation with, and makes no claim against, any trademarks held by Glassbox Ltd or any affiliated entities. If you are Glassbox Ltd and have a trademark concern, please contact [mahale.ajay01@gmail.com](mailto:mahale.ajay01@gmail.com) before taking legal action.

---

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">
Built by <a href="mailto:mahale.ajay01@gmail.com">Ajay Pravin Mahale</a> · MSc 2026 · Made in Germany<br>
<strong>Glassbox AI — see inside every prediction</strong>
</div>
