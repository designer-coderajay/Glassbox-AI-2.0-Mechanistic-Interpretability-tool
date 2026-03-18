<div align="center">

# Glassbox 2.0

**The only open-source EU AI Act Annex IV compliance audit platform. Works on any LLM.**

[![PyPI](https://img.shields.io/pypi/v/glassbox-mech-interp?color=blue&label=PyPI)](https://pypi.org/project/glassbox-mech-interp/)
[![Downloads](https://static.pepy.tech/badge/glassbox-mech-interp)](https://pepy.tech/project/glassbox-mech-interp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)
[![Live API](https://img.shields.io/badge/API-Live%20on%20Render-success)](https://glassbox-ai-2-0-mechanistic.onrender.com)
[![arXiv](https://img.shields.io/badge/arXiv-2603.09988-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.09988)
[![Tests](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml)

[**Live Demo**](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) · [**Dashboard**](https://glassbox-ai-2-0-mechanistic.onrender.com/dashboard) · [**API**](https://glassbox-ai-2-0-mechanistic.onrender.com) · [**Docs**](https://glassbox-ai-2-0-mechanistic.onrender.com/docs) · [**Paper**](https://arxiv.org/abs/2603.09988) · [**PyPI**](https://pypi.org/project/glassbox-mech-interp/)

</div>

---

**For compliance teams:** Regulation (EU) 2024/1689 (AI Act) requires Annex IV technical documentation for every high-risk AI system. Enforcement starts August 2026. Glassbox generates the full 9-section report automatically — from open-source models (white-box) or any proprietary API like GPT-4 and Claude (black-box). No other open-source tool does this.

**For researchers:** one function call discovers the minimum faithful circuit in a transformer — the smallest subgraph of attention heads causally responsible for a prediction. Preliminary benchmarks show 15–37× faster than ACDC on GPT-2 (single-run, Apple M2 Pro — see [Benchmarks](#benchmarks)). Every approximation is disclosed.

---

## Table of Contents

- [Live Services](#live-services)
- [Quickstart](#quickstart)
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
- [License](#license)

---

## Live Services

| Service | URL | Description |
|---------|-----|-------------|
| **Compliance Dashboard** | [/dashboard](https://glassbox-ai-2-0-mechanistic.onrender.com/dashboard) | Web UI for compliance officers. No install needed. |
| **REST API** | [glassbox-ai-2-0-mechanistic.onrender.com](https://glassbox-ai-2-0-mechanistic.onrender.com) | JSON API. See [/docs](https://glassbox-ai-2-0-mechanistic.onrender.com/docs) for Swagger UI. |
| **White-Box Demo** | [HuggingFace Space](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) | Interactive circuit analysis on open-source models. |
| **PyPI Package** | [glassbox-mech-interp](https://pypi.org/project/glassbox-mech-interp/) | `pip install glassbox-mech-interp` — v2.7.0 |

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
# {'sufficiency': 0.80, 'comprehensiveness': 0.37, 'f1': 0.49,
#  'category': 'backup_mechanisms', 'suff_is_approx': True}
```

No model weights? Use the [live HuggingFace demo](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) — no install required.

---

## EU AI Act Compliance — Annex IV Reports

[Regulation (EU) 2024/1689](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) requires Annex IV technical documentation (Article 11) for high-risk AI systems in finance, healthcare, HR, legal, and critical infrastructure. Enforcement starts August 2026. Non-compliance: up to €15M or 3% of global annual turnover, whichever is higher (Article 99(4)).

Glassbox generates all 9 required sections as a signed PDF + machine-readable JSON from a single function call:

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
report.to_pdf("annex_iv_report.pdf")   # legally-structured PDF
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

| Model | Layers | Heads | Suff.* | Comp. | F1 | Glassbox (s) | ACDC (s) | Speedup |
|-------|--------|-------|--------|-------|----|----------|------|---------|
| GPT-2 small | 12 | 12 | 80.0% | 37.2% | 48.8% | **1.2** | 43.2 | **~37×** |
| GPT-2 medium | 24 | 16 | 35.1% | 23.7% | 27.9% | **4.9** | 115.2 | **~24×** |
| GPT-2 large | 36 | 20 | 18.2% | 14.2% | 15.9% | **14.3** | 216.0 | **~15×** |

*Sufficiency is a first-order Taylor approximation. Exact causal sufficiency (full ablation over non-circuit heads) is higher — see the [arXiv paper](https://arxiv.org/abs/2603.09988). FCAS and faithfulness metrics are cross-validated across 5 IOI prompts (see `benchmarks/run_ioi.py`).

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

```bash
# Core — circuit analysis only (no extra deps beyond PyTorch + TransformerLens)
pip install glassbox-mech-interp

# With EU AI Act compliance report generation
pip install "glassbox-mech-interp[compliance]"

# With full REST API stack
pip install "glassbox-mech-interp[api]"

# With SAE feature attribution
pip install glassbox-mech-interp sae-lens

# Full development install
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
pip install -e ".[dev]"
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

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">
Built by <a href="mailto:mahale.ajay01@gmail.com">Ajay Pravin Mahale</a> · MSc 2026 · Made in Germany<br>
<strong>Glassbox AI — see inside every prediction</strong>
</div>
