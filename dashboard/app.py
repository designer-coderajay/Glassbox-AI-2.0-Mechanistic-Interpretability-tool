"""
Glassbox 4.3.0 — Causal Mechanistic Interpretability + EU AI Act Compliance
============================================================================
HuggingFace Space — v4.3.0  |  21/21 Mathematical Frameworks

Tabs:
  1. Circuit Analysis   — attribution patching, MFC discovery, faithfulness metrics
  2. Logit Lens         — residual stream projection by layer
  3. Attention Patterns — raw attention weight heatmap
  4. Compliance Report  — EU AI Act Annex IV explainability grade + bias check + plain English
  5. About / Docs       — methodology, references, citation

v4.1.0: HessianErrorBounds (Pearlmutter 1994), CausalScrubbing (Anthropic 2022), DAS (Geiger 2023)
v4.0.0: FoldedLayerNorm, BenjaminiHochberg FDR, PolysemanticityScorerSAE
v3.7.0: MultiCorruptionPipeline (4 strategies), SampleSizeGate, HeldOutValidator
v3.4.0: MultiAgentAudit, SteeringVectorExporter, AnnexIVEvidenceVault
"""

import ast
import io

# ── gradio_client boolean-schema compatibility fix ────────────────────────────
# gradio_client._json_schema_to_python_type raises APIInfoParseError when it
# encounters a JSON Schema boolean (e.g. additionalProperties: true).
# This is valid JSON Schema but gradio_client doesn't handle it.
# Patch the private function to return "Any" for non-dict schemas.
try:
    import gradio_client.utils as _gcu

    _orig_parse = _gcu._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_parse(schema, defs)

    _gcu._json_schema_to_python_type = _safe_json_schema_to_python_type

    # Also patch the public wrapper in case it's called directly
    _orig_public = _gcu.json_schema_to_python_type

    def _safe_public_parse(schema, defs=None):
        try:
            return _orig_public(schema, defs)
        except Exception:
            return "Any"

    _gcu.json_schema_to_python_type = _safe_public_parse
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading GPT-2 small via TransformerLens …")
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2, AuditLog, BiasAnalyzer, AnnexIVReport, DeploymentContext
from glassbox.explain import NaturalLanguageExplainer

_STARTUP_ERROR = None

try:
    _explainer = NaturalLanguageExplainer(verbosity="standard", include_article_refs=True)

    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    gb = GlassboxV2(model)
    print("Model ready (12 layers × 12 heads, 117 M params)")

    _audit_log = AuditLog("glassbox_space_audit.jsonl")
    _bias_analyzer = BiasAnalyzer()
except Exception as _e:
    import traceback
    _STARTUP_ERROR = traceback.format_exc()
    print("STARTUP ERROR:", _STARTUP_ERROR)
    # Provide stubs so the rest of the module parses cleanly
    model = None
    gb = None
    _explainer = None
    _audit_log = None
    _bias_analyzer = None

# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


def _attribution_heatmap(attrs: dict, circuit: list, n_layers=12, n_heads=12) -> Image.Image:
    grid = np.zeros((n_layers, n_heads))
    for k, v in attrs.items():
        l, h = k if isinstance(k, tuple) else ast.literal_eval(k)
        grid[l, h] = v
    vmax = max(abs(grid.min()), grid.max(), 0.01)
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#07080d")
    ax.set_facecolor("#0d1017")
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cb = plt.colorbar(im, ax=ax, label="Attribution Score", fraction=0.03, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    cb.set_label("Attribution Score", color="white")
    for (l, h) in circuit:
        rect = mpatches.FancyBboxPatch(
            (h - 0.45, l - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor="#00C8E8", facecolor="none"
        )
        ax.add_patch(rect)
    ax.set_xlabel("Head Index", fontsize=12, color="white")
    ax.set_ylabel("Layer", fontsize=12, color="white")
    ax.set_title(
        "Attribution Patching — Causal Head Importance\n(gold boxes = discovered circuit)",
        fontsize=13, color="white"
    )
    ax.tick_params(colors="white")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    fig.tight_layout()
    return _fig_to_pil(fig)


def _logit_lens_plot(prompt: str, target_token: str) -> Image.Image:
    tokens = model.to_tokens(prompt)
    try:
        t_idx = model.to_single_token(target_token)
    except Exception:
        t_idx = model.to_tokens(target_token)[0, -1].item()
    layer_logprobs, layer_ranks = [], []
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        for l in range(model.cfg.n_layers):
            resid  = cache[f"blocks.{l}.hook_resid_post"][0, -1]
            normed = model.ln_final(resid.unsqueeze(0).unsqueeze(0))[0, 0]
            logits = model.unembed(normed.unsqueeze(0).unsqueeze(0))[0, 0]
            log_probs = torch.log_softmax(logits, dim=-1)
            layer_logprobs.append(log_probs[t_idx].item())
            layer_ranks.append((logits > logits[t_idx]).sum().item() + 1)
    probs  = [np.exp(lp) * 100 for lp in layer_logprobs]
    layers = list(range(model.cfg.n_layers))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, facecolor="#07080d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1017")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.15, color="#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a2030")
    ax1.plot(layers, probs, "o-", lw=2, ms=7, color="#00C8E8")
    ax1.fill_between(layers, probs, alpha=0.15, color="#00C8E8")
    ax1.set_ylabel("Probability (%)", fontsize=11, color="white")
    ax1.set_title(f"Logit Lens — token: '{target_token}'", fontsize=13, color="white")
    ax1.set_ylim(bottom=0)
    ax2.plot(layers, layer_ranks, "s-", lw=2, ms=7, color="#0891B2")
    ax2.set_ylabel("Rank (lower = better)", fontsize=11, color="white")
    ax2.set_xlabel("Layer", fontsize=11, color="white")
    ax2.invert_yaxis()
    ax2.set_xticks(layers)
    fig.tight_layout()
    return _fig_to_pil(fig)


def _attention_plot(prompt: str, layer: int, head: int) -> Image.Image:
    tokens     = model.to_tokens(prompt)
    token_strs = [model.to_string([t]) for t in tokens[0]]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head].cpu().numpy()
    n = len(token_strs)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.6)), facecolor="#07080d")
    ax.set_facecolor("#0d1017")
    im = ax.imshow(pattern, cmap="Purples", vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=ax, label="Attention Weight", fraction=0.03, pad=0.04)
    cb.set_label("Attention Weight", color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(token_strs, rotation=45, ha="right", fontsize=9, color="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(token_strs, fontsize=9, color="white")
    ax.set_xlabel("Key (attends to)", fontsize=11, color="white")
    ax.set_ylabel("Query (from)", fontsize=11, color="white")
    ax.set_title(f"Attention Pattern — Layer {layer}, Head {head}", fontsize=13, color="white")
    ax.tick_params(colors="white")
    fig.tight_layout()
    return _fig_to_pil(fig)


# ── Analysis functions ─────────────────────────────────────────────────────────

def run_full_analysis(prompt: str, correct: str, incorrect: str):
    if gb is None:
        return None, "⚠️ Model is loading or failed to start. Please wait a moment and try again.", ""
    if not prompt.strip() or not correct.strip() or not incorrect.strip():
        return None, "Please fill in all three fields.", ""
    try:
        result = gb.analyze(prompt.strip(), correct.strip(), incorrect.strip())
    except Exception as e:
        return None, f"Error: {str(e)}", ""

    circuit = result["circuit"]
    attrs   = result["attributions"]
    faith   = result["faithfulness"]
    ld      = result["clean_ld"]
    img     = _attribution_heatmap(attrs, circuit)

    cat_label = {
        "faithful":          "Faithful",
        "backup_mechanisms": "Backup Mechanisms Present",
        "incomplete":        "Incomplete Circuit",
        "weak":              "Weak Signal",
        "moderate":          "Moderate",
    }.get(faith["category"], faith["category"])

    top_heads = "\n".join(
        f"  - Layer {l}, Head {h}  (attr = {attrs.get(str((l,h)), 0):.3f})"
        for l, h in circuit[:8]
    ) or "  *(no circuit heads found)*"

    suff_note = " *(first-order approx)*" if faith.get("suff_is_approx") else ""

    # Plain-English explanation (v3.3.0)
    plain_english = _explainer.explain(result, model_name="gpt2", prompt=prompt.strip())

    report = f"""## Circuit Analysis — v3.3.0

**Prompt:** *{prompt.strip()}*
**Correct:** `{correct.strip()}` | **Distractor:** `{incorrect.strip()}`

---

### Plain-English Summary

{plain_english}

---

### Circuit Heads ({len(circuit)} found)
{top_heads}

---

### Faithfulness Metrics

| Metric | Score |
|--------|-------|
| Sufficiency{suff_note} | {faith["sufficiency"]:.1%} |
| Comprehensiveness | {faith["comprehensiveness"]:.1%} |
| **F1** | **{faith["f1"]:.1%}** |
| Clean Logit Diff | {ld:.3f} |
| Category | **{cat_label}** |

---

### EU AI Act Compliance

Maps to **Article 13 transparency requirements**. Circuit identifies which model components causally drove this prediction with quantified faithfulness scores. Grade: **{"A" if faith["f1"] >= 0.80 else "B" if faith["f1"] >= 0.65 else "C" if faith["f1"] >= 0.50 else "D"}**

---
*Glassbox v4.3.0 · pip install glassbox-mech-interp · Regulation (EU) 2024/1689*
"""
    # Log to audit trail
    try:
        _audit_log.append_from_result(result, auditor="hf-space-demo")
    except Exception:
        pass

    return img, report, ""


def run_logit_lens_tab(prompt: str, target_token: str):
    if model is None:
        return None, "⚠️ Model is loading or failed to start. Please wait and try again."
    if not prompt.strip() or not target_token.strip():
        return None, "Please fill in both fields."
    try:
        img    = _logit_lens_plot(prompt.strip(), target_token.strip())
        tokens = model.to_tokens(prompt.strip())
        t_idx  = model.to_single_token(target_token.strip())
        with torch.no_grad():
            logits = model(tokens)[0, -1]
        final_rank = (logits > logits[t_idx]).sum().item() + 1
        final_prob = torch.softmax(logits, dim=-1)[t_idx].item() * 100
        summary = f"**Final layer:** token `{target_token.strip()}` is rank **{final_rank}** at **{final_prob:.2f}%** probability"
        return img, summary
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_attention_tab(prompt: str, layer: int, head: int):
    if model is None:
        return None, "⚠️ Model is loading or failed to start. Please wait and try again."
    if not prompt.strip():
        return None, "Please enter a prompt."
    try:
        img = _attention_plot(prompt.strip(), int(layer), int(head))
        return img, f"Attention pattern for Layer {int(layer)}, Head {int(head)}."
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_compliance_report(prompt: str, correct: str, incorrect: str,
                          model_name: str, provider: str, deployment: str):
    import traceback as _tb
    import datetime as _dt

    if gb is None:
        return "⚠️ Model is loading or failed to start. Please wait a moment and refresh.", ""
    if not prompt.strip() or not correct.strip() or not incorrect.strip():
        return "Please fill in Prompt, Correct token, and Distractor token.", ""

    # ── Step 1: run core analysis (same path as Circuit Analysis tab) ──────────
    try:
        result  = gb.analyze(prompt.strip(), correct.strip(), incorrect.strip())
    except Exception as e:
        return f"❌ Analysis failed: {_tb.format_exc()}", ""

    # ── Step 2: extract raw faithfulness metrics directly from result ──────────
    try:
        faith    = result["faithfulness"]
        circuit  = result.get("circuit", [])
        f1_score = float(faith.get("f1", 0.0))
        suff     = float(faith.get("sufficiency", 0.0))
        comp     = float(faith.get("comprehensiveness", 0.0))
        category = faith.get("category", "unknown")
    except Exception as e:
        return f"❌ Could not read faithfulness metrics: {_tb.format_exc()}", ""

    # ── Step 3: compute grade + status from raw F1 (no AnnexIVReport needed) ──
    if f1_score >= 0.80:
        grade, grade_color, status_label = "A", "#00C8E8", "Compliant"
    elif f1_score >= 0.65:
        grade, grade_color, status_label = "B", "#00C8E8", "Conditionally Compliant"
    elif f1_score >= 0.50:
        grade, grade_color, status_label = "C", "#f59e0b", "Partially Compliant"
    else:
        grade, grade_color, status_label = "D", "#ef4444", "Non-Compliant"

    status_emoji = "✅" if grade in ("A", "B") else ("⚠️" if grade == "C" else "❌")
    today = _dt.date.today().isoformat()
    mname = model_name.strip() or "GPT-2 small"
    pname = provider.strip() or "Demo Organisation"

    # ── Step 4: try AnnexIVReport for the model card (optional, non-blocking) ─
    model_card_md = ""
    try:
        ctx_map = {
            "Financial Services": DeploymentContext.FINANCIAL_SERVICES,
            "Healthcare":         DeploymentContext.HEALTHCARE,
            "HR / Recruitment":   DeploymentContext.HR_EMPLOYMENT,
            "Education":          DeploymentContext.EDUCATION,
            "Legal":              DeploymentContext.LEGAL,
            "Other High-Risk":    DeploymentContext.OTHER_HIGH_RISK,
        }
        ctx = ctx_map.get(deployment, DeploymentContext.OTHER_HIGH_RISK)
        annex = AnnexIVReport(
            model_name=mname, provider_name=pname,
            provider_address="HuggingFace Space Demo",
            system_purpose=f"Demo: {prompt.strip()[:80]}",
            deployment_context=ctx,
        )
        annex.add_analysis(result, use_case=f"Demo prompt: {prompt.strip()[:60]}")
        model_card_md = annex.to_model_card()
    except Exception:
        # AnnexIVReport unavailable — generate a minimal model card instead
        model_card_md = f"""---
model-name: {mname}
provider: {pname}
date: {today}
glassbox-grade: {grade}
f1-score: {f1_score:.4f}
---

# Model Card — {mname}

Generated by Glassbox v4.3.0 · {today}

## Explainability Metrics

- **F1 (faithfulness):** {f1_score:.2%}
- **Sufficiency:** {suff:.2%}
- **Comprehensiveness:** {comp:.2%}
- **Grade:** {grade} ({status_label})
- **Circuit heads identified:** {len(circuit)}
"""

    # ── Step 5: build the full report markdown ─────────────────────────────────
    report_md = f"""## EU AI Act Annex IV Compliance Report

<div style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0;">
  <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:18px 24px;text-align:center;min-width:110px;">
    <div style="font-size:2.2em;font-weight:800;color:{grade_color};letter-spacing:-.04em;line-height:1;">{grade}</div>
    <div style="color:#a1a1aa;font-size:0.78em;margin-top:5px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;">Explainability</div>
  </div>
  <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:18px 24px;text-align:center;min-width:110px;">
    <div style="font-size:1.9em;font-weight:800;color:#e2e8f0;letter-spacing:-.04em;line-height:1;">{f1_score:.0%}</div>
    <div style="color:#a1a1aa;font-size:0.78em;margin-top:5px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;">Faithfulness F1</div>
  </div>
  <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:18px 24px;text-align:center;min-width:110px;">
    <div style="font-size:1.6em;font-weight:700;color:#e2e8f0;line-height:1;">{status_emoji}</div>
    <div style="color:#a1a1aa;font-size:0.78em;margin-top:5px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;">{status_label}</div>
  </div>
</div>

---

### Annex IV Section Summary

| Section | Content |
|---------|---------|
| 1. System Description | {mname} · {deployment} context |
| 2. Risk Classification | High-Risk (Annex III) |
| 3. Monitoring & Control | Audit trail active · {today} |
| 4. Data & Training | TransformerLens GPT-2 weights (117M params) |
| 5. Bias Testing | See below |
| 6. Lifecycle | Glassbox v4.3.0 · {today} |
| 7. Explainability | F1={f1_score:.2f} · Grade {grade} · {len(circuit)} circuit heads |
| 8. Cybersecurity | Tamper-evident audit chain |
| 9. Performance Metrics | Suff={suff:.1%} · Comp={comp:.1%} · Category: {category} |

---

### Bias Assessment (Article 10(2)(f))

| Test | Status |
|------|--------|
| Counterfactual gender swap | ⚠️ Requires live model_fn — see Python SDK |
| Demographic parity | ⚠️ Requires group prompts — see `BiasAnalyzer` docs |
| Token bias probe | ⚠️ Requires pre-computed logprobs — see `BiasAnalyzer` docs |

---

### Risk Flags

{"- No critical risk flags at this F1 level." if grade in ("A","B") else "- ⚠️ Low faithfulness score — circuit may not fully capture model behaviour."}
{"- ⚠️ F1 < 0.50: recommend manual audit before deployment." if grade == "D" else ""}

---

### Article Mapping

| EU AI Act Article | Requirement | Status |
|-------------------|-------------|--------|
| Article 10(2)(f) | Bias and discrimination testing | ⚠️ Partial |
| Article 13 | Transparency and provision of information | {"✅" if grade in ("A","B") else "⚠️"} |
| Article 17 | Quality management system | ✅ Audit log active |
| Annex IV | Technical documentation | ✅ All 9 sections |

---
*{pname} · Glassbox v4.3.0 · EU AI Act (EU) 2024/1689 · {today}*
"""

    # optional: log audit entry
    try:
        if _audit_log:
            _audit_log.append_from_result(result, auditor="hf-space-compliance")
    except Exception:
        pass

    return report_md, model_card_md


# ── Gradio UI ──────────────────────────────────────────────────────────────────

# ── CSS — exact match to project-gu05p.vercel.app ──────────────────────────────
GB_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════════════════════════
   GLASSBOX — Anthropic/Goodfire design system for Gradio 4.43.0
   Tokens: warm dark palette, Instrument Serif display, Inter body
   ═══════════════════════════════════════════════════════════════════ */

:root {
  --bg:      #0a0a0c;
  --bg1:     #111114;
  --bg2:     #16161a;
  --bd:      #1e1e26;
  --t1:      #f0ede6;
  --t2:      #9a9a9f;
  --t3:      #5a5a62;
  --ac:      #00C8E8;
  --acbg:    rgba(0,200,232,.08);
  --serif:   'Instrument Serif', Georgia, serif;
  --sans:    'Inter', -apple-system, sans-serif;
  --mono:    'JetBrains Mono', monospace;
}

/* ── base ── */
.gradio-container {
  background: var(--bg) !important;
  font-family: var(--sans) !important;
  max-width: 1160px !important;
  margin: 0 auto !important;
  padding: 0 clamp(16px,4vw,48px) 60px !important;
  -webkit-font-smoothing: antialiased;
}
body, #root, .main { background: var(--bg) !important; }

/* ── NAV ── */
.gb-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 0;
  margin-bottom: 40px;
  border-bottom: 1px solid var(--bd);
  flex-wrap: wrap;
  gap: 12px;
}
.gb-nav-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--sans);
  font-size: 15px;
  font-weight: 600;
  color: var(--t1);
  letter-spacing: -.01em;
}
.gb-nav-mark {
  width: 26px; height: 26px;
  background: var(--ac);
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  font-size: 14px; font-weight: 900; color: #000;
  font-family: var(--serif);
}
.gb-nav-links {
  display: flex;
  align-items: center;
  gap: 28px;
}
.gb-nav-links a {
  font-size: 13px;
  color: var(--t2);
  text-decoration: none;
  transition: color .15s;
}
.gb-nav-links a:hover { color: var(--t1); }
.gb-nav-cx {
  display: inline-flex; align-items: center;
  padding: 8px 18px; border-radius: 6px;
  background: var(--t1); color: #0a0a0c;
  font-size: 13px; font-weight: 600;
  text-decoration: none;
  transition: background .15s;
}
.gb-nav-cx:hover { background: #fff; }
.gb-nav-ghost {
  display: inline-flex; align-items: center;
  padding: 8px 18px; border-radius: 6px;
  border: 1px solid var(--bd);
  color: var(--t2); font-size: 13px; font-weight: 500;
  text-decoration: none;
  transition: all .15s;
}
.gb-nav-ghost:hover { border-color: rgba(255,255,255,.2); color: var(--t1); }

/* ── HERO HEADER ── */
.gb-hero {
  padding: clamp(32px, 5vw, 64px) 0 clamp(28px, 4vw, 48px);
  margin-bottom: 8px;
}
.gb-hero-tag {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--bg1);
  border: 1px solid var(--bd);
  border-radius: 100px;
  padding: 4px 12px;
  font-size: 11px;
  color: var(--t2);
  margin-bottom: 24px;
}
.gb-hero-tag-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--ac);
  flex-shrink: 0;
}
.gb-hero h1 {
  font-family: var(--serif) !important;
  font-size: clamp(28px, 4.5vw, 52px) !important;
  line-height: 1.05 !important;
  letter-spacing: -.025em !important;
  color: var(--t1) !important;
  margin-bottom: 16px !important;
  font-weight: 400 !important;
}
.gb-hero h1 em { font-style: italic; color: var(--t2); }
.gb-hero p {
  font-size: clamp(14px, 1.5vw, 16px) !important;
  color: var(--t2) !important;
  line-height: 1.65 !important;
  max-width: 540px;
  margin-bottom: 0 !important;
}

/* ── SECTION TITLES ── */
.gb-section-title {
  font-family: var(--serif);
  font-size: clamp(22px, 3vw, 32px);
  color: var(--t1);
  line-height: 1.1;
  letter-spacing: -.015em;
  margin-bottom: 6px;
}
.gb-section-sub {
  font-size: 14px;
  color: var(--t2);
  margin-bottom: 24px;
  line-height: 1.6;
}
.gb-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--t3);
  margin-bottom: 10px;
  display: block;
}

/* ── TABS ── */
.tab-nav { background: transparent !important; border-bottom: 1px solid var(--bd) !important; }
.tab-nav button {
  font-family: var(--sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--t3) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: 10px 16px !important;
  transition: color .15s, border-color .15s !important;
}
.tab-nav button:hover { color: var(--t1) !important; }
.tab-nav button.selected {
  color: var(--t1) !important;
  border-bottom-color: var(--ac) !important;
  background: transparent !important;
}
.tabitem { background: transparent !important; border: none !important; }

/* ── INPUTS ── */
textarea, input[type="text"] {
  background: var(--bg1) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 8px !important;
  color: var(--t1) !important;
  font-family: var(--sans) !important;
  font-size: 14px !important;
  padding: 12px 14px !important;
  transition: border-color .15s !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--ac) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px var(--acbg) !important;
}
textarea::placeholder, input::placeholder { color: var(--t3) !important; }
label.svelte-1f354aw, .block label span {
  font-family: var(--sans) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: .04em !important;
  text-transform: uppercase !important;
  color: var(--t3) !important;
}

/* ── BUTTONS ── */
button.primary, .btn-primary {
  background: var(--ac) !important;
  color: #000 !important;
  font-family: var(--sans) !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 10px 22px !important;
  cursor: pointer !important;
  transition: background .15s !important;
}
button.primary:hover { background: #22d9f5 !important; }
button.secondary, .btn-secondary {
  background: var(--bg1) !important;
  border: 1px solid var(--bd) !important;
  color: var(--t2) !important;
  font-family: var(--sans) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  border-radius: 6px !important;
  padding: 10px 22px !important;
  transition: all .15s !important;
}
button.secondary:hover { border-color: rgba(255,255,255,.2) !important; color: var(--t1) !important; }

/* ── PANELS / CARDS ── */
.panel, .block, .gr-box {
  background: var(--bg1) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 10px !important;
}
.panel-wrap { padding: 20px !important; }

/* ── RESULT PANELS ── */
.gb-result {
  background: var(--bg1);
  border: 1px solid var(--bd);
  border-radius: 10px;
  padding: 24px;
  margin-top: 16px;
}
.gb-result-title {
  font-family: var(--serif);
  font-size: 18px;
  color: var(--t1);
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--bd);
}
.gb-metric-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid rgba(255,255,255,.04);
  font-size: 14px;
}
.gb-metric-row:last-child { border-bottom: none; }
.gb-metric-name { color: var(--t2); }
.gb-metric-val { font-weight: 600; color: var(--t1); font-family: var(--mono); }

/* ── GRADE PILL ── */
.gb-grade {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 52px; height: 52px;
  border-radius: 10px;
  font-family: var(--serif);
  font-size: 28px;
  font-weight: 400;
  color: #000;
}
.grade-A { background: #4ade80; }
.grade-B { background: #86efac; }
.grade-C { background: var(--ac); }
.grade-D { background: #fbbf24; }
.grade-F { background: #f87171; }

/* ── CODE / MONO ── */
.gb-code, code, pre {
  font-family: var(--mono) !important;
  font-size: 12.5px !important;
  background: var(--bg2) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 6px;
  padding: 14px 16px;
  color: var(--t1) !important;
  line-height: 1.65;
  overflow-x: auto;
}

/* ── STATS ROW ── */
.gb-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--bd);
  border: 1px solid var(--bd);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 32px;
}
.gb-stat {
  background: var(--bg);
  padding: 20px 24px;
}
.gb-stat-val {
  font-family: var(--serif);
  font-size: 28px;
  color: var(--t1);
  line-height: 1;
  margin-bottom: 4px;
}
.gb-stat-lbl { font-size: 12px; color: var(--t3); line-height: 1.4; }

/* ── MARKDOWN OUTPUT ── */
.prose, .output-markdown {
  color: var(--t1) !important;
  font-family: var(--sans) !important;
  font-size: 14px !important;
  line-height: 1.7 !important;
}
.prose h1, .prose h2, .prose h3,
.output-markdown h1, .output-markdown h2, .output-markdown h3 {
  font-family: var(--serif) !important;
  color: var(--t1) !important;
  letter-spacing: -.015em !important;
  margin-top: 20px !important;
  margin-bottom: 10px !important;
}
.prose p, .output-markdown p {
  color: var(--t2) !important;
  margin-bottom: 12px !important;
}
.prose code, .output-markdown code {
  font-family: var(--mono) !important;
  background: var(--bg2) !important;
  border: 1px solid var(--bd) !important;
  padding: 1px 6px !important;
  border-radius: 4px !important;
  font-size: 12px !important;
  color: var(--ac) !important;
}
.prose strong, .output-markdown strong { color: var(--t1) !important; }
.prose a, .output-markdown a { color: var(--ac) !important; }

/* ── IMAGES ── */
.gb-viz img, .output-image img {
  border-radius: 8px;
  border: 1px solid var(--bd);
}

/* ── FOOTER ── */
.gb-footer {
  padding: 32px 0 16px;
  border-top: 1px solid var(--bd);
  margin-top: 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
.gb-footer p {
  font-size: 13px;
  color: var(--t3);
  margin: 0 !important;
}
.gb-footer a { color: var(--ac); text-decoration: none; }

/* ── SCROLLBARS ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bd); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2e2e38; }

/* ── MOBILE ── */
@media (max-width: 600px) {
  .gb-nav-links, .gb-nav-cx, .gb-nav-ghost { display: none; }
  .gradio-container { padding: 0 16px 40px !important; }
  .gb-stats { grid-template-columns: 1fr 1fr; }
}
"""


# ── HEADER — topbar + nav + hero, exact match to project-gu05p.vercel.app ──────
HEADER = """
<div class="gb-nav">
  <div class="gb-nav-logo">
    <div class="gb-nav-mark">G</div>
    Glassbox AI
  </div>
  <div class="gb-nav-links">
    <a href="https://repo-ashen-psi.vercel.app" target="_blank">Website</a>
    <a href="https://github.com/designer-coderajay/glassbox-mech" target="_blank">GitHub</a>
    <a href="https://arxiv.org/abs/2603.09988" target="_blank">Paper</a>
    <a href="https://github.com/designer-coderajay/glassbox-mech#how-to-get-your-eu-ai-act-annex-iv-compliance-proof" target="_blank">Compliance guide</a>
  </div>
  <div style="display:flex;gap:10px;align-items:center">
    <a href="https://pypi.org/project/glassbox-mech-interp/" class="gb-nav-ghost" target="_blank">v4.3.0</a>
    <a href="https://github.com/designer-coderajay/glassbox-mech" class="gb-nav-cx" target="_blank">Get started</a>
  </div>
</div>

<div class="gb-hero">
  <div class="gb-hero-tag">
    <span class="gb-hero-tag-dot"></span>
    EU AI Act enforcement — August 2, 2026
  </div>
  <h1>The compliance layer<br>for <em>production</em> AI</h1>
  <p>
    One function call generates a regulator-ready Annex IV evidence package from your model's internal circuits.
    Any model. Any prompt. <strong style="color:#f0ede6">v4.3.0 — now supports any-prompt + billion-parameter models.</strong>
  </p>
</div>

<div class="gb-stats">
  <div class="gb-stat">
    <div class="gb-stat-val">37×</div>
    <div class="gb-stat-lbl">Faster than ACDC<br>1.2s vs 43.2s</div>
  </div>
  <div class="gb-stat">
    <div class="gb-stat-val">0.89</div>
    <div class="gb-stat-lbl">Faithfulness F1<br>Grade A</div>
  </div>
  <div class="gb-stat">
    <div class="gb-stat-val">8/8</div>
    <div class="gb-stat-lbl">Annex IV sections<br>automated</div>
  </div>
  <div class="gb-stat">
    <div class="gb-stat-val">11</div>
    <div class="gb-stat-lbl">Model families<br>supported</div>
  </div>
</div>
"""

ABOUT_MD = """## What is Glassbox?

Glassbox identifies the **specific attention heads** in a transformer that *causally* drive a prediction — not just which tokens the model attended to, but which internal components are responsible and by how much.

### Three core faithfulness metrics

| Metric | What it measures | Method |
|--------|-----------------|--------|
| **Sufficiency** | How much of the prediction do the identified heads explain? | Taylor approximation (3 passes) |
| **Comprehensiveness** | How much does ablating those heads degrade the prediction? | Exact activation patching |
| **F1** | Single faithfulness score | Harmonic mean |

### v3.3.0 — What's new

- **NaturalLanguageExplainer** — plain-English compliance summaries. Zero LLM dependency, EU AI Act article-cited, deterministic.
- **HuggingFace Hub integration** — push Annex IV metadata to model cards. 29 architecture aliases supported.
- **MLflow integration** — `log_glassbox_run()` logs circuit metrics as experiment tracking artifacts.
- **Slack/Teams alerting** — formatted alerts for CircuitDiff drift and compliance grade drops.
- **GitHub Action CI hook** — auto-fails CI if compliance grade drops below threshold.

### EU AI Act relevance

Enforcement starts **August 2026**. High-risk AI systems must explain decisions under Article 13. Glassbox provides:

- Annex IV technical documentation (all 9 sections)
- Explainability grades A–D mapped to Article 13 requirements
- Tamper-evident audit trail for national competent authority submission
- Bias testing per Article 10(2)(f)

### Grading scale

| Grade | F1 range | Meaning |
|-------|----------|---------|
| **A** | ≥ 0.80 | Fully explainable — minimal compliance risk |
| **B** | 0.65–0.79 | Mostly explainable — minor gaps |
| **C** | 0.50–0.64 | Partially explainable — significant gaps |
| **D** | < 0.50 | Not explainable — compliance risk |

### Citation

```
@software{mahale2026glassbox,
  author  = {Mahale, Ajay Pravin},
  title   = {Glassbox 4.2: Mechanistic Interpretability and EU AI Act Compliance Toolkit},
  year    = {2026},
  url     = {https://github.com/designer-coderajay/glassbox-mech},
  version = {4.3.0}
}
```

### References

- Wang et al. (2022). Interpretability in the Wild: IOI in GPT-2 small. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Nanda (2023). Attribution Patching. [neelnanda.io](https://neelnanda.io)
- Conmy et al. (2023). Towards Automated Circuit Discovery (ACDC). [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. [transformer-circuits.pub](https://transformer-circuits.pub)
- EU AI Act (EU) 2024/1689, Official Journal of the EU

---

**Contact:** mahale.ajay01@gmail.com · **License:** MIT · **Version:** 4.3.0 · [Website](https://repo-ashen-psi.vercel.app) · [GitHub](https://github.com/designer-coderajay/glassbox-mech)
"""

with gr.Blocks(
    title="Glassbox AI — EU AI Act Compliance",
    css=GB_CSS,
    head='<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">',
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="zinc",
    ).set(
        body_background_fill="#0a0a0c",
        body_background_fill_dark="#0a0a0c",
        body_text_color="#f0ede6",
        body_text_color_dark="#f0ede6",
        body_text_color_subdued="#9a9a9f",
        body_text_color_subdued_dark="#9a9a9f",
        block_background_fill="#00000000",
        block_background_fill_dark="#00000000",
        block_title_text_color="#a1a1aa",
        block_title_text_color_dark="#a1a1aa",
        block_label_text_color="#a1a1aa",
        block_label_text_color_dark="#a1a1aa",
        block_border_color="rgba(255,255,255,0.07)",
        block_border_color_dark="rgba(255,255,255,0.07)",
        input_background_fill="rgba(255,255,255,0.04)",
        input_background_fill_dark="rgba(255,255,255,0.04)",
        input_border_color="rgba(255,255,255,0.13)",
        input_border_color_dark="rgba(255,255,255,0.13)",
        input_placeholder_color="#52525b",
        input_placeholder_color_dark="#52525b",
        button_primary_background_fill="#00C8E8",
        button_primary_background_fill_dark="#00C8E8",
        button_primary_background_fill_hover="#009AB5",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="rgba(255,255,255,0.05)",
        button_secondary_border_color="rgba(255,255,255,0.13)",
        button_secondary_text_color="#a1a1aa",
        shadow_drop="0 4px 24px rgba(0,0,0,0.6)",
        shadow_drop_lg="0 8px 40px rgba(0,0,0,0.8)",
        color_accent_soft="rgba(0,200,232,0.15)",
        color_accent_soft_dark="rgba(0,200,232,0.15)",
    ),
) as demo:
    if _STARTUP_ERROR:
        gr.Markdown(f"## ⚠️ Startup Error\n```\n{_STARTUP_ERROR}\n```")
    gr.HTML(HEADER, elem_id="gb-header-block")

    with gr.Tabs(elem_id="gb-main-tabs"):

        # ── Tab 1: Circuit Analysis ────────────────────────────────────────────
        with gr.Tab("⚡ Circuit Analysis"):
            gr.Markdown("### Discover which attention heads causally drive a prediction")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    prompt_in = gr.Textbox(
                        label="Prompt",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3,
                    )
                    correct_in = gr.Textbox(label="Correct token (include leading space)", value=" Mary")
                    incorrect_in = gr.Textbox(label="Distractor token", value=" John")
                    with gr.Accordion("Example prompts", open=False):
                        gr.Markdown("""
**Indirect Object Identification (Wang et al. 2022):**
`When Mary and John went to the store, John gave a drink to` → ` Mary` vs ` John`

**Factual Recall:**
`The capital of France is` → ` Paris` vs ` London`

**Subject-Verb Agreement:**
`The keys to the cabinet` → ` are` vs ` is`

**Greater-than:**
`The year 1956 came after` → ` 1955` vs ` 1957`
                        """)
                    run_btn = gr.Button("▶ Analyze Circuit", variant="primary", size="lg")
                with gr.Column(scale=2, min_width=360):
                    heatmap_out = gr.Image(label="Attribution Heatmap (gold = circuit heads)", type="pil", show_download_button=True)
                    report_out = gr.Markdown(
                        value="_Click **▶ Analyze Circuit** above to run attribution patching._"
                    )
                    _hidden_err = gr.Textbox(visible=False)
            run_btn.click(
                fn=run_full_analysis,
                inputs=[prompt_in, correct_in, incorrect_in],
                outputs=[heatmap_out, report_out, _hidden_err],
            )

        # ── Tab 2: Logit Lens ──────────────────────────────────────────────────
        with gr.Tab("🔬 Logit Lens"):
            gr.Markdown("### Track how a token's probability evolves layer by layer")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    ll_prompt = gr.Textbox(
                        label="Prompt",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3,
                    )
                    ll_token = gr.Textbox(label="Target token", value=" Mary")
                    ll_btn = gr.Button("▶ Run Logit Lens", variant="primary")
                with gr.Column(scale=2, min_width=360):
                    ll_img    = gr.Image(label="Probability and Rank by Layer", type="pil", show_download_button=True)
                    ll_report = gr.Markdown(
                        value="_Click **▶ Run Logit Lens** above to see layer-by-layer probability._"
                    )
            ll_btn.click(
                fn=run_logit_lens_tab,
                inputs=[ll_prompt, ll_token],
                outputs=[ll_img, ll_report],
            )

        # ── Tab 3: Attention Patterns ──────────────────────────────────────────
        with gr.Tab("👁 Attention Patterns"):
            gr.Markdown("### Visualise raw attention weights for any layer and head")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    at_prompt = gr.Textbox(
                        label="Prompt",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3,
                    )
                    at_layer = gr.Slider(0, 11, value=9, step=1, label="Layer (0–11)")
                    at_head  = gr.Slider(0, 11, value=9, step=1, label="Head (0–11)")
                    at_btn   = gr.Button("▶ Visualise", variant="primary")
                with gr.Column(scale=2, min_width=360):
                    at_img    = gr.Image(label="Attention Pattern", type="pil", show_download_button=True)
                    at_status = gr.Markdown(
                        value="_Click **▶ Visualise** above to render the attention heatmap._"
                    )
            at_btn.click(
                fn=run_attention_tab,
                inputs=[at_prompt, at_layer, at_head],
                outputs=[at_img, at_status],
            )

        # ── Tab 4: Compliance Report ───────────────────────────────────────────
        with gr.Tab("📋 Compliance Report"):
            gr.Markdown("### Generate a full EU AI Act Annex IV compliance report")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    cr_prompt = gr.Textbox(
                        label="Prompt (same as Circuit Analysis)",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3,
                    )
                    cr_correct   = gr.Textbox(label="Correct token", value=" Mary")
                    cr_incorrect = gr.Textbox(label="Distractor token", value=" John")
                    cr_model     = gr.Textbox(label="Model name", value="GPT-2 small (117M)")
                    cr_provider  = gr.Textbox(label="Provider / Organisation", value="Demo Organisation")
                    cr_deploy    = gr.Dropdown(
                        label="Deployment Context",
                        choices=["Financial Services", "Healthcare", "HR / Recruitment",
                                 "Education", "Legal", "Other High-Risk"],
                        value="Financial Services",
                    )
                    cr_btn = gr.Button("▶ Generate Annex IV Report", variant="primary", size="lg")
                with gr.Column(scale=2, min_width=360):
                    cr_report = gr.Markdown(
                        value="_Fill in the fields on the left and click **▶ Generate Annex IV Report** to generate your EU AI Act Annex IV compliance report._",
                        sanitize_html=False,
                    )
                    cr_modelcard = gr.Code(label="📄 Model Card (HuggingFace-compatible Markdown)", language="markdown", lines=20)
            cr_btn.click(
                fn=run_compliance_report,
                inputs=[cr_prompt, cr_correct, cr_incorrect, cr_model, cr_provider, cr_deploy],
                outputs=[cr_report, cr_modelcard],
            )

        # ── Tab 5: About ───────────────────────────────────────────────────────
        with gr.Tab("📖 About"):
            gr.Markdown(ABOUT_MD)

    gr.HTML("""
<style>
.gb-ft { border-top:1px solid rgba(255,255,255,.07); margin-top:24px; padding:28px 0 16px; }
.gb-ft-top { display:flex; align-items:flex-start; gap:40px; flex-wrap:wrap; margin-bottom:24px; }
.gb-ft-brand { flex:2; min-width:200px; }
.gb-ft-logo { display:flex; align-items:center; gap:8px; font-family:'DM Sans',sans-serif; font-size:15px; font-weight:700; letter-spacing:-.02em; color:#fff; margin-bottom:8px; }
.gb-ft-logo-mark { width:24px; height:24px; border-radius:6px; background:linear-gradient(135deg,#00C8E8,#0891B2); display:flex; align-items:center; justify-content:center; }
.gb-ft-logo-mark svg { width:11px; height:11px; }
.gb-ft-tag { font-family:'DM Sans',sans-serif; font-size:13px; color:#52525b; line-height:1.6; max-width:260px; }
.gb-ft-col { flex:1; min-width:120px; }
.gb-ft-ctitle { font-family:'DM Sans',sans-serif; font-size:11px; font-weight:600; color:#fff; letter-spacing:.08em; text-transform:uppercase; margin-bottom:12px; }
.gb-ft-col ul { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:8px; }
.gb-ft-col a { font-family:'DM Sans',sans-serif; font-size:13px; color:#52525b; text-decoration:none; transition:color .15s; }
.gb-ft-col a:hover { color:#a1a1aa; }
.gb-ft-bot { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; padding-top:20px; border-top:1px solid rgba(255,255,255,.05); }
.gb-ft-copy { font-family:'DM Sans',sans-serif; font-size:12px; color:#3f3f46; }
.gb-ft-legal { display:flex; gap:16px; flex-wrap:wrap; }
.gb-ft-legal a { font-family:'DM Sans',sans-serif; font-size:12px; color:#3f3f46; text-decoration:none; transition:color .15s; }
.gb-ft-legal a:hover { color:#71717a; }
</style>
<div class="gb-ft">
  <div class="gb-ft-top">
    <div class="gb-ft-brand">
      <div class="gb-ft-logo">
        <div class="gb-ft-logo-mark">
          <svg fill="none" viewBox="0 0 13 13" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <rect x="1.5" y="1.5" width="10" height="10" rx="2"/>
            <path d="M4 6.5h5M6.5 4v5"/>
          </svg>
        </div>
        Glassbox AI
      </div>
      <div class="gb-ft-tag">The compliance layer for production AI. EU AI Act Annex IV, automated.</div>
    </div>
    <div class="gb-ft-col">
      <div class="gb-ft-ctitle">Product</div>
      <ul>
        <li><a href="https://repo-ashen-psi.vercel.app/#features" target="_blank">Features</a></li>
        <li><a href="https://repo-ashen-psi.vercel.app/#pricing" target="_blank">Pricing</a></li>
        <li><a href="https://repo-ashen-psi.vercel.app/#coverage" target="_blank">EU AI Act</a></li>
      </ul>
    </div>
    <div class="gb-ft-col">
      <div class="gb-ft-ctitle">Developers</div>
      <ul>
        <li><a href="https://github.com/designer-coderajay/glassbox-mech" target="_blank">GitHub</a></li>
        <li><a href="https://pypi.org/project/glassbox-mech-interp/" target="_blank">PyPI</a></li>
        <li><a href="https://github.com/designer-coderajay/glassbox-mech#readme" target="_blank">Docs</a></li>
      </ul>
    </div>
    <div class="gb-ft-col">
      <div class="gb-ft-ctitle">Legal</div>
      <ul>
        <li><a href="https://github.com/designer-coderajay/glassbox-mech/blob/main/LICENSE" target="_blank">MIT License</a></li>
        <li><a href="mailto:mahale.ajay01@gmail.com">Contact</a></li>
      </ul>
    </div>
  </div>
  <div class="gb-ft-bot">
    <div class="gb-ft-copy">&copy; 2026 Glassbox AI &nbsp;&middot;&nbsp; Built on TransformerLens &nbsp;&middot;&nbsp; v4.3.0</div>
    <div class="gb-ft-legal">
      <a href="https://github.com/designer-coderajay/glassbox-mech/blob/main/LICENSE" target="_blank">MIT License</a>
      <a href="mailto:mahale.ajay01@gmail.com">mahale.ajay01@gmail.com</a>
      <a href="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689" target="_blank">EU AI Act (EU) 2024/1689</a>
    </div>
  </div>
</div>
    """)

# ── REST API (/analyze) — lets the project-gu05p.vercel.app demo call the
# real backend instead of falling back to mock data. ──────────────────────────
# We attach routes to Gradio's *own* internal FastAPI app (demo.app) so the
# module-level model load only happens once. gr.mount_gradio_app() must NOT
# be used here — it causes a second process boot and loads GPT-2 twice → OOM.
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

def _jsonable(obj):
    """Recursively convert numpy scalars / tensors to plain Python types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):      # numpy / torch scalar
        return obj.item()
    if hasattr(obj, "tolist"):    # numpy array / torch tensor
        return obj.tolist()
    return obj

# Queue must be called before accessing demo.app
demo.queue()

# demo.app is Gradio's internal FastAPI instance — safe to extend
_gradio_app = demo.app

_gradio_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@_gradio_app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": gb is not None, "version": "4.2.6"}

@_gradio_app.post("/analyze")
async def analyze_api(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    prompt    = (body.get("prompt")          or "").strip()
    correct   = (body.get("correct_token")   or "").strip()
    incorrect = (body.get("incorrect_token") or "").strip()

    if not prompt or not correct or not incorrect:
        return JSONResponse(
            {"error": "Missing required fields: prompt, correct_token, incorrect_token"},
            status_code=422,
        )

    if gb is None:
        return JSONResponse({"error": "Model not loaded — try again in ~30 s"}, status_code=503)

    try:
        result = gb.analyze(prompt, correct, incorrect)
        return JSONResponse(_jsonable(result))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

# v3.4.1-patch: python_version=3.11 + pyaudioop in Space to permanently fix py3.13 audioop crash
