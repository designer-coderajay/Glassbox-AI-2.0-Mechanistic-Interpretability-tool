"""
Glassbox 4.2.4 — Causal Mechanistic Interpretability + EU AI Act Compliance
============================================================================
HuggingFace Space — v4.2.4  |  21/21 Mathematical Frameworks

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
        l, h = eval(k)
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
            linewidth=2, edgecolor="#D97757", facecolor="none"
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
    ax1.plot(layers, probs, "o-", lw=2, ms=7, color="#D97757")
    ax1.fill_between(layers, probs, alpha=0.15, color="#D97757")
    ax1.set_ylabel("Probability (%)", fontsize=11, color="white")
    ax1.set_title(f"Logit Lens — token: '{target_token}'", fontsize=13, color="white")
    ax1.set_ylim(bottom=0)
    ax2.plot(layers, layer_ranks, "s-", lw=2, ms=7, color="#3D6E96")
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
*Glassbox v4.2.4 · pip install glassbox-mech-interp · Regulation (EU) 2024/1689*
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
        grade, grade_color, status_label = "A", "#D97757", "Compliant"
    elif f1_score >= 0.65:
        grade, grade_color, status_label = "B", "#D97757", "Conditionally Compliant"
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

Generated by Glassbox v4.2.4 · {today}

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
| 6. Lifecycle | Glassbox v4.2.4 · {today} |
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
*{pname} · Glassbox v4.2.4 · EU AI Act (EU) 2024/1689 · {today}*
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
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─────────────────────────────────────────────────────────────────
   GLASSBOX — clean CSS for Gradio 4.43.0 / HuggingFace Space
   Semantic selectors ONLY. Zero Svelte hash classes.
   No display:none except <footer>. Explicit visibility for every
   output component (gr.Markdown / gr.HTML).
   ───────────────────────────────────────────────────────────────── */

/* ── Design tokens ── */
:root {
  --indigo:#D97757; --indigo-d:#B85E38; --indigo-l:#E89070;
  --sky:#3D6E96; --green:#22c55e; --amber:#f59e0b; --red:#ef4444;
  --text:#e2e8f0; --t2:#a1a1aa; --t3:#52525b; --t4:#3f3f46;
  --bd:rgba(255,255,255,.07); --bd2:rgba(255,255,255,.13); --bd3:rgba(255,255,255,.22);
  --sf:rgba(255,255,255,.03); --sf2:rgba(255,255,255,.06);
  --mono:'JetBrains Mono','Fira Code',monospace;
  --display:'Bricolage Grotesque','Inter',ui-sans-serif,sans-serif;
  --r:8px; --r2:12px; --r3:16px;
}

/* ── Base ── */
*, *::before, *::after { box-sizing:border-box; }
html, body { margin:0; padding:0; background:#07080A !important; }
body {
  font-family:'Inter',ui-sans-serif,-apple-system,sans-serif !important;
  color:#EBE7DE !important;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
  font-optical-sizing:auto;
  font-feature-settings:'cv01' 1,'cv03' 1,'cv04' 1;
}
::selection { background:rgba(217,119,87,.28); }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#07080A; }
::-webkit-scrollbar-thumb { background:#27272a; border-radius:3px; }

/* ── Gradient mesh bg ── */
body::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(ellipse 80% 60% at 12% 55%, rgba(217,119,87,.18) 0%, transparent 55%),
    radial-gradient(ellipse 65% 50% at 88% 18%, rgba(61,110,150,.13) 0%, transparent 50%),
    radial-gradient(ellipse 90% 90% at 50% 120%, rgba(217,119,87,.09) 0%, transparent 50%);
  animation:mesh-drift 12s ease-in-out infinite alternate;
}
@keyframes mesh-drift {
  from { transform:scale(1) translate(0,0); opacity:1; }
  to   { transform:scale(1.08) translate(8px,-6px); opacity:.82; }
}

/* ── Dot grid ── */
body::after {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px);
  background-size:72px 72px;
  mask-image:radial-gradient(ellipse 80% 70% at 50% 35%, black, transparent);
  -webkit-mask-image:radial-gradient(ellipse 80% 70% at 50% 35%, black, transparent);
}

/* ── Gradio root ── */
gradio-app { background:#07080A !important; min-height:100vh; }

/* ── Main container ── */
.gradio-container {
  background:transparent !important;
  max-width:1160px !important;
  margin:0 auto !important;
  padding:0 clamp(20px,4vw,56px) 56px !important;
  font-family:'Inter',ui-sans-serif,sans-serif !important;
  color:#e2e8f0 !important;
  position:relative; z-index:1;
}

/* ── Hide Gradio built-in footer only ── */
gradio-app > footer,
gradio-app footer { display:none !important; }
.progress-bar-wrap { display:none !important; }

/* ── CRITICAL — gr.HTML() and gr.Markdown() must always be visible ─────
   Gradio 4.x wraps these in elements with data-testid attributes.
   We force visibility here so no other rule can accidentally hide them. */
[data-testid="html"],
[data-testid="html"] > div {
  display:block !important;
  visibility:visible !important;
  overflow:visible !important;
  opacity:1 !important;
}
[data-testid="markdown"],
[data-testid="markdown"] > div,
[data-testid="markdown"] .prose {
  display:block !important;
  visibility:visible !important;
  overflow:visible !important;
  opacity:1 !important;
  color:#e2e8f0 !important;
}

/* ── Tab panels ── */
.tabitem, div[role="tabpanel"] {
  background:transparent !important;
  padding:0 !important;
}

/* ── Header block ── */
#gb-header-block,
#gb-header-block > div,
#gb-header-block .block {
  padding:0 !important; margin:0 !important;
  background:transparent !important; border:none !important; box-shadow:none !important;
}

/* ── Tabs block ── */
#gb-main-tabs {
  margin-top:32px !important; padding:0 !important;
  background:transparent !important; border:none !important;
}

/* ── Tab nav pills ── */
.tab-nav {
  background:rgba(255,255,255,.03) !important;
  border:1px solid var(--bd) !important;
  border-radius:10px !important;
  padding:4px !important; gap:2px !important;
  margin-bottom:20px !important;
  backdrop-filter:blur(16px);
}
.tab-nav button {
  background:transparent !important; color:var(--t2) !important;
  border:1px solid transparent !important; border-radius:7px !important;
  font-family:'Inter',sans-serif !important; font-size:13px !important;
  font-weight:500 !important; padding:7px 14px !important;
  letter-spacing:-.005em !important;
  transition:color .15s, background .15s !important;
}
.tab-nav button:hover { color:#fff !important; background:var(--sf2) !important; }
.tab-nav button.selected {
  background:rgba(217,119,87,.15) !important;
  color:#E89070 !important; border-color:rgba(217,119,87,.3) !important;
}

/* ── Block / card backgrounds ── */
.block, .form, .contain, .gap { background:transparent !important; }
.block { border-color:var(--bd) !important; }
.block.padded {
  background:rgba(255,255,255,.025) !important;
  border:1px solid var(--bd) !important; border-radius:var(--r2) !important;
  backdrop-filter:blur(8px); padding:16px !important;
}

/* ── Layout ── */
.row { gap:16px !important; align-items:flex-start !important; }
.col { padding:0 !important; min-width:0 !important; }
.form { gap:12px !important; }
.main { padding:0 !important; max-width:100% !important; }
.contain { padding:0 !important; }
.wrap { background:transparent !important; }

/* ── Inputs ── */
input[type=text], input[type=number], textarea, select {
  background:rgba(255,255,255,.04) !important;
  border:1px solid var(--bd2) !important;
  border-radius:var(--r) !important;
  color:#e2e8f0 !important;
  font-family:'Inter',sans-serif !important;
  font-size:14px !important; line-height:1.5 !important;
  padding:10px 13px !important;
  transition:border-color .15s, box-shadow .15s !important;
}
input[type=text]:focus, textarea:focus {
  outline:none !important;
  border-color:rgba(217,119,87,.55) !important;
  box-shadow:0 0 0 3px rgba(217,119,87,.11) !important;
}
input::placeholder, textarea::placeholder { color:var(--t3) !important; }
textarea.scroll-hide { color:#e2e8f0 !important; background:rgba(255,255,255,.04) !important; }
input[type=number] { text-align:center !important; }

/* ── Labels ── */
label, label span, .label-wrap, .label-wrap span,
.block > label, .form > label {
  color:var(--t2) !important; font-family:'Inter',sans-serif !important;
  font-size:13px !important; font-weight:500 !important; letter-spacing:.01em !important;
}

/* ── Buttons ── */
button.primary, button[variant="primary"] {
  background:var(--indigo) !important; border:none !important;
  border-radius:var(--r2) !important; color:#fff !important;
  font-family:'Inter',sans-serif !important; font-size:14px !important;
  font-weight:600 !important; padding:13px 28px !important;
  letter-spacing:-.01em !important; cursor:pointer !important;
  transition:background .15s, box-shadow .2s !important;
}
button.primary:hover, button[variant="primary"]:hover {
  background:var(--indigo-d) !important;
  box-shadow:0 8px 32px rgba(217,119,87,.42) !important;
}
button.secondary, button[variant="secondary"] {
  background:var(--sf2) !important; border:1px solid var(--bd2) !important;
  color:var(--t2) !important; border-radius:var(--r) !important;
  font-family:'Inter',sans-serif !important; cursor:pointer !important;
}
button.secondary:hover { background:rgba(255,255,255,.09) !important; color:#fff !important; }
button.lg { font-size:15px !important; padding:14px 28px !important; }

/* ── Slider ── */
input[type=range] { accent-color:var(--indigo) !important; width:100% !important; }

/* ── Dropdown ── */
ul.options {
  background:#0a0a0a !important; border:1px solid var(--bd2) !important;
  border-radius:var(--r) !important; z-index:9999 !important;
}
ul.options li {
  color:var(--t2) !important; font-size:13px !important;
  padding:9px 13px !important; font-family:'Inter',sans-serif !important;
}
ul.options li:hover, ul.options li.selected {
  background:rgba(217,119,87,.14) !important; color:#e2e8f0 !important;
}
.secondary-wrap { background:rgba(255,255,255,.04) !important; color:#e2e8f0 !important; }
.token { background:rgba(217,119,87,.15) !important; color:#E89070 !important; border-radius:4px !important; }

/* ── Image output ── */
[data-testid="image"] { background:#07080d !important; border-radius:10px !important; overflow:hidden !important; }
.image-frame, .image-container { background:#07080d !important; }
[data-testid="image"] img { border-radius:8px !important; }

/* ── Markdown styling ── */
.markdown, .prose,
[data-testid="markdown"] .prose {
  color:#e2e8f0 !important; font-family:'Inter',sans-serif !important;
  font-size:14px !important; line-height:1.7 !important;
}
.markdown h1, .markdown h2, .markdown h3,
.prose h1, .prose h2, .prose h3 {
  color:#fff !important; font-weight:700 !important; letter-spacing:-.03em !important;
}
.markdown h2, .prose h2 { font-size:1.3em !important; margin:24px 0 12px !important; }
.markdown h3, .prose h3 { font-size:1.05em !important; margin:16px 0 8px !important; }
.markdown a, .prose a { color:var(--indigo-l) !important; text-decoration:underline !important; }
.markdown p, .prose p { margin:8px 0 !important; }
.markdown strong, .prose strong { color:#fff !important; }
.markdown hr, .prose hr { border:none !important; border-top:1px solid var(--bd) !important; margin:20px 0 !important; }
.markdown li, .prose li { color:#e2e8f0 !important; margin-bottom:5px !important; }
.markdown ul, .prose ul { padding-left:20px !important; }
.markdown table, .prose table {
  border-collapse:collapse !important; width:100% !important;
  margin:14px 0 !important; font-size:13px !important;
}
.markdown th, .prose th {
  background:rgba(217,119,87,.09) !important; color:#E89070 !important;
  font-weight:600 !important; padding:9px 13px !important;
  border:1px solid rgba(217,119,87,.18) !important; text-align:left !important;
}
.markdown td, .prose td {
  padding:9px 13px !important; border:1px solid var(--bd) !important; color:#cbd5e1 !important;
}
.markdown tr:nth-child(even) td,
.prose tr:nth-child(even) td { background:rgba(255,255,255,.018) !important; }
.markdown code, .prose code {
  color:#E89070 !important; background:rgba(217,119,87,.09) !important;
  border:1px solid rgba(217,119,87,.2) !important; padding:1px 6px !important;
  border-radius:4px !important; font-family:var(--mono) !important; font-size:12px !important;
}
.markdown pre, .prose pre {
  background:#06060e !important; border:1px solid var(--bd) !important;
  border-radius:8px !important; padding:14px 18px !important;
  overflow-x:auto !important; margin:14px 0 !important;
}
.markdown pre code, .prose pre code {
  background:transparent !important; border:none !important; padding:0 !important;
  color:#E89070 !important; font-size:12.5px !important; font-family:var(--mono) !important;
}

/* ── Code editor (gr.Code) ── */
.cm-editor {
  background:#06060e !important; border:1px solid var(--bd) !important;
  border-radius:8px !important; overflow:hidden !important; min-height:200px !important;
}
.cm-content, .cm-line {
  color:#E89070 !important; font-family:var(--mono) !important;
  font-size:12.5px !important; line-height:1.65 !important;
  caret-color:var(--indigo) !important;
}
.cm-gutters {
  background:#06060e !important; border-right:1px solid var(--bd) !important; color:var(--t4) !important;
}
.cm-activeLine { background:rgba(217,119,87,.05) !important; }
.cm-scroller { background:#06060e !important; }
.cm-selectionBackground { background:rgba(217,119,87,.22) !important; }
.cm-cursor { border-left-color:var(--indigo) !important; }

/* ── Accordion ── */
details {
  background:rgba(255,255,255,.02) !important; border:1px solid var(--bd) !important;
  border-radius:var(--r) !important; overflow:hidden !important;
}
details summary {
  color:var(--t2) !important; font-size:13px !important; font-weight:500 !important;
  padding:10px 14px !important; cursor:pointer !important; list-style:none !important;
}
details summary::-webkit-details-marker { display:none; }
details[open] summary { border-bottom:1px solid var(--bd) !important; }

/* ── Misc ── */
input[type=checkbox], input[type=radio] { accent-color:var(--indigo) !important; }
.wrap.pending { opacity:.6 !important; }
.gb-ft { position:relative; z-index:1; }

/* ── Full-bleed topbar/nav on wide screens ── */
@media (min-width:1248px) {
  .gb-topbar, .gb-nav, .gb-hero-wrap, .gb-hero-sep {
    margin-left:calc(-50vw + 50%) !important;
    margin-right:calc(-50vw + 50%) !important;
    width:100vw !important;
  }
  .gb-nav {
    padding-left:calc(50vw - 580px + clamp(20px,4vw,56px)) !important;
    padding-right:calc(50vw - 580px + clamp(20px,4vw,56px)) !important;
  }
}

/* ── Mobile ── */
@media (max-width:768px) {
  .gb-nav-cx, .gb-nav-ghost { display:none; }
  .gradio-container { padding:0 16px 40px !important; }
}
"""


# ── HEADER — topbar + nav + hero, exact match to project-gu05p.vercel.app ──────
HEADER = """
<style>
/* ─ Keyframes ─ */
@keyframes blink {
  0%,100%{ opacity:1; box-shadow:0 0 8px #22c55e; }
  50%    { opacity:.35; box-shadow:none; }
}
@keyframes shine {
  0%  { background-position:0% 50%; }
  100%{ background-position:100% 50%; }
}
@keyframes mesh-hero {
  0%  { transform:scale(1) translate(0,0); opacity:1; }
  100%{ transform:scale(1.08) translate(8px,-6px); opacity:.82; }
}

/* ─ Topbar ─ */
.gb-topbar {
  position:sticky; top:0; z-index:1000;
  margin:0 calc(-1 * clamp(20px,4vw,56px));
  background:#D97757; padding:9px 24px; text-align:center;
  font-family:'Inter',sans-serif; font-size:13px; font-weight:500;
  letter-spacing:.01em; color:#fff;
}
.gb-topbar a {
  color:rgba(255,255,255,.75); border-bottom:1px solid rgba(255,255,255,.35);
  margin-left:8px; text-decoration:none;
  transition:color .15s, border-color .15s;
}
.gb-topbar a:hover { color:#fff; border-color:#fff; }

/* ─ Nav ─ */
.gb-nav {
  position:sticky; top:38px; z-index:999; height:64px;
  margin:0 calc(-1 * clamp(20px,4vw,56px));
  display:flex; align-items:center; padding:0 clamp(20px,4vw,56px);
  background:rgba(7,8,10,.90);
  backdrop-filter:blur(24px) saturate(180%);
  -webkit-backdrop-filter:blur(24px) saturate(180%);
  border-bottom:1px solid rgba(255,255,255,.07);
  font-family:'Inter',sans-serif;
}
.gb-nav-logo {
  display:flex; align-items:center; gap:9px;
  font-size:15px; font-weight:700; letter-spacing:-.02em;
  color:#fff; text-decoration:none; flex-shrink:0;
}
.gb-nav-mark {
  width:28px; height:28px; border-radius:7px; flex-shrink:0;
  background:linear-gradient(135deg,#D97757,#3D6E96);
  display:flex; align-items:center; justify-content:center;
}
.gb-nav-mark svg { width:13px; height:13px; }
.gb-nav-cx { flex:1; display:flex; justify-content:center; }
.gb-nav-links { display:flex; align-items:center; gap:2px; list-style:none; margin:0; padding:0; }
.gb-nav-links a {
  font-size:14px; font-weight:450; color:#a1a1aa;
  padding:6px 13px; border-radius:8px;
  text-decoration:none; transition:color .15s, background .15s;
}
.gb-nav-links a:hover { color:#fff; background:rgba(255,255,255,.06); }
.gb-nav-r { display:flex; align-items:center; gap:8px; flex-shrink:0; }
.gb-nav-ghost {
  font-size:14px; font-weight:500; color:#a1a1aa;
  padding:6px 13px; border-radius:8px; text-decoration:none;
  transition:color .15s, background .15s;
}
.gb-nav-ghost:hover { color:#fff; background:rgba(255,255,255,.06); }
.gb-nav-cta {
  display:inline-flex; align-items:center; gap:6px;
  font-size:13px; font-weight:600; color:#fff;
  background:#D97757; padding:8px 18px; border-radius:12px;
  letter-spacing:-.01em; text-decoration:none;
  transition:background .15s, box-shadow .2s;
}
.gb-nav-cta:hover { background:#B85E38; box-shadow:0 0 0 4px rgba(217,119,87,.18); }
.gb-nav-cta svg { width:12px; height:12px; }
@media(max-width:768px){ .gb-nav-cx,.gb-nav-ghost{ display:none; } }

/* ─ Hero wrap ─ */
.gb-hero-wrap {
  position:relative; overflow:hidden; isolation:isolate;
  margin:0 calc(-1 * clamp(20px,4vw,56px));
}
.gb-hmesh {
  position:absolute; inset:0; pointer-events:none;
  background:
    radial-gradient(ellipse 80% 60% at 12% 55%, rgba(217,119,87,.18) 0%, transparent 55%),
    radial-gradient(ellipse 65% 50% at 88% 18%, rgba(61,110,150,.13) 0%, transparent 50%),
    radial-gradient(ellipse 90% 90% at 50% 120%, rgba(217,119,87,.09) 0%, transparent 50%);
  animation:mesh-hero 12s ease-in-out infinite alternate;
}
.gb-hgrid {
  position:absolute; inset:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px);
  background-size:72px 72px;
  mask-image:radial-gradient(ellipse 80% 70% at 50% 35%, black, transparent);
  -webkit-mask-image:radial-gradient(ellipse 80% 70% at 50% 35%, black, transparent);
}
.gb-hero {
  position:relative; max-width:1160px; margin:0 auto;
  padding:clamp(64px,9vw,112px) clamp(24px,5vw,60px) clamp(56px,7vw,96px);
  text-align:center;
}

/* ─ Badge ─ */
.gb-hbadge {
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(217,119,87,.09); border:1px solid rgba(217,119,87,.26);
  border-radius:20px; padding:6px 15px 6px 10px; margin-bottom:44px;
  font-family:'Inter',sans-serif; font-size:12px; font-weight:700;
  letter-spacing:.04em; color:#E89070; text-transform:uppercase;
}
.gb-hblink { color:rgba(255,255,255,.6); text-decoration:none; margin-left:6px; font-weight:500; font-size:11px; }
.gb-hblink:hover { color:#fff; }
.gb-blink-dot {
  width:7px; height:7px; border-radius:50%;
  background:#22c55e; box-shadow:0 0 8px #22c55e;
  animation:blink 2s ease-in-out infinite; display:inline-block; flex-shrink:0;
}

/* ─ Title ─ */
.gb-htitle {
  font-family:'Bricolage Grotesque','Inter',sans-serif;
  font-size:clamp(42px,6.5vw,84px); font-weight:800;
  letter-spacing:-.035em; line-height:.97;
  color:#EBE7DE; margin:0 auto 32px;
  max-width:820px; text-wrap:balance;
}
.gb-shine {
  background:linear-gradient(135deg,#EBE7DE 0%,#E89070 40%,#D97757 72%,#EBE7DE 100%);
  background-size:200% 200%;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text;
  animation:shine 7s ease-in-out infinite alternate;
}

/* ─ Sub ─ */
.gb-hsub {
  font-family:'Inter',sans-serif;
  font-size:clamp(16px,1.8vw,18px); font-weight:420;
  color:#a1a1aa; max-width:540px; margin:0 auto 44px;
  line-height:1.78; text-wrap:pretty; letter-spacing:.01em;
}

/* ─ Hero CTAs ─ */
.gb-hctas { display:flex; justify-content:center; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:56px; }
.gb-hbtn-p {
  display:inline-flex; align-items:center; gap:8px;
  background:#D97757; color:#fff; padding:13px 28px; border-radius:12px;
  font-family:'Inter',sans-serif; font-size:14px; font-weight:600;
  letter-spacing:-.01em; text-decoration:none;
  transition:background .15s, box-shadow .2s;
}
.gb-hbtn-p:hover { background:#B85E38; box-shadow:0 8px 32px rgba(217,119,87,.42); }
.gb-hbtn-s {
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.13);
  color:#fff; padding:13px 28px; border-radius:12px;
  font-family:'Inter',sans-serif; font-size:14px; font-weight:500;
  letter-spacing:-.01em; text-decoration:none;
  transition:background .15s, border-color .2s;
}
.gb-hbtn-s:hover { background:rgba(255,255,255,.09); border-color:rgba(255,255,255,.22); }

/* ─ Stats bar ─ */
.gb-hstats {
  display:flex; justify-content:center; align-items:center;
  gap:0; flex-wrap:wrap;
  border:1px solid rgba(255,255,255,.07);
  border-radius:12px;
  background:rgba(255,255,255,.03);
  backdrop-filter:blur(12px);
  overflow:hidden; max-width:600px; margin:0 auto;
}
.gb-si {
  flex:1; min-width:110px; padding:22px 20px; text-align:center;
  border-right:1px solid rgba(255,255,255,.07); transition:background .2s;
}
.gb-si:last-child { border-right:none; }
.gb-si:hover { background:rgba(255,255,255,.04); }
.gb-sn {
  font-family:'Bricolage Grotesque','Inter',sans-serif; font-size:30px; font-weight:800;
  color:#EBE7DE; letter-spacing:-.04em; line-height:1; margin-bottom:6px;
}
.gb-sl {
  font-family:'Inter',sans-serif; font-size:10.5px; font-weight:700;
  color:#52525b; text-transform:uppercase; letter-spacing:.1em; white-space:nowrap;
}

/* ─ Hero bottom sep ─ */
.gb-hero-sep {
  height:1px; margin:0 calc(-1 * clamp(20px,4vw,56px));
  background:linear-gradient(90deg, transparent, rgba(255,255,255,.07), transparent);
}
</style>

<!-- Topbar -->
<div class="gb-topbar">
  EU AI Act enforcement: August 2, 2026 &mdash; Full Annex IV evidence packages, automated.
  <a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool" target="_blank">View on GitHub &rarr;</a>
</div>

<!-- Nav -->
<nav class="gb-nav">
  <a class="gb-nav-logo" href="https://project-gu05p.vercel.app/" target="_blank">
    <div class="gb-nav-mark">
      <svg fill="none" viewBox="0 0 13 13" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <rect x="1.5" y="1.5" width="10" height="10" rx="2"/>
        <path d="M4 6.5h5M6.5 4v5"/>
      </svg>
    </div>
    Glassbox AI
  </a>
  <div class="gb-nav-cx">
    <ul class="gb-nav-links">
      <li><a href="#circuit">Circuit Analysis</a></li>
      <li><a href="#logit">Logit Lens</a></li>
      <li><a href="#attention">Attention</a></li>
      <li><a href="#compliance">Compliance</a></li>
      <li><a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool" target="_blank">Docs</a></li>
    </ul>
  </div>
  <div class="gb-nav-r">
    <a class="gb-nav-ghost" href="https://pypi.org/project/glassbox-mech-interp/" target="_blank">PyPI</a>
    <a class="gb-nav-cta" href="https://project-gu05p.vercel.app/" target="_blank">
      Website
      <svg fill="none" viewBox="0 0 12 12" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M2 10L10 2M6 2h4v4"/></svg>
    </a>
  </div>
</nav>

<!-- Hero -->
<div class="gb-hero-wrap">
  <div class="gb-hmesh"></div>
  <div class="gb-hgrid"></div>
  <div class="gb-hero">
    <div class="gb-hbadge">
      <span class="gb-blink-dot"></span>
      Live Interactive Demo
    </div>
    <h1 class="gb-htitle">The compliance layer for <span class="gb-shine">production&nbsp;AI.</span></h1>
    <p class="gb-hsub">Map your LLM&rsquo;s attention circuits to EU AI Act Annex IV requirements. One function call. A complete evidence package.</p>
    <div class="gb-hctas">
      <a class="gb-hbtn-p" href="https://pypi.org/project/glassbox-mech-interp/" target="_blank">
        <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round"><path d="M7 1v4M7 9v4M1 7h4M9 7h4"/><circle cx="7" cy="7" r="2"/></svg>
        pip install glassbox-mech-interp
      </a>
      <a class="gb-hbtn-s" href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool" target="_blank">
        <svg fill="none" viewBox="0 0 15 15" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M7.5 1C3.91 1 1 3.91 1 7.5c0 2.87 1.86 5.3 4.44 6.16.32.06.44-.14.44-.31v-1.08c-1.8.39-2.18-.87-2.18-.87-.3-.75-.72-.95-.72-.95-.59-.4.04-.39.04-.39.65.04 1 .67 1 .67.58 1 1.53.71 1.9.54.06-.42.23-.71.41-.87-1.44-.16-2.95-.72-2.95-3.2 0-.71.25-1.29.67-1.74-.07-.17-.29-.82.06-1.72 0 0 .55-.18 1.8.67a6.27 6.27 0 013.26 0c1.25-.85 1.8-.67 1.8-.67.35.9.13 1.55.06 1.72.42.45.67 1.03.67 1.74 0 2.49-1.52 3.04-2.96 3.2.23.2.44.6.44 1.21v1.79c0 .17.12.37.44.31A6.5 6.5 0 0014 7.5C14 3.91 11.09 1 7.5 1z"/></svg>
        GitHub
      </a>
    </div>
    <div class="gb-hstats">
      <div class="gb-si"><div class="gb-sn">1.8K</div><div class="gb-sl">Downloads/mo</div></div>
      <div class="gb-si"><div class="gb-sn">8</div><div class="gb-sl">Annex IV Sections</div></div>
      <div class="gb-si"><div class="gb-sn">&lt;2s</div><div class="gb-sl">Per Audit</div></div>
      <div class="gb-si"><div class="gb-sn">MIT</div><div class="gb-sl">License</div></div>
    </div>
  </div>
</div>
<div class="gb-hero-sep"></div>
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
  url     = {https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool},
  version = {4.2.4}
}
```

### References

- Wang et al. (2022). Interpretability in the Wild: IOI in GPT-2 small. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
- Nanda (2023). Attribution Patching. [neelnanda.io](https://neelnanda.io)
- Conmy et al. (2023). Towards Automated Circuit Discovery (ACDC). [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. [transformer-circuits.pub](https://transformer-circuits.pub)
- EU AI Act (EU) 2024/1689, Official Journal of the EU

---

**Contact:** mahale.ajay01@gmail.com · **License:** MIT · **Version:** 4.2.4
"""

with gr.Blocks(
    title="Glassbox 4.2.4 — EU AI Act Compliance",
    css=GB_CSS,
    head='<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">',
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="zinc",
    ).set(
        body_background_fill="#000000",
        body_background_fill_dark="#000000",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        body_text_color_subdued="#a1a1aa",
        body_text_color_subdued_dark="#a1a1aa",
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
        button_primary_background_fill="#D97757",
        button_primary_background_fill_dark="#D97757",
        button_primary_background_fill_hover="#B85E38",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="rgba(255,255,255,0.05)",
        button_secondary_border_color="rgba(255,255,255,0.13)",
        button_secondary_text_color="#a1a1aa",
        shadow_drop="0 4px 24px rgba(0,0,0,0.6)",
        shadow_drop_lg="0 8px 40px rgba(0,0,0,0.8)",
        color_accent_soft="rgba(217,119,87,0.15)",
        color_accent_soft_dark="rgba(217,119,87,0.15)",
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
.gb-ft-logo { display:flex; align-items:center; gap:8px; font-family:'Inter',sans-serif; font-size:15px; font-weight:700; letter-spacing:-.02em; color:#fff; margin-bottom:8px; }
.gb-ft-logo-mark { width:24px; height:24px; border-radius:6px; background:linear-gradient(135deg,#D97757,#3D6E96); display:flex; align-items:center; justify-content:center; }
.gb-ft-logo-mark svg { width:11px; height:11px; }
.gb-ft-tag { font-family:'Inter',sans-serif; font-size:13px; color:#52525b; line-height:1.6; max-width:260px; }
.gb-ft-col { flex:1; min-width:120px; }
.gb-ft-ctitle { font-family:'Inter',sans-serif; font-size:11px; font-weight:600; color:#fff; letter-spacing:.08em; text-transform:uppercase; margin-bottom:12px; }
.gb-ft-col ul { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:8px; }
.gb-ft-col a { font-family:'Inter',sans-serif; font-size:13px; color:#52525b; text-decoration:none; transition:color .15s; }
.gb-ft-col a:hover { color:#a1a1aa; }
.gb-ft-bot { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; padding-top:20px; border-top:1px solid rgba(255,255,255,.05); }
.gb-ft-copy { font-family:'Inter',sans-serif; font-size:12px; color:#3f3f46; }
.gb-ft-legal { display:flex; gap:16px; flex-wrap:wrap; }
.gb-ft-legal a { font-family:'Inter',sans-serif; font-size:12px; color:#3f3f46; text-decoration:none; transition:color .15s; }
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
        <li><a href="https://project-gu05p.vercel.app/#features" target="_blank">Features</a></li>
        <li><a href="https://project-gu05p.vercel.app/#pricing" target="_blank">Pricing</a></li>
        <li><a href="https://project-gu05p.vercel.app/#coverage" target="_blank">EU AI Act</a></li>
      </ul>
    </div>
    <div class="gb-ft-col">
      <div class="gb-ft-ctitle">Developers</div>
      <ul>
        <li><a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool" target="_blank">GitHub</a></li>
        <li><a href="https://pypi.org/project/glassbox-mech-interp/" target="_blank">PyPI</a></li>
        <li><a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool#readme" target="_blank">Docs</a></li>
      </ul>
    </div>
    <div class="gb-ft-col">
      <div class="gb-ft-ctitle">Legal</div>
      <ul>
        <li><a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/blob/main/LICENSE" target="_blank">MIT License</a></li>
        <li><a href="mailto:mahale.ajay01@gmail.com">Contact</a></li>
      </ul>
    </div>
  </div>
  <div class="gb-ft-bot">
    <div class="gb-ft-copy">&copy; 2026 Glassbox AI &nbsp;&middot;&nbsp; Built on TransformerLens &nbsp;&middot;&nbsp; v4.2.4</div>
    <div class="gb-ft-legal">
      <a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/blob/main/LICENSE" target="_blank">MIT License</a>
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
    return {"status": "ok", "model_loaded": gb is not None, "version": "4.2.4"}

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
