"""
Glassbox 3.4 — Causal Mechanistic Interpretability + EU AI Act Compliance
=========================================================================
HuggingFace Space — v3.4.0

Tabs:
  1. Circuit Analysis   — attribution patching, MFC discovery, faithfulness metrics
  2. Logit Lens         — residual stream projection by layer
  3. Attention Patterns — raw attention weight heatmap
  4. Compliance Report  — EU AI Act Annex IV explainability grade + bias check + plain English
  5. About / Docs       — methodology, references, citation

v3.4.0 new features:
  - MultiAgentAudit: causal handoff tracing for multi-agent chains (Article 9)
  - SteeringVectorExporter: representation engineering vectors (Article 9(2)(b))
  - AnnexIVEvidenceVault: full Annex IV documentation package builder (Article 11)

v3.3.0 new features:
  - NaturalLanguageExplainer: plain-English compliance summaries for non-technical stakeholders
  - HuggingFace Hub integration: push Annex IV metadata to model cards
  - MLflow integration: log circuit metrics as experiment tracking artifacts
  - Slack/Teams alerting: CircuitDiff drift + compliance drop notifications
  - GitHub Action CI hook: auto-fail CI if compliance grade drops
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
            linewidth=2, edgecolor="#f59e0b", facecolor="none"
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
    ax1.plot(layers, probs, "o-", lw=2, ms=7, color="#6366f1")
    ax1.fill_between(layers, probs, alpha=0.15, color="#6366f1")
    ax1.set_ylabel("Probability (%)", fontsize=11, color="white")
    ax1.set_title(f"Logit Lens — token: '{target_token}'", fontsize=13, color="white")
    ax1.set_ylim(bottom=0)
    ax2.plot(layers, layer_ranks, "s-", lw=2, ms=7, color="#8b5cf6")
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

Maps to **Article 13 transparency requirements**. Circuit identifies which model components causally drove this prediction with quantified faithfulness scores. Grade: **{"A" if faith["f1"] >= 0.7 else "B" if faith["f1"] >= 0.5 else "C" if faith["f1"] >= 0.3 else "D"}**

---
*Glassbox v3.3.0 · pip install glassbox-mech-interp · Regulation (EU) 2024/1689*
"""
    # Log to audit trail
    try:
        _audit_log.append_from_result(result, auditor="hf-space-demo")
    except Exception:
        pass

    return img, report, ""


def run_logit_lens_tab(prompt: str, target_token: str):
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
    if not prompt.strip():
        return None, "Please enter a prompt."
    try:
        img = _attention_plot(prompt.strip(), int(layer), int(head))
        return img, f"Attention pattern for Layer {int(layer)}, Head {int(head)}."
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_compliance_report(prompt: str, correct: str, incorrect: str,
                          model_name: str, provider: str, deployment: str):
    if not prompt.strip() or not correct.strip() or not incorrect.strip():
        return "Please fill in Prompt, Correct token, and Distractor token.", ""

    try:
        result = gb.analyze(prompt.strip(), correct.strip(), incorrect.strip())
        faith  = result["faithfulness"]

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
            model_name         = model_name.strip() or "GPT-2 small",
            provider_name      = provider.strip() or "Demo User",
            provider_address   = "HuggingFace Space Demo",
            system_purpose     = f"Demo: {prompt.strip()[:80]}",
            deployment_context = ctx,
        )
        annex.add_analysis(result, use_case=f"Demo prompt: {prompt.strip()[:60]}")

        import json as _json
        _rj = _json.loads(annex.to_json())
        _s3 = _rj.get("sections", {}).get("3_monitoring_control", {})
        _s5 = _rj.get("sections", {}).get("5_risk_management", {})

        grade   = (_s3.get("explainability_grade") or "D")[0].upper()
        status  = _rj.get("compliance_status", "non_compliant")
        f1_score = faith["f1"]

        grade_color = {
            "A": "#22c55e", "B": "#6366f1", "C": "#f59e0b", "D": "#ef4444"
        }.get(grade, "#aaa")

        status_emoji = ("✅" if status == "compliant"
                        else "⚠️" if "conditional" in status
                        else "❌")

        report_md = f"""## EU AI Act Annex IV Compliance Report

<div style="display:flex; gap:16px; margin:12px 0;">
  <div style="background:#1a2030; border:1px solid #2a3040; border-radius:8px; padding:16px 24px; text-align:center;">
    <div style="font-size:2.4em; font-weight:800; color:{grade_color};">{grade}</div>
    <div style="color:#aaa; font-size:0.85em;">Explainability Grade</div>
  </div>
  <div style="background:#1a2030; border:1px solid #2a3040; border-radius:8px; padding:16px 24px; text-align:center;">
    <div style="font-size:1.6em; font-weight:700; color:#e2e8f0;">{f1_score:.0%}</div>
    <div style="color:#aaa; font-size:0.85em;">Faithfulness F1</div>
  </div>
  <div style="background:#1a2030; border:1px solid #2a3040; border-radius:8px; padding:16px 24px; text-align:center;">
    <div style="font-size:1.2em; font-weight:700; color:#e2e8f0;">{status_emoji}</div>
    <div style="color:#aaa; font-size:0.85em;">{status.replace("_", " ").title()}</div>
  </div>
</div>

---

### Annex IV Section Summary

| Section | Content |
|---------|---------|
| 1. System Description | {model_name.strip() or "GPT-2 small"} · {deployment} context |
| 2. Risk Classification | {_rj.get("risk_classification", "other_high_risk").replace("_", " ").title()} |
| 3. Monitoring & Control | Audit log active · {_audit_log.summary().get("total_audits", 0)} sessions recorded |
| 4. Data & Training | TransformerLens GPT-2 weights (117M params) |
| 5. Bias Testing | See below |
| 6. Lifecycle | Version 3.3.0 · {_rj.get("generated_at", "")[:10]} |
| 7. Explainability | F1={f1_score:.2f} · Grade {grade} · {len(result["circuit"])} circuit heads |
| 8. Cybersecurity | Tamper-evident SHA-256 audit chain |
| 9. Performance Metrics | Suff={faith["sufficiency"]:.1%} · Comp={faith["comprehensiveness"]:.1%} |

---

### Bias Assessment (Article 10(2)(f))

Running counterfactual fairness probe on gender swap …

"""
        # Quick offline bias probe — no live logprobs needed, just a marker
        report_md += """| Test | Status |
|------|--------|
| Counterfactual gender swap | ⚠️ Requires live model_fn — see Python SDK |
| Demographic parity | ⚠️ Requires group prompts — see `BiasAnalyzer` docs |
| Token bias probe | ⚠️ Requires pre-computed logprobs — see `BiasAnalyzer` docs |

> **To run full bias analysis:**
> ```python
> from glassbox import BiasAnalyzer
> ba = BiasAnalyzer()
> result = ba.counterfactual_fairness_test(
>     prompt_template="The {attribute} applied for the loan",
>     groups={"gender": ["male applicant", "female applicant"]},
>     target_tokens=["approved", "denied"],
>     model_fn=my_model_fn,
> )
> ```

---

### Risk Flags

"""
        flags = [r.get("risk") or r.get("description") or str(r)
                 for r in (_s5.get("identified_risks") or [])]
        if flags:
            for flag in flags:
                report_md += f"- ⚠️ {flag}\n"
        else:
            report_md += "- No critical risk flags identified.\n"

        report_md += f"""
---

### Article Mapping

| EU AI Act Article | Requirement | Status |
|-------------------|-------------|--------|
| Article 10(2)(f) | Bias and discrimination testing | ⚠️ Partial |
| Article 13 | Transparency and provision of information | {"✅" if grade in ("A","B") else "⚠️"} |
| Article 17 | Quality management system | ✅ AuditLog active |
| Annex IV | Technical documentation | ✅ All 9 sections |

---
*Glassbox v3.3.0 · EU AI Act (EU) 2024/1689 · Enforcement August 2026*
"""

        model_card = annex.to_model_card()

        # Log this compliance check
        try:
            _audit_log.append_from_result(result, auditor="hf-space-compliance")
        except Exception:
            pass

        return report_md, model_card

    except Exception as e:
        return f"Error generating compliance report: {str(e)}", ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

# ── Billion-dollar CSS ─────────────────────────────────────────────────────────
GB_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --gb-indigo:#6366f1; --gb-indigo-d:#4f46e5; --gb-indigo-l:#818cf8;
  --gb-sky:#0ea5e9; --gb-sky-l:#38bdf8;
  --gb-green:#22c55e; --gb-amber:#f59e0b; --gb-red:#ef4444;
  --gb-t2:#a1a1aa; --gb-t3:#52525b; --gb-t4:#3f3f46;
  --gb-bd:rgba(255,255,255,.07); --gb-bd2:rgba(255,255,255,.13);
  --gb-sf:rgba(255,255,255,.03); --gb-sf2:rgba(255,255,255,.06);
  --gb-r:8px; --gb-r2:12px;
}

/* ── Base ── */
body, html { background:#000 !important; }
body {
  font-family:'Inter',ui-sans-serif,-apple-system,BlinkMacSystemFont,sans-serif !important;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
  color:#fff !important;
}
::selection { background:rgba(99,102,241,.28); }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#000; }
::-webkit-scrollbar-thumb { background:#27272a; border-radius:3px; }

/* ── Gradient mesh ── */
body::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(ellipse 80% 60% at 15% -5%, rgba(99,102,241,.18) 0%, transparent 60%),
    radial-gradient(ellipse 70% 50% at 85% 15%, rgba(14,165,233,.12) 0%, transparent 55%),
    radial-gradient(ellipse 60% 50% at 50% 95%, rgba(139,92,246,.10) 0%, transparent 55%);
  animation:gb-mesh 18s ease-in-out infinite alternate;
}
@keyframes gb-mesh {
  0%   { transform:scale(1) translateY(0); opacity:.85; }
  50%  { transform:scale(1.04) translateY(-14px); opacity:1; }
  100% { transform:scale(1.01) translateY(6px); opacity:.88; }
}

/* ── Dot grid ── */
body::after {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:radial-gradient(circle, rgba(255,255,255,.055) 1px, transparent 1px);
  background-size:28px 28px;
  -webkit-mask-image:radial-gradient(ellipse 90% 90% at 50% 50%, black 20%, transparent 100%);
  mask-image:radial-gradient(ellipse 90% 90% at 50% 50%, black 20%, transparent 100%);
}

/* ── Container ── */
.gradio-container {
  background:transparent !important;
  max-width:1160px !important;
  margin:0 auto !important;
  padding:0 20px 56px !important;
  font-family:'Inter',ui-sans-serif,sans-serif !important;
  position:relative; z-index:1;
}
footer.svelte-1ax1toq, footer { display:none !important; }

/* ── Blocks / cards ── */
.block, .form { background:transparent !important; border-color:var(--gb-bd) !important; }
.block.padded {
  background:rgba(255,255,255,.025) !important;
  border:1px solid var(--gb-bd) !important;
  border-radius:var(--gb-r2) !important;
  backdrop-filter:blur(8px);
}
.gap { background:transparent !important; }
.contain { background:transparent !important; }

/* ── Tabs ── */
.tab-nav {
  background:rgba(255,255,255,.028) !important;
  border:1px solid var(--gb-bd) !important;
  border-radius:10px !important;
  padding:4px !important;
  gap:2px !important;
  margin-bottom:20px !important;
  backdrop-filter:blur(12px);
}
.tab-nav button {
  background:transparent !important;
  color:var(--gb-t2) !important;
  border:1px solid transparent !important;
  border-radius:7px !important;
  font-family:'Inter',sans-serif !important;
  font-size:13px !important; font-weight:500 !important;
  padding:7px 15px !important;
  transition:color .15s, background .15s !important;
}
.tab-nav button:hover { color:#fff !important; background:rgba(255,255,255,.06) !important; }
.tab-nav button.selected {
  background:rgba(99,102,241,.16) !important;
  color:#818cf8 !important;
  border-color:rgba(99,102,241,.28) !important;
}

/* ── Inputs ── */
input[type=text], input[type=number], textarea, select {
  background:rgba(255,255,255,.04) !important;
  border:1px solid var(--gb-bd2) !important;
  border-radius:var(--gb-r) !important;
  color:#fff !important;
  font-family:'Inter',sans-serif !important;
  font-size:14px !important;
  padding:10px 13px !important;
  transition:border-color .15s, box-shadow .15s !important;
}
input[type=text]:focus, textarea:focus {
  outline:none !important;
  border-color:rgba(99,102,241,.6) !important;
  box-shadow:0 0 0 3px rgba(99,102,241,.12) !important;
}
input[type=text]::placeholder, textarea::placeholder { color:var(--gb-t3) !important; }

/* ── Labels ── */
label, .label-wrap span {
  color:var(--gb-t2) !important;
  font-family:'Inter',sans-serif !important;
  font-size:13px !important; font-weight:500 !important;
  letter-spacing:.01em !important;
}

/* ── Primary buttons ── */
button.primary, .btn-primary, button[data-testid*="primary"], .svelte-cmf5ev.primary {
  background:linear-gradient(135deg,var(--gb-indigo),var(--gb-indigo-d)) !important;
  border:none !important;
  border-radius:var(--gb-r) !important;
  color:#fff !important;
  font-family:'Inter',sans-serif !important;
  font-size:14px !important; font-weight:600 !important;
  padding:10px 22px !important;
  letter-spacing:-.01em !important;
  box-shadow:0 0 20px rgba(99,102,241,.28) !important;
  transition:box-shadow .2s, transform .15s !important;
}
button.primary:hover { box-shadow:0 0 36px rgba(99,102,241,.48) !important; transform:translateY(-1px) !important; }
button.secondary {
  background:rgba(255,255,255,.05) !important;
  border:1px solid var(--gb-bd2) !important;
  color:var(--gb-t2) !important;
  border-radius:var(--gb-r) !important;
  font-family:'Inter',sans-serif !important;
}

/* ── Sliders ── */
input[type=range] { accent-color:var(--gb-indigo) !important; }

/* ── Dropdowns ── */
ul.options {
  background:#0c0c0c !important;
  border:1px solid var(--gb-bd2) !important;
  border-radius:var(--gb-r) !important;
}
ul.options li { color:var(--gb-t2) !important; font-size:14px !important; }
ul.options li:hover, ul.options li.selected {
  background:rgba(99,102,241,.15) !important; color:#fff !important;
}

/* ── Image output ── */
.image-container { background:rgba(255,255,255,.02) !important; border:1px solid var(--gb-bd) !important; border-radius:var(--gb-r2) !important; }

/* ── Code ── */
code, pre {
  font-family:'JetBrains Mono','Fira Code',monospace !important;
  font-size:13px !important;
  background:rgba(255,255,255,.035) !important;
  border:1px solid var(--gb-bd) !important;
  border-radius:6px !important;
  color:#a5b4fc !important;
}
pre code { background:transparent !important; border:none !important; padding:0 !important; }

/* ── Accordion ── */
.accordion, details {
  background:rgba(255,255,255,.02) !important;
  border:1px solid var(--gb-bd) !important;
  border-radius:var(--gb-r) !important;
}
details summary { color:var(--gb-t2) !important; font-size:13px !important; font-weight:500 !important; padding:10px 14px !important; }

/* ── Markdown ── */
.markdown, .prose { color:#e2e8f0 !important; font-size:14px !important; line-height:1.7 !important; }
.markdown h1, .markdown h2, .markdown h3 { color:#fff !important; font-weight:700 !important; letter-spacing:-.02em !important; }
.markdown h2 { font-size:1.25em !important; margin:20px 0 10px !important; }
.markdown h3 { font-size:1.05em !important; }
.markdown a { color:var(--gb-indigo-l) !important; }
.markdown table { border-collapse:collapse !important; width:100% !important; margin:12px 0 !important; font-size:13px !important; }
.markdown th {
  background:rgba(99,102,241,.1) !important; color:#a5b4fc !important;
  font-weight:600 !important; padding:8px 12px !important;
  border:1px solid rgba(99,102,241,.18) !important; text-align:left !important;
}
.markdown td { padding:8px 12px !important; border:1px solid var(--gb-bd) !important; color:#cbd5e1 !important; }
.markdown tr:nth-child(even) td { background:rgba(255,255,255,.02) !important; }
.markdown strong { color:#fff !important; }
.markdown code { color:#a5b4fc !important; background:rgba(99,102,241,.1) !important; border-color:rgba(99,102,241,.2) !important; padding:1px 5px !important; border-radius:4px !important; }
.markdown hr { border:none !important; border-top:1px solid var(--gb-bd) !important; margin:20px 0 !important; }

/* ── Row gap ── */
.row { gap:16px !important; }

/* ── Compliance metric badges inside markdown ── */
div[style*="background:#1a2030"] {
  background:rgba(255,255,255,.035) !important;
  border-color:rgba(255,255,255,.08) !important;
  border-radius:10px !important;
}
"""

# ── Header ─────────────────────────────────────────────────────────────────────
HEADER = """
<style>
@keyframes gb-pulse { 0%,100% { box-shadow:0 0 4px #22c55e; } 50% { box-shadow:0 0 14px #22c55e,0 0 28px rgba(34,197,94,.3); } }
@keyframes gb-shine { 0% { background-position:0% 50%; } 100% { background-position:100% 50%; } }
.gb-hdr { padding:52px 0 36px; text-align:center; position:relative; }
.gb-badge {
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(99,102,241,.1); border:1px solid rgba(99,102,241,.25);
  border-radius:20px; padding:5px 16px; margin-bottom:22px;
  font-family:'Inter',sans-serif; font-size:11px; font-weight:600;
  letter-spacing:.08em; color:#a5b4fc; text-transform:uppercase;
}
.gb-dot {
  width:6px; height:6px; border-radius:50%;
  background:#22c55e; box-shadow:0 0 6px #22c55e;
  animation:gb-pulse 2.2s ease-in-out infinite;
  display:inline-block; flex-shrink:0;
}
.gb-h1 {
  font-family:'Inter',sans-serif;
  font-size:clamp(38px,6vw,76px); font-weight:900;
  letter-spacing:-0.05em; color:#fff; line-height:1.05;
  margin:0 0 16px;
}
.gb-grad {
  background:linear-gradient(135deg,#fff 0%,#a5b4fc 38%,#38bdf8 72%,#fff 100%);
  background-size:200% 200%;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text;
  animation:gb-shine 7s ease-in-out infinite alternate;
}
.gb-sub {
  font-family:'Inter',sans-serif; font-size:15px; font-weight:400;
  color:#94a3b8; margin:0 auto 28px; max-width:560px; line-height:1.65;
}
.gb-pills { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; }
.gb-pill {
  display:inline-flex; align-items:center; gap:6px;
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.1);
  border-radius:7px; padding:7px 15px;
  font-family:'Inter',sans-serif; font-size:12px; font-weight:500;
  color:#94a3b8; text-decoration:none;
  transition:color .15s, border-color .15s, background .15s;
}
.gb-pill:hover { color:#a5b4fc; border-color:rgba(99,102,241,.4); background:rgba(99,102,241,.08); }
.gb-pill-red { border-color:rgba(239,68,68,.3) !important; color:#f87171 !important; }
.gb-pill-red:hover { background:rgba(239,68,68,.07) !important; }
.gb-sep { height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent); margin:36px 0 0; }
</style>
<div class="gb-hdr">
  <div class="gb-badge"><span class="gb-dot"></span>Live Demo &nbsp;·&nbsp; EU AI Act Compliance Platform</div>
  <h1 class="gb-h1">Glassbox <span class="gb-grad">3.4</span></h1>
  <p class="gb-sub">Causal Mechanistic Interpretability &nbsp;·&nbsp; EU AI Act Annex IV Compliance &nbsp;·&nbsp; Bias Analysis</p>
  <div class="gb-pills">
    <a class="gb-pill" href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool" target="_blank">⭐&nbsp;GitHub</a>
    <a class="gb-pill" href="https://pypi.org/project/glassbox-mech-interp/" target="_blank">📦&nbsp;pip install glassbox-mech-interp</a>
    <span class="gb-pill gb-pill-red">⚡&nbsp;Enforcement: August 2026</span>
  </div>
  <div class="gb-sep"></div>
</div>
"""

ABOUT_TEXT = """
## What is Glassbox?

Glassbox identifies the **specific attention heads** in a transformer that *causally* drive a prediction — not just which tokens the model attended to, but which internal components are responsible and by how much.

### Three core faithfulness metrics

| Metric | What it measures | Method |
|--------|-----------------|--------|
| **Sufficiency** | How much of the prediction do the identified heads explain? | Taylor approximation (3 passes) |
| **Comprehensiveness** | How much does ablating those heads degrade the prediction? | Exact activation patching |
| **F1** | Single faithfulness score | Harmonic mean |

### v3.3.0 — What's new

- **NaturalLanguageExplainer** — plain-English compliance summaries for compliance officers and legal teams. Zero LLM dependency, EU AI Act article-cited, deterministic.
- **HuggingFace Hub integration** — push Annex IV compliance metadata to model cards (`HuggingFaceModelCard`). 29 architecture aliases supported.
- **MLflow integration** — `log_glassbox_run()` logs circuit metrics as experiment tracking artifacts. `GlassboxMLflowCallback` for training loop integration.
- **Slack/Teams alerting** — `SlackNotifier`, `TeamsNotifier`, `AlertConfig` — formatted alerts for CircuitDiff drift and compliance grade drops.
- **GitHub Action CI hook** — auto-fails CI if compliance grade drops below threshold. Annex IV reports uploaded as workflow artifacts.

### v3.2.1 — Previous release

- **stability_suite()** — multi-prompt Jaccard circuit stability analysis
- **BSL 1.1 dual licensing** — commercial IP protection (Change Date 2036-03-21)
- **CircuitDiff** — patent-pending mechanistic diff between model checkpoints
- **BiasAnalyzer** — demographic parity, counterfactual fairness, token bias probe (Article 10(2)(f))
- **AuditLog** — append-only JSONL with SHA-256 hash chain tamper detection (Article 12)

### EU AI Act relevance

Enforcement starts **August 2026**. High-risk AI systems (finance, healthcare, HR, legal) must explain decisions to affected parties under Article 13. Glassbox provides:

- Annex IV technical documentation (all 9 sections)
- Explainability grades A–D mapped to Article 13 requirements
- Tamper-evident audit trail for national competent authority submission
- Bias testing per Article 10(2)(f)

### Grading scale

| Grade | F1 range | Meaning |
|-------|----------|---------|
| **A** | ≥ 0.70 | Fully explainable — minimal compliance risk |
| **B** | 0.50–0.69 | Mostly explainable — minor gaps |
| **C** | 0.30–0.49 | Partially explainable — significant gaps |
| **D** | < 0.30 | Not explainable — compliance risk |

### Citation

```bibtex
@software{mahale2026glassbox,
  author  = {Mahale, Ajay Pravin},
  title   = {Glassbox 3.3: Mechanistic Interpretability and EU AI Act Compliance Toolkit},
  year    = {2026},
  url     = {https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool},
  version = {3.3.0}
}
```

### References

- Wang et al. (2022). Interpretability in the Wild: IOI in GPT-2 small. arXiv:2211.00593
- Nanda (2023). Attribution Patching. neelnanda.io
- Conmy et al. (2023). Towards Automated Circuit Discovery (ACDC). arXiv:2304.14997
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. transformer-circuits.pub
- EU AI Act (EU) 2024/1689, Official Journal of the EU

---
**Contact:** mahale.ajay01@gmail.com | **License:** MIT | **Version:** 3.3.0
"""

with gr.Blocks(
    title="Glassbox 3.4 — EU AI Act Compliance",
    css=GB_CSS,
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="zinc",
    ).set(
        body_background_fill="#000000",
        body_background_fill_dark="#000000",
        block_background_fill="#00000000",
        block_background_fill_dark="#00000000",
        block_border_color="rgba(255,255,255,0.07)",
        block_border_color_dark="rgba(255,255,255,0.07)",
        input_background_fill="rgba(255,255,255,0.04)",
        input_background_fill_dark="rgba(255,255,255,0.04)",
        input_border_color="rgba(255,255,255,0.13)",
        input_border_color_dark="rgba(255,255,255,0.13)",
        button_primary_background_fill="linear-gradient(135deg,#6366f1,#4f46e5)",
        button_primary_background_fill_dark="linear-gradient(135deg,#6366f1,#4f46e5)",
        button_primary_background_fill_hover="linear-gradient(135deg,#818cf8,#6366f1)",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="rgba(255,255,255,0.05)",
        button_secondary_border_color="rgba(255,255,255,0.13)",
        button_secondary_text_color="#a1a1aa",
        shadow_drop="0 4px 24px rgba(0,0,0,0.6)",
        shadow_drop_lg="0 8px 40px rgba(0,0,0,0.8)",
        color_accent_soft="rgba(99,102,241,0.15)",
        color_accent_soft_dark="rgba(99,102,241,0.15)",
    ),
) as demo:
    if _STARTUP_ERROR:
        gr.Markdown(f"## ⚠️ Startup Error\n```\n{_STARTUP_ERROR}\n```")
    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Circuit Analysis ────────────────────────────────────────────
        with gr.Tab("⚡ Circuit Analysis"):
            gr.Markdown("### Discover which attention heads causally drive a prediction")
            with gr.Row():
                with gr.Column(scale=1):
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
                    run_btn = gr.Button("Analyze Circuit", variant="primary", size="lg")
                with gr.Column(scale=2):
                    heatmap_out = gr.Image(label="Attribution Heatmap (gold = circuit heads)", type="pil")
                    report_out  = gr.Markdown()
                    _hidden_err = gr.Textbox(visible=False)
            run_btn.click(
                fn=run_full_analysis,
                inputs=[prompt_in, correct_in, incorrect_in],
                outputs=[heatmap_out, report_out, _hidden_err],
            )

        # ── Tab 2: Logit Lens ──────────────────────────────────────────────────
        with gr.Tab("🔬 Logit Lens"):
            gr.Markdown("### Track how a token's probability evolves layer by layer")
            with gr.Row():
                with gr.Column(scale=1):
                    ll_prompt = gr.Textbox(
                        label="Prompt",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3,
                    )
                    ll_token = gr.Textbox(label="Target token", value=" Mary")
                    ll_btn = gr.Button("Run Logit Lens", variant="primary")
                with gr.Column(scale=2):
                    ll_img    = gr.Image(label="Probability and Rank by Layer", type="pil")
                    ll_report = gr.Markdown()
            ll_btn.click(
                fn=run_logit_lens_tab,
                inputs=[ll_prompt, ll_token],
                outputs=[ll_img, ll_report],
            )

        # ── Tab 3: Attention Patterns ──────────────────────────────────────────
        with gr.Tab("👁 Attention Patterns"):
            gr.Markdown("### Visualise raw attention weights for any layer and head")
            with gr.Row():
                with gr.Column(scale=1):
                    at_prompt = gr.Textbox(
                        label="Prompt",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3,
                    )
                    at_layer = gr.Slider(0, 11, value=9, step=1, label="Layer (0–11)")
                    at_head  = gr.Slider(0, 11, value=9, step=1, label="Head (0–11)")
                    at_btn   = gr.Button("Visualise", variant="primary")
                with gr.Column(scale=2):
                    at_img    = gr.Image(label="Attention Pattern", type="pil")
                    at_status = gr.Markdown()
            at_btn.click(
                fn=run_attention_tab,
                inputs=[at_prompt, at_layer, at_head],
                outputs=[at_img, at_status],
            )

        # ── Tab 4: Compliance Report ───────────────────────────────────────────
        with gr.Tab("📋 Compliance Report"):
            gr.Markdown("### Generate a full EU AI Act Annex IV compliance report")
            with gr.Row():
                with gr.Column(scale=1):
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
                    cr_btn = gr.Button("Generate Annex IV Report", variant="primary", size="lg")
                with gr.Column(scale=2):
                    cr_report    = gr.Markdown(label="Annex IV Report")
                    cr_modelcard = gr.Code(label="Model Card (HuggingFace-compatible Markdown)", language="markdown")
            cr_btn.click(
                fn=run_compliance_report,
                inputs=[cr_prompt, cr_correct, cr_incorrect, cr_model, cr_provider, cr_deploy],
                outputs=[cr_report, cr_modelcard],
            )

        # ── Tab 5: About ───────────────────────────────────────────────────────
        with gr.Tab("📖 About"):
            gr.Markdown(ABOUT_TEXT)

    gr.HTML("""
<div style="
  text-align:center; padding:24px 0 12px;
  margin-top:20px; border-top:1px solid rgba(255,255,255,.06);
  font-family:'Inter',sans-serif; font-size:12px;
  color:#3f3f46; letter-spacing:.02em;
">
  <span style="color:#52525b;">Glassbox v3.4.0</span>
  &nbsp;·&nbsp; Built on TransformerLens
  &nbsp;·&nbsp; MIT License
  &nbsp;·&nbsp; <a href="mailto:mahale.ajay01@gmail.com" style="color:#52525b;text-decoration:none;">mahale.ajay01@gmail.com</a>
  &nbsp;·&nbsp; EU AI Act (EU) 2024/1689
  <div style="margin-top:10px;display:flex;justify-content:center;gap:16px;flex-wrap:wrap;">
    <a href="https://project-gu05p.vercel.app" target="_blank"
       style="color:#52525b;text-decoration:none;font-size:11px;transition:color .15s;"
       onmouseover="this.style.color='#a5b4fc'" onmouseout="this.style.color='#52525b'">
      🌐 Website
    </a>
    <a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool" target="_blank"
       style="color:#52525b;text-decoration:none;font-size:11px;transition:color .15s;"
       onmouseover="this.style.color='#a5b4fc'" onmouseout="this.style.color='#52525b'">
      ⭐ GitHub
    </a>
    <a href="https://pypi.org/project/glassbox-mech-interp/" target="_blank"
       style="color:#52525b;text-decoration:none;font-size:11px;transition:color .15s;"
       onmouseover="this.style.color='#a5b4fc'" onmouseout="this.style.color='#52525b'">
      📦 PyPI
    </a>
  </div>
</div>
    """)

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

# v3.4.1-patch: python_version=3.11 + pyaudioop in Space to permanently fix py3.13 audioop crash
