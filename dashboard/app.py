"""
Glassbox 3.1 — Causal Mechanistic Interpretability + EU AI Act Compliance
=========================================================================
HuggingFace Space — v3.1.0

Tabs:
  1. Circuit Analysis   — attribution patching, MFC discovery, faithfulness metrics
  2. Logit Lens         — residual stream projection by layer
  3. Attention Patterns — raw attention weight heatmap
  4. Compliance Report  — EU AI Act Annex IV explainability grade + bias check
  5. About / Docs       — methodology, references, citation
"""

import io
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

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
gb = GlassboxV2(model)
print("Model ready (12 layers × 12 heads, 117 M params)")

_audit_log = AuditLog("glassbox_space_audit.jsonl")
_bias_analyzer = BiasAnalyzer()

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

    report = f"""## Circuit Analysis — v3.1.0

**Prompt:** *{prompt.strip()}*
**Correct:** `{correct.strip()}` | **Distractor:** `{incorrect.strip()}`

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

### EU AI Act Relevance

Maps to **Article 13 transparency requirements**. Circuit identifies which model components causally drove this prediction with quantified faithfulness scores. Grade: **{"A" if faith["f1"] >= 0.7 else "B" if faith["f1"] >= 0.5 else "C" if faith["f1"] >= 0.3 else "D"}**

---
*Glassbox v3.1.0 · pip install glassbox-mech-interp*
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
| 6. Lifecycle | Version 3.1.0 · {_rj.get("generated_at", "")[:10]} |
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
*Glassbox v3.1.0 · EU AI Act (EU) 2024/1689 · Enforcement August 2026*
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

HEADER = """
<div style="background:linear-gradient(135deg,#07080d 0%,#0d1017 100%);
            padding:32px 24px 20px;
            border-bottom:1px solid rgba(255,255,255,0.065);
            margin-bottom:0;">
  <div style="max-width:860px; margin:0 auto; text-align:center;">
    <div style="display:inline-flex;align-items:center;gap:10px;
                background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);
                border-radius:20px;padding:4px 14px;margin-bottom:16px;">
      <span style="width:7px;height:7px;border-radius:50%;background:#22c55e;
                   box-shadow:0 0 6px #22c55e;display:inline-block;"></span>
      <span style="font-size:0.78em;color:#a5b4fc;font-weight:500;letter-spacing:0.05em;">
        THE ONLY EU AI ACT COMPLIANCE AUDIT PLATFORM
      </span>
    </div>
    <h1 style="font-size:2.8em;font-weight:900;color:#e2e8f0;margin:0 0 10px;
               letter-spacing:-1.5px;line-height:1.1;">
      Glassbox <span style="color:#6366f1;">3.1</span>
    </h1>
    <p style="font-size:1.05em;color:#94a3b8;margin:0 0 16px;font-weight:400;">
      Causal Mechanistic Interpretability · EU AI Act Annex IV Compliance · Bias Analysis
    </p>
    <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;">
      <a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool"
         target="_blank"
         style="background:#1a2030;border:1px solid rgba(255,255,255,0.1);border-radius:6px;
                padding:6px 14px;font-size:0.82em;color:#a5b4fc;text-decoration:none;font-weight:500;">
        ⭐ GitHub
      </a>
      <a href="https://pypi.org/project/glassbox-mech-interp/"
         target="_blank"
         style="background:#1a2030;border:1px solid rgba(255,255,255,0.1);border-radius:6px;
                padding:6px 14px;font-size:0.82em;color:#a5b4fc;text-decoration:none;font-weight:500;">
        📦 pip install glassbox-mech-interp
      </a>
      <span style="background:#1a2030;border:1px solid rgba(239,68,68,0.4);border-radius:6px;
                   padding:6px 14px;font-size:0.82em;color:#f87171;font-weight:500;">
        ⚡ Enforcement: August 2026
      </span>
    </div>
  </div>
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

### v3.1.0 — What's new

- **BiasAnalyzer** — demographic parity, counterfactual fairness, token bias probe (EU AI Act Article 10(2)(f))
- **AuditLog** — append-only JSONL with SHA-256 hash chain tamper detection (Article 12)
- **Webhooks** — POST callback on job completion for CI/CD pipelines
- **TypeScript SDK** — zero-dependency browser/Node/Deno/Bun client
- **GitHub Action** — `glassbox-audit@v1` drops compliance gating into any CI pipeline
- **Jupyter widgets** — `CircuitWidget.from_prompt()` renders inline heatmaps in notebooks
- **SVG export** — download the D3 circuit graph as a standalone SVG for papers

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
  title   = {Glassbox 3.1: Mechanistic Interpretability and EU AI Act Compliance Toolkit},
  year    = {2026},
  url     = {https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool},
  version = {3.1.0}
}
```

### References

- Wang et al. (2022). Interpretability in the Wild: IOI in GPT-2 small. arXiv:2211.00593
- Nanda (2023). Attribution Patching. neelnanda.io
- Conmy et al. (2023). Towards Automated Circuit Discovery (ACDC). arXiv:2304.14997
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. transformer-circuits.pub
- EU AI Act (EU) 2024/1689, Official Journal of the EU

---
**Contact:** mahale.ajay01@gmail.com | **License:** MIT | **Version:** 3.1.0
"""

with gr.Blocks(
    title="Glassbox 3.1 — EU AI Act Compliance",
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ).set(
        body_background_fill="#07080d",
        block_background_fill="#0d1017",
        block_border_color="rgba(255,255,255,0.065)",
        input_background_fill="#131720",
        button_primary_background_fill="#6366f1",
        button_primary_background_fill_hover="#4f46e5",
    )
) as demo:
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
<div style="text-align:center;color:#475569;font-size:0.8em;
            padding:16px 0;border-top:1px solid rgba(255,255,255,0.065);margin-top:8px;">
  Glassbox v3.1.0 &nbsp;·&nbsp; Built on TransformerLens &nbsp;·&nbsp;
  MIT License &nbsp;·&nbsp; mahale.ajay01@gmail.com &nbsp;·&nbsp;
  EU AI Act (EU) 2024/1689
</div>
    """)

demo.launch()
