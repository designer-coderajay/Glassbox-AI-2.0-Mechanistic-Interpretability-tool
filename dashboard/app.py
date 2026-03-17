"""
Glassbox — Causal Mechanistic Interpretability
==============================================
HuggingFace Space — product-level UI v2.6.0

Uses the published glassbox-mech-interp v2.6.0 API:
  from glassbox import GlassboxV2
  model = HookedTransformer.from_pretrained("gpt2")
  gb    = GlassboxV2(model)
  result = gb.analyze(prompt, correct, incorrect)
"""

import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from PIL import Image

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading GPT-2 small via TransformerLens ...")
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
gb = GlassboxV2(model)
print("Model ready  (12 layers x 12 heads, 117M params)")

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Attribution Score", fraction=0.03, pad=0.04)

    # Highlight circuit heads with a white rectangle
    for (l, h) in circuit:
        rect = mpatches.FancyBboxPatch(
            (h - 0.45, l - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor="white", facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        "Attribution Patching — Causal Head Importance\n(white boxes = discovered circuit)",
        fontsize=13
    )
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    fig.tight_layout()
    return _fig_to_pil(fig)


def _logit_lens_plot(prompt: str, target_token: str) -> Image.Image:
    """Project residual stream through unembed at each layer."""
    tokens = model.to_tokens(prompt)
    try:
        t_idx = model.to_single_token(target_token)
    except Exception:
        t_idx = model.to_tokens(target_token)[0, -1].item()

    layer_logprobs = []
    layer_ranks    = []

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        for l in range(model.cfg.n_layers):
            resid  = cache[f"blocks.{l}.hook_resid_post"][0, -1]
            normed = model.ln_final(resid.unsqueeze(0).unsqueeze(0))[0, 0]
            logits = model.unembed(normed.unsqueeze(0).unsqueeze(0))[0, 0]
            log_probs = torch.log_softmax(logits, dim=-1)
            layer_logprobs.append(log_probs[t_idx].item())
            rank = (logits > logits[t_idx]).sum().item() + 1
            layer_ranks.append(rank)

    probs  = [np.exp(lp) * 100 for lp in layer_logprobs]
    layers = list(range(model.cfg.n_layers))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(layers, probs, "o-", lw=2, ms=7, color="#2E86AB")
    ax1.fill_between(layers, probs, alpha=0.15, color="#2E86AB")
    ax1.set_ylabel("Probability (%)", fontsize=11)
    ax1.set_title(f"Logit Lens -- token: '{target_token}'", fontsize=13)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(bottom=0)

    ax2.plot(layers, layer_ranks, "s-", lw=2, ms=7, color="#A23B72")
    ax2.set_ylabel("Rank (lower = better)", fontsize=11)
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25)
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

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(7, n * 0.6)))
    im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Attention", fraction=0.03, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_xticklabels(token_strs, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(token_strs, fontsize=9)
    ax.set_xlabel("Key (attends to)", fontsize=11)
    ax.set_ylabel("Query (from)", fontsize=11)
    ax.set_title(f"Attention Pattern -- Layer {layer}, Head {head}", fontsize=13)
    fig.tight_layout()
    return _fig_to_pil(fig)


# ── Core analysis functions ────────────────────────────────────────────────────

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

    img = _attribution_heatmap(attrs, circuit)

    cat_label = {
        "faithful":          "Faithful",
        "backup_mechanisms": "Backup Mechanisms Present",
        "incomplete":        "Incomplete Circuit",
        "weak":              "Weak Signal",
        "moderate":          "Moderate",
    }.get(faith["category"], faith["category"])

    top_heads = "\n".join(
        f"  - Layer {l}, Head {h}  (attr = {attrs.get(str((l,h)), 0):.3f})"
        for l, h in circuit[:6]
    ) or "  *(no circuit heads found)*"

    suff_note = " *(first-order approximation)*" if faith["suff_is_approx"] else ""

    report = f"""## Circuit Analysis Report

**Prompt:** *{prompt.strip()}*
**Correct:** `{correct.strip()}` | **Distractor:** `{incorrect.strip()}`
**Corrupted prompt:** *{result["corr_prompt"]}*

---

### Circuit Heads ({len(circuit)} found)
{top_heads}

---

### Faithfulness Metrics

| Metric | Score | Method |
|--------|-------|--------|
| Sufficiency{suff_note} | {faith["sufficiency"]:.1%} | Taylor approx (3 passes) |
| Comprehensiveness | {faith["comprehensiveness"]:.1%} | Exact activation patching |
| F1 | {faith["f1"]:.1%} | Harmonic mean |
| Clean Logit Diff | {ld:.3f} | logit(correct) - logit(distractor) |
| Category | **{cat_label}** | |

---

### Interpretation

- **Sufficiency {faith["sufficiency"]:.0%}**: The identified circuit accounts for {faith["sufficiency"]:.0%} of the model's prediction for this token.
- **Comprehensiveness {faith["comprehensiveness"]:.0%}**: Ablating this circuit reduces the prediction by {faith["comprehensiveness"]:.0%}.
{"- **Backup mechanisms**: High sufficiency + lower comprehensiveness = the model has redundant pathways compensating when this circuit is removed. This is common in Name Mover heads." if faith["category"] == "backup_mechanisms" else ""}
{"- **Incomplete**: The identified heads do not fully explain the prediction. The model may distribute this computation across many small contributions." if faith["category"] == "incomplete" else ""}

---

### EU AI Act Relevance

This report maps to **Article 13 transparency requirements**. The circuit identifies which model components (by layer and head index) causally drove this prediction, with quantified causal faithfulness scores suitable for audit documentation.

---
*Glassbox v2.6.0 | pip install glassbox-mech-interp*
"""
    return img, report, ""


def run_logit_lens_tab(prompt: str, target_token: str):
    if not prompt.strip() or not target_token.strip():
        return None, "Please fill in both fields."
    try:
        img = _logit_lens_plot(prompt.strip(), target_token.strip())
        tokens = model.to_tokens(prompt.strip())
        t_idx  = model.to_single_token(target_token.strip())
        with torch.no_grad():
            logits = model(tokens)[0, -1]
        final_rank = (logits > logits[t_idx]).sum().item() + 1
        final_prob = torch.softmax(logits, dim=-1)[t_idx].item() * 100
        summary = f"**Final layer:** token `{target_token.strip()}` is rank **{final_rank}** with probability **{final_prob:.2f}%**"
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


# ── Gradio UI ─────────────────────────────────────────────────────────────────

HEADER = """
<div style="text-align:center; padding: 20px 0 10px 0;">
  <h1 style="font-size:2.4em; font-weight:800; color:#1a1a2e; margin:0; letter-spacing:-1px;">
    Glassbox
  </h1>
  <p style="font-size:1.1em; color:#555; margin:8px 0 4px 0; font-weight:400;">
    Causal Mechanistic Interpretability. See exactly why your LLM made that prediction.
  </p>
  <p style="font-size:0.88em; color:#999; margin:4px 0;">
    Circuit discovery in ~1.2s on CPU &nbsp;&middot;&nbsp; GPT-2 small (117M params) &nbsp;&middot;&nbsp;
    <a href="https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool"
       target="_blank" style="color:#4a90d9;">GitHub</a>
    &nbsp;&middot;&nbsp;
    <a href="https://pypi.org/project/glassbox-mech-interp/"
       target="_blank" style="color:#4a90d9;">pip install glassbox-mech-interp</a>
  </p>
</div>
"""

ABOUT_TEXT = """
### What is Glassbox?

Glassbox identifies the **specific attention heads** in a transformer that *causally* drive a
prediction. Not just which tokens the model attended to, but which internal components are
responsible and how much.

**Three metrics per analysis:**

| Metric | What it measures | Method |
|--------|-----------------|--------|
| **Sufficiency** | How much of the prediction do the identified heads explain? | Taylor approximation (3 passes) |
| **Comprehensiveness** | How much does the prediction degrade when those heads are ablated? | Exact activation patching (2 passes) |
| **F1** | Single faithfulness score | Harmonic mean of above |

**Why this matters beyond research:**

The EU AI Act Article 13 requires high-risk AI systems (finance, healthcare, HR, legal) to explain
their decisions to affected parties. Enforcement starts **August 2026**. Glassbox's structured output
provides causal circuit documentation suitable for AI audits.

**Algorithm overview:**

1. Attribution patching (Nanda, 2023) scores all attention heads in 3 forward passes.
2. Greedy forward selection builds a candidate circuit until 85% Taylor-sufficiency is reached.
3. Backward pruning removes causally redundant heads using exact comprehensiveness.
4. Total: O(3 + 2p) passes where p = pruning steps (typically 2-4).

**Limitations disclosed:**

- Sufficiency is a first-order Taylor approximation, not exact. Flagged in every output with `suff_is_approx: True`.
- Head-level granularity only -- not edge-level. ACDC provides higher granularity at 37x the compute cost.
- Works on TransformerLens-supported models only. Llama, Mistral, and other architectures require TransformerLens wrapping.

### Citation

```bibtex
@software{mahale2026glassbox,
  author  = {Mahale, Ajay Pravin},
  title   = {Glassbox: A Causal Mechanistic Interpretability Toolkit},
  year    = {2026},
  url     = {https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool},
  version = {2.6.0}
}
```

### References

- Wang et al. (2022). Interpretability in the Wild: A Circuit for IOI in GPT-2 small. arXiv:2211.00593
- Nanda (2023). Attribution Patching. neelnanda.io
- Conmy et al. (2023). Towards Automated Circuit Discovery (ACDC). arXiv:2304.14997
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. transformer-circuits.pub

---
**Contact:** mahale.ajay01@gmail.com &nbsp;|&nbsp; **License:** MIT
"""

with gr.Blocks(title="Glassbox -- Mechanistic Interpretability") as demo:
    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Circuit Analysis ─────────────────────────────────────────
        with gr.Tab("Circuit Analysis"):
            gr.Markdown("### Discover which attention heads causally drive a prediction")

            with gr.Row():
                with gr.Column(scale=1):
                    prompt_in    = gr.Textbox(
                        label="Prompt",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3,
                        placeholder="Enter any text prompt..."
                    )
                    correct_in   = gr.Textbox(
                        label="Correct next token (include leading space if needed)",
                        value=" Mary",
                        placeholder="e.g.  Mary"
                    )
                    incorrect_in = gr.Textbox(
                        label="Distractor token",
                        value=" John",
                        placeholder="e.g.  John"
                    )

                    with gr.Accordion("Example prompts", open=False):
                        gr.Markdown("""
**Indirect Object Identification (Wang et al. 2022):**
- Prompt: `When Mary and John went to the store, John gave a drink to`
- Correct: ` Mary` | Distractor: ` John`

**Factual Recall:**
- Prompt: `The capital of France is`
- Correct: ` Paris` | Distractor: ` London`

**Subject-Verb Agreement:**
- Prompt: `The keys to the cabinet`
- Correct: ` are` | Distractor: ` is`

**Greater-than:**
- Prompt: `The year 1956 came after`
- Correct: ` 1955` | Distractor: ` 1957`
                        """)

                    run_btn = gr.Button("Analyze Circuit", variant="primary", size="lg")

                with gr.Column(scale=2):
                    heatmap_out = gr.Image(
                        label="Attribution Heatmap (white boxes = circuit heads)",
                        type="pil"
                    )
                    report_out  = gr.Markdown()
                    _hidden_err = gr.Textbox(visible=False)

            run_btn.click(
                fn=run_full_analysis,
                inputs=[prompt_in, correct_in, incorrect_in],
                outputs=[heatmap_out, report_out, _hidden_err],
            )

        # ── Tab 2: Logit Lens ───────────────────────────────────────────────
        with gr.Tab("Logit Lens"):
            gr.Markdown("### Track how a token's probability evolves through the layers")

            with gr.Row():
                with gr.Column(scale=1):
                    ll_prompt = gr.Textbox(
                        label="Prompt",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3
                    )
                    ll_token  = gr.Textbox(
                        label="Target token",
                        value=" Mary",
                        placeholder="e.g.  Mary"
                    )
                    ll_btn = gr.Button("Run Logit Lens", variant="primary")

                with gr.Column(scale=2):
                    ll_img    = gr.Image(label="Probability and Rank by Layer", type="pil")
                    ll_report = gr.Markdown()

            ll_btn.click(
                fn=run_logit_lens_tab,
                inputs=[ll_prompt, ll_token],
                outputs=[ll_img, ll_report],
            )

        # ── Tab 3: Attention Patterns ───────────────────────────────────────
        with gr.Tab("Attention Patterns"):
            gr.Markdown("### Visualise raw attention weights for any layer and head")

            with gr.Row():
                with gr.Column(scale=1):
                    at_prompt = gr.Textbox(
                        label="Prompt",
                        value="When Mary and John went to the store, John gave a drink to",
                        lines=3
                    )
                    at_layer  = gr.Slider(0, 11, value=9, step=1, label="Layer (0-11)")
                    at_head   = gr.Slider(0, 11, value=9, step=1, label="Head (0-11)")
                    at_btn    = gr.Button("Visualise", variant="primary")

                with gr.Column(scale=2):
                    at_img    = gr.Image(label="Attention Pattern", type="pil")
                    at_status = gr.Markdown()

            at_btn.click(
                fn=run_attention_tab,
                inputs=[at_prompt, at_layer, at_head],
                outputs=[at_img, at_status],
            )

        # ── Tab 4: About ────────────────────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(ABOUT_TEXT)

    gr.Markdown("""
---
<div style="text-align:center; color:#aaa; font-size:0.82em; padding:8px 0;">
Glassbox v2.6.0 &nbsp;|&nbsp; Built on TransformerLens &nbsp;|&nbsp;
MIT License &nbsp;|&nbsp; mahale.ajay01@gmail.com
</div>
    """)

demo.launch()
