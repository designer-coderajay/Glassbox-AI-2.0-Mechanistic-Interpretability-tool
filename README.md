<div align="center">

# Glassbox 2.0
**Open-source mechanistic interpretability for transformer models.**

[![PyPI](https://img.shields.io/pypi/v/glassbox-mech-interp?color=blue&label=PyPI)](https://pypi.org/project/glassbox-mech-interp/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/glassbox-mech-interp?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GRAY&left_text=downloads)](https://pepy.tech/projects/glassbox-mech-interp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)
[![arXiv](https://img.shields.io/badge/arXiv-2603.09988-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.09988)

[**Live Demo**](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) · [**Docs**](https://designer-coderajay.github.io/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/) · [**PyPI**](https://pypi.org/project/glassbox-mech-interp/)

</div>

---

Glassbox 2.0 identifies the attention heads responsible for a model's prediction, quantifies their causal contribution, and tells you exactly why a transformer made the choice it did — in one function call.

Built on attribution patching with O(3) complexity. Benchmarked against ACDC. Grounded in peer-reviewed mechanistic interpretability research.

---

## Highlights

- **O(3) attribution patching** — identifies circuits in a single forward-backward pass, not exhaustive edge enumeration
- **37x faster than ACDC** on GPT-2 small (1.2s vs 43.2s)
- **Bootstrap 95% CI** — every faithfulness score from `bootstrap_metrics()` ships with confidence intervals, not point estimates
- **FCAS cross-model alignment** — quantifies how similar circuits are across model sizes (GPT-2 family: 0.783-0.835)
- **Interactive dashboard** — Streamlit UI on HuggingFace Spaces, no setup required

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
    correct   = "Mary",
    incorrect = "John",
)

print(result["faithfulness"])
# {
#   "sufficiency":       0.80,   <- Taylor approximation (see note below)
#   "comprehensiveness": 0.37,   <- exact causal value
#   "f1":                0.49,
#   "category":          "moderate",
#   "suff_is_approx":    True
# }

# Circuit is a list of (layer, head) tuples, sorted by attribution score
print(result["circuit"])
# [(9, 9), (8, 10), (7, 3), ...]

# To see attribution scores for each head:
attrs = result["attributions"]
for (layer, head) in result["circuit"]:
    score = attrs.get(str((layer, head)), 0.0)
    print(f"L{layer:02d}H{head:02d} -> {score:.4f}")

# For confidence intervals, use bootstrap_metrics() instead:
boot = gb.bootstrap_metrics(prompts=[
    ("When Mary and John went to the store, John gave a drink to", "Mary", "John"),
    ("When Alice and Bob entered the room, Bob handed the key to", "Alice", "Bob"),
    # ... add more prompts for reliable CIs (recommended n >= 20)
], n_boot=500)
print(boot["sufficiency"])
# {"mean": 0.82, "std": 0.06, "ci_lo": 0.71, "ci_hi": 0.91, "n": 2}
```

> **Note on Sufficiency:** The `sufficiency` value in `analyze()` is a first-order Taylor
> approximation (Nanda et al. 2023), not the exact causal value. This is why the benchmark
> table below shows ~80% while the MSc thesis paper reports ~100% — the paper used exact
> Wang et al. (2022) sufficiency computed by ablating non-circuit heads. Both are valid
> measures; they differ by methodology. The `suff_is_approx: True` flag in the output
> makes this explicit.

Try it instantly — no install needed: **[huggingface.co/spaces/designer-coderajay/Glassbox-ai](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)**

---

## Benchmarks

Evaluated on the IOI (Indirect Object Identification) task across the GPT-2 family.

| Model        | Layers | Heads | Sufficiency* | Comprehensiveness | F1     | Glassbox | ACDC    | Speedup |
|-------------|--------|-------|-------------|-------------------|--------|----------|---------|---------|
| GPT-2 small  | 12     | 12    | 80.0%       | 37.2%             | 48.8%  | 1.2s     | 43.2s   | **37x** |
| GPT-2 medium | 24     | 16    | 35.1%       | 23.7%             | 27.9%  | 4.9s     | 115.2s  | **24x** |
| GPT-2 large  | 36     | 20    | 18.2%       | 14.2%             | 15.9%  | 14.3s    | 216.0s  | **15x** |

*Sufficiency values are first-order Taylor approximations. Exact causal sufficiency (requiring
full ablation runs) is higher — see the [arXiv paper](https://arxiv.org/abs/2603.09988) for exact values.

**Cross-model circuit alignment (FCAS):**

| Pair                        | FCAS  |
|----------------------------|-------|
| GPT-2 small <-> GPT-2 medium  | 0.835 |
| GPT-2 small <-> GPT-2 large   | 0.783 |
| GPT-2 medium <-> GPT-2 large  | 0.833 |

High FCAS scores confirm the IOI circuit is structurally conserved across model scale — consistent with Wang et al. (2022).

---

## How It Works

Glassbox runs attribution patching with name-swap corruption, matching the methodology of Wang et al. (2022).

```
Clean prompt     ->  model  ->  logit(Mary)
Corrupted prompt ->  model  ->  logit(John)

For each attention head:
  Patch clean activation -> corrupted run
  Measure delta_logit(Mary - John)
  Normalize -> attribution score
```

**Faithfulness metrics** follow the ERASER framework:

- **Sufficiency** — does the circuit alone recover the clean prediction? (Taylor approx in `analyze()`, exact in paper)
- **Comprehensiveness** — how much does ablating the circuit hurt? (exact causal measurement)
- **F1** — harmonic mean of both

Confidence intervals are available via `bootstrap_metrics()` with n_boot resamples.

---

## Installation

```bash
# From PyPI
pip install glassbox-mech-interp

# From source
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
pip install -e .
```

**Requirements:** Python >= 3.8, PyTorch >= 2.0, TransformerLens >= 1.0

---

## Run the Dashboard Locally

```bash
pip install glassbox-mech-interp streamlit plotly
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
streamlit run dashboard/app.py
```

Or use the hosted version at [huggingface.co/spaces/designer-coderajay/Glassbox-ai](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai).

---

## API Reference

### `GlassboxV2(model)`

```python
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

model = HookedTransformer.from_pretrained("gpt2")
gb    = GlassboxV2(model)
```

| Method | Description |
|--------|-------------|
| `gb.analyze(prompt, correct, incorrect)` | Full circuit analysis. Returns dict with `circuit`, `attributions`, `faithfulness`, `corr_prompt`. |
| `gb.attribution_patching(clean_tokens, corr_tokens, target_id, distractor_id)` | Raw per-head attribution scores. Expects tokenized tensors and integer token IDs. |
| `gb.bootstrap_metrics(prompts, n_boot, alpha)` | Bootstrap 95% CI on Suff/Comp/F1. Pass list of (prompt, correct, incorrect) tuples. |
| `gb.functional_circuit_alignment(heads_a, heads_b, top_k, n_null)` | Cross-model FCAS score with null distribution and z-score. |

**`analyze()` return structure:**

```python
{
    "circuit":      [(9, 9), (8, 10), ...],       # List of (layer, head) tuples
    "n_heads":      int,
    "clean_ld":     float,                         # logit(correct) - logit(incorrect)
    "corr_prompt":  str,                           # name-swapped corrupted prompt
    "attributions": {"(9, 9)": 0.174, ...},        # string keys, float values
    "faithfulness": {
        "sufficiency":       float,                # Taylor approximation
        "comprehensiveness": float,                # exact causal value
        "f1":                float,
        "category":          str,                  # one of: faithful, backup_mechanisms,
                                                   #   moderate, incomplete, weak
        "suff_is_approx":    True,
    }
}
```

---

## Citation

If you use Glassbox 2.0 in your research, please cite:

```bibtex
@software{mahale2025glassbox,
  author    = {Mahale, Ajay Pravin},
  title     = {Glassbox 2.0: Causally Grounded Mechanistic Interpretability for Transformer Models},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool}
}
```

---

## Related Work

- [Wang et al. (2022)](https://arxiv.org/abs/2211.00593) — IOI circuit discovery in GPT-2
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — mechanistic interpretability library this builds on
- [ACDC](https://github.com/ArthurConmy/Automatic-Circuit-DisCovery) — automatic circuit discovery (baseline we benchmark against)
- [ERASER](https://eraser-benchmark.github.io/) — faithfulness evaluation framework

---

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">
Built by <a href="mailto:mahale.ajay01@gmail.com">Ajay Pravin Mahale</a> · Made in Germany · Glassbox AI
</div>
