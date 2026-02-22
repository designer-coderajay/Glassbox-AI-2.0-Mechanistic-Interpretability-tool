README = '''<div align="center">

# 🔬 Glassbox 2.0

**Open-source mechanistic interpretability for transformer models.**

[![PyPI](https://img.shields.io/pypi/v/glassbox-mech-interp?color=blue&label=PyPI)](https://pypi.org/project/glassbox-mech-interp/)
[![Tests](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)

[**Live Demo**](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) · [**Paper**](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) · [**Docs**](https://designer-coderajay.github.io/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/) · [**PyPI**](https://pypi.org/project/glassbox-mech-interp/)

</div>

---

Glassbox 2.0 identifies the attention heads responsible for a model\'s prediction, quantifies their causal contribution, and tells you exactly why a transformer made the choice it did — in one function call.

Built on attribution patching with O(3) complexity. Benchmarked against ACDC. Grounded in peer-reviewed mechanistic interpretability research.

---

## Highlights

- **O(3) attribution patching** — identifies circuits in a single forward-backward pass, not exhaustive edge enumeration
- **37× faster than ACDC** on GPT-2 small (1.2s vs 43.2s)
- **Bootstrap 95% CI** — every faithfulness score ships with confidence intervals, not point estimates
- **FCAS cross-model alignment** — quantifies how similar circuits are across model sizes (GPT-2 family: 0.783–0.835)
- **12/12 tests green** — full pytest suite, runs in CI on every push
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
    prompt     = "When Mary and John went to the store, John gave a drink to",
    correct    = "Mary",
    incorrect  = "John",
)

print(result["faithfulness"])
# {
#   "sufficiency":       0.80,
#   "comprehensiveness": 0.37,
#   "f1":                0.49,
#   "category":          "good",
#   "ci_95":             [0.71, 0.89]
# }

print(result["circuit"]["top_heads"])
# [{"layer": 9, "head": 9, "score": 0.174, "role": "name-mover"}, ...]
```

Try it instantly — no install needed: **[huggingface.co/spaces/designer-coderajay/Glassbox-ai](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)**

---

## Benchmarks

Evaluated on the IOI (Indirect Object Identification) task across the GPT-2 family.

| Model        | Layers | Heads | Sufficiency | Comprehensiveness | F1     | Glassbox | ACDC    | Speedup |
|-------------|--------|-------|-------------|-------------------|--------|----------|---------|---------|
| GPT-2 small  | 12     | 12    | 80.0%       | 37.2%             | 48.8%  | 1.2s     | 43.2s   | **37×** |
| GPT-2 medium | 24     | 16    | 35.1%       | 23.7%             | 27.9%  | 4.9s     | 115.2s  | **24×** |
| GPT-2 large  | 36     | 20    | 18.2%       | 14.2%             | 15.9%  | 14.3s    | 216.0s  | **15×** |

**Cross-model circuit alignment (FCAS):**

| Pair                        | FCAS  |
|----------------------------|-------|
| GPT-2 small ↔ GPT-2 medium  | 0.835 |
| GPT-2 small ↔ GPT-2 large   | 0.783 |
| GPT-2 medium ↔ GPT-2 large  | 0.833 |

High FCAS scores confirm the IOI circuit is structurally conserved across model scale — consistent with Wang et al. (2022).

---

## How It Works

Glassbox runs attribution patching with name-swap corruption, matching the methodology of Wang et al. (2022).
```
Clean prompt     →  model  →  logit(Mary)
Corrupted prompt →  model  →  logit(John)

For each attention head:
  Patch clean activation → corrupted run
  Measure Δlogit(Mary - John)
  Normalize → attribution score
```

**Faithfulness metrics** follow the ERASER framework:
- **Sufficiency** — does the circuit alone recover the clean prediction?
- **Comprehensiveness** — how much does ablating the circuit hurt?
- **F1** — harmonic mean of both

All scores ship with **bootstrap 95% confidence intervals** (n=1000 resamples).

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

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, TransformerLens ≥ 1.0

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
gb = GlassboxV2(model)
```

| Method | Description |
|--------|-------------|
| `gb.analyze(prompt, correct, incorrect)` | Full circuit analysis. Returns faithfulness scores + circuit heads. |
| `gb.attribution_patching(clean_tok, corr_tok, target_id, distractor_id)` | Raw attribution scores as `(n_layers, n_heads)` tensor. |

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

