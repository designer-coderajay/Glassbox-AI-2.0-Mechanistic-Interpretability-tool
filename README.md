# Glassbox 2.0 - Mechanistic Interpretability

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

> Open-source transformer circuit analysis. Attribution patching in O(3) forward passes,
> automatic circuit discovery, cross-model alignment scoring, and bootstrap confidence intervals.

---

## Why Glassbox 2.0?

| Method | Passes Required | Comprehensiveness | Circuit Discovery | Cross-Model Alignment |
|---|---|---|---|---|
| Full Activation Patching | O(2N) ~192x | Causal | Manual | None |
| TransformerLens (mean ablation) | O(2N) | Approximate | Manual | None |
| Anthropic ACDC | O(2N) | Causal | Auto | None |
| **Glassbox 2.0** | **O(3)** | **Corrupted patching** | **MFC auto-discovery** | **FCAS = 0.929** |

**96x faster than full activation patching** on GPT-2 Small (144 heads = 288 passes vs. 3).

---

## Novel Contributions

1. **O(3) Attribution Patching** — `attr(h) = grad_z * (z_clean - z_corrupt)`. Three forward
   passes regardless of model size. Validated against Wang et al. (2022) IOI ground truth.

2. **Minimum Faithful Circuit (MFC)** — Greedy forward selection (add heads until sufficiency
   >= 85%) + greedy backward pruning (remove heads while comprehensiveness >= 15%).
   Automatically finds the *smallest* causally faithful head set.

3. **Functional Circuit Alignment Score (FCAS)** — `1 - mean(|rel_depth_A - rel_depth_B|)`
   over matched heads. GPT-2 Small vs GPT-2 Medium: **FCAS = 0.929**.
   Name-mover circuits concentrate at relative depth ~0.82 across both scales.

4. **Bootstrap 95% CI** — Percentile bootstrap (n=500) on Sufficiency / Comprehensiveness / F1
   across the full prompt distribution. No cherry-picking.

---

## Benchmark Results

| Task | Domain | Suff | Comp | F1 | Category |
|---|---|---|---|---|---|
| IOI (Indirect Object ID) | Logic | 100% | 34.5% +/-14.6% | 49.4% | Moderate |
| SVA (Subject-Verb Agreement) | Grammar | 33.7% +/-4.9% | 51.7% +/-8.6% | 40.7% | Distributed |
| GEO (Country to Capital) | Factual | 90.2% +/-13.9% | 90.0% +/-14.1% | 90.1% | Faithful |

Top IOI head: **L9H9** (+4.20) -- consistent with Wang et al. (2022).
Top GEO head: **L9H8** (+2.32).

---

## Install

    pip install git+https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool.git

## Quick Start

    from glassbox import GlassboxV2

    gb = GlassboxV2("gpt2")
    result = gb.analyze(
        prompt="When Mary and John went to the store, John gave a gift to",
        correct=" Mary",
        incorrect=" John"
    )
    print(result["faithfulness"])
    # {'suff': 1.0, 'comp': 0.345, 'f1': 0.494, 'category': 'moderate'}

    for (layer, head), score in sorted(result["circuit"].items(), key=lambda x: -x[1]):
        print(f"L{layer:02d}H{head:02d}  {score:.4f}")

## CLI

    glassbox analyze \
      --prompt "When Mary and John went to the store, John gave a gift to" \
      --correct " Mary" \
      --incorrect " John"

---

## How It Works

    Input prompt -> Pass 1 (clean activations, no grad)
                 -> Pass 2 (corrupted activations, no grad)
                 -> Pass 3 (gradient pass: patch clean z, backward on logit diff)

    attr(layer, head) = grad * (z_clean - z_corrupt)   # per head, last position

    MFC Discovery:
      Phase 1 (forward):  add heads by |attr| until suff >= 0.85
      Phase 2 (backward): prune heads while comp >= 0.15

    Comprehensiveness (corrupted activation patching, Wang et al. 2022):
      Replace circuit heads' clean activations with corrupted activations.
      comp = 1 - (patched_logit_diff / clean_logit_diff)

---

## Project Structure

    glassbox/
    |-- core.py          # GlassboxV2 engine
    |-- cli.py           # Command-line interface
    |-- __init__.py

    benchmarks/          # Standalone evaluation scripts (IOI / SVA / GEO)
    tests/               # Unit tests for patching hooks and circuit metrics
    dashboard/           # Streamlit web UI for visual circuit inspection

---

## Citation

    @software{mahale2026glassbox,
      author  = {Mahale, Ajay},
      title   = {Glassbox 2.0: O(3) Attribution Patching and Minimum Faithful Circuit Discovery},
      year    = {2026},
      url     = {https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool}
    }

---

## References

- Wang et al. (2022). Interpretability in the Wild: a Circuit for IOI in GPT-2 Small. ICLR 2023.
- Conmy et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS 2023.
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. Anthropic.
