<div align="center">

# Glassbox 2.0
**Causal mechanistic interpretability for transformer models. See exactly why your LLM made that prediction.**

[![PyPI](https://img.shields.io/pypi/v/glassbox-mech-interp?color=blue&label=PyPI)](https://pypi.org/project/glassbox-mech-interp/)
[![Downloads](https://static.pepy.tech/badge/glassbox-mech-interp)](https://pepy.tech/projects/glassbox-mech-interp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)
[![arXiv](https://img.shields.io/badge/arXiv-2603.09988-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.09988)
[![Tests](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml/badge.svg)](https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/actions/workflows/tests.yml)
[![LessWrong](https://img.shields.io/badge/LessWrong-Discussion-teal)]([https://lesswrong.com](https://www.lesswrong.com/posts/Bkp73EaixnXdL4tKa/glassbox-i-built-a-circuit-discovery-toolkit-that-s-37))

[**Live Demo**](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai) · [**Paper**](https://arxiv.org/abs/2603.09988) · [**Docs**](https://designer-coderajay.github.io/Glassbox-AI-2.0-Mechanistic-Interpretability-tool/) · [**PyPI**](https://pypi.org/project/glassbox-mech-interp/) · [**Discussion**](https://lesswrong.com)

</div>

---

Glassbox answers a single question: **what is this transformer actually doing?**

One function call identifies the circuit — the sparse subgraph of attention heads and MLP layers causally responsible for a prediction. Every score is grounded in peer-reviewed mechanistic interpretability research. Every approximation is disclosed.

Built for researchers. Designed for production.

---

## Research

**Paper:** [Glassbox: A Causal Mechanistic Interpretability Toolkit with Circuit Alignment Scoring](https://arxiv.org/abs/2603.09988)
Introduces the **Functional Circuit Alignment Score (FCAS)**, automated Minimum Faithful Circuit (MFC) discovery, and bootstrap CIs on circuit faithfulness. 37x faster than ACDC on GPT-2 small.

**Discussion:** [LessWrong post]([https://lesswrong.com](https://www.lesswrong.com/posts/Bkp73EaixnXdL4tKa/glassbox-i-built-a-circuit-discovery-toolkit-that-s-37)) — technical deep-dive with caveats, questions on FCAS validity, and discussion of backup mechanisms.

**ICML 2026 Workshop submission deadline: April 24, 2026.**

---

## What's Novel

Glassbox v2.6.0 ships features that exist **nowhere else as a unified toolkit**:

| Feature | Glassbox | TransformerLens | Baukit | Pyvene |
|---------|:--------:|:---------------:|:------:|:------:|
| O(3) Attribution Patching | ✅ | ✅ (manual) | ✅ (manual) | ✅ (manual) |
| Integrated Gradients (path-integral) | ✅ | ❌ | ❌ | ❌ |
| Edge Attribution Patching (Syed et al. 2024) | ✅ | ❌ | ❌ | ❌ |
| Logit Lens + Per-head Direct Effects | ✅ | Partial | ❌ | ❌ |
| Attribution Stability (Kendall τ-b) | ✅ | ❌ | ❌ | ❌ |
| **SAE Feature Attribution (sae-lens)** | ✅ | ❌ | ❌ | ❌ |
| **QK / OV Composition Scores** | ✅ | ❌ | ❌ | ❌ |
| **Token-level Saliency Maps** | ✅ | ❌ | ❌ | ❌ |
| **Attention Pattern Analysis + Head Typing** | ✅ | ❌ | ❌ | ❌ |
| Bootstrap 95% CI on faithfulness | ✅ | ❌ | ❌ | ❌ |
| Cross-model circuit alignment (FCAS) | ✅ | ❌ | ❌ | ❌ |
| MLP attribution | ✅ | ❌ | ❌ | ❌ |
| One-call API | ✅ | ❌ | ❌ | ❌ |
| Interactive dashboard (HF Spaces) | ✅ | ❌ | ❌ | ❌ |

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

# One-call circuit analysis
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

Try it instantly — no install needed: **[huggingface.co/spaces/designer-coderajay/Glassbox-ai](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)**

---

## Full API

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

# MLP attribution — which layers contribute?
mlp_attrs = gb.mlp_attribution(tokens_c, tokens_corr, t_tok, d_tok)
# Returns {layer: score} dict

# Minimum faithful circuit — greedy pruning
circuit, attrs, clean_ld = gb.minimum_faithful_circuit(tokens_c, tokens_corr, t_tok, d_tok)
```

### Logit Lens + Direct Effects

```python
# Layer-by-layer prediction tracking + per-head direct effects
ll = gb.logit_lens(tokens_c, " Mary", " John")

print(ll["logit_diffs"])
# [0.12, 0.18, 0.34, ..., 3.21]   <- LD at embedding + after each layer

print(ll["logit_shifts"])
# [0.06, 0.16, ...]   <- ΔLD per block

print(ll["head_direct_effects"][9])
# [0.02, -0.01, ..., 0.31, ...]   <- n_heads direct effects at layer 9

# Or include in analyze()
result = gb.analyze(
    "When Mary and John went to the store, John gave a drink to",
    " Mary", " John",
    include_logit_lens=True,
)
print(result["logit_lens"]["logit_diffs"])
```

### Edge Attribution Patching (EAP)

```python
# Scores every directed edge (sender → receiver) in the computation graph
# More informative than node-level AP (Syed et al. 2024)
eap = gb.edge_attribution_patching(tokens_c, tokens_corr, t_tok, d_tok, top_k=50)

for edge in eap["top_edges"][:5]:
    print(f"{edge['sender']:15s} → {edge['receiver']:15s}  score={edge['score']:.4f}")
# attn_L09H09      → resid_pre_L10    score=0.3421
# attn_L09H06      → resid_pre_L10    score=0.2187
# ...
```

### Attribution Stability

```python
# Novel metric: how stable are attribution rankings across random corruptions?
# Returns per-head stability S ∈ [0,1] + global Kendall τ-b rank consistency
stability = gb.attribution_stability(tokens_c, t_tok, d_tok, n_corruptions=25, seed=42)

print(stability["rank_consistency"])        # Kendall τ-b ∈ [-1, 1]
print(stability["top_stable_heads"][:3])    # most consistently attributed heads
```

### Token Attribution (Saliency Maps)

```python
# Which input tokens are most responsible for the prediction?
tok_attr = gb.token_attribution(tokens_c, t_tok, d_tok)

for t in tok_attr["top_tokens"]:
    sign = "+" if t["attribution"] > 0 else "-"
    print(f"  [{sign}] {t['token_str']!r:15s}  |attr|={abs(t['attribution']):.4f}")
# [+] ' Mary'           |attr|=0.4231
# [+] ' John'           |attr|=0.3187
# [-] ' gave'           |attr|=0.1043
```

### Attention Patterns + Head Typing

```python
# Full attention matrices + entropy + heuristic head type classification
attn = gb.attention_patterns(tokens_c, heads=[(9, 9), (10, 0), (5, 5)])

print(attn["entropy"])
# {'L09H09': 0.71, 'L10H00': 1.24, 'L05H05': 2.18}

print(attn["head_types"])
# {'L09H09': 'focused', 'L10H00': 'previous_token', 'L05H05': 'uniform'}

# Auto-select most "interesting" heads
attn_auto = gb.attention_patterns(tokens_c, heads=None, top_k=10)
```

### SAE Feature Attribution

> Requires: `pip install sae-lens`

```python
from glassbox import SAEFeatureAttributor

sfa    = SAEFeatureAttributor(model)          # auto-detects release for GPT-2
tokens = model.to_tokens("When Mary and John went to the store, John gave a drink to")

# Decompose residual stream at layers 9-11 into SAE features
feats  = sfa.attribute(tokens, " Mary", " John", layers=[9, 10, 11])

for f in feats["top_features"][:5]:
    print(f"  Layer {f['layer']}  Feature {f['feature_id']:5d}  "
          f"LD={f['ld_contribution']:+.4f}  dir={f['direction']}")
    if f["neuronpedia_url"]:
        print(f"    → {f['neuronpedia_url']}")
# Layer 9   Feature  4821  LD=+0.3124  dir=target
#   → https://www.neuronpedia.org/gpt2-small/9-res-jb/4821
# Layer 10  Feature 12553  LD=+0.2891  dir=target
# ...

# Circuit-scoped SAE attribution (which features activate in circuit heads?)
circuit_feats = sfa.attribute_circuit_heads(
    result["circuit"], tokens, " Mary", " John",
)
print(circuit_feats["top_features"][:3])
```

### Head Composition Scores (Elhage et al. 2021)

```python
from glassbox import HeadCompositionAnalyzer

comp = HeadCompositionAnalyzer(model)

# Q-composition between head (5,5) → (9,9)
# Does (9,9)'s queries attend to what (5,5) wrote?
q_score = comp.q_composition_score(5, 5, 9, 9)
print(f"Q-comp (5,5)→(9,9): {q_score:.4f}")

# Full composition matrix for a circuit
circuit = [(5, 5), (7, 3), (9, 9), (9, 6)]
all_comp = comp.all_composition_scores(circuit, min_score=0.05)

for edge in all_comp["combined_edges"][:5]:
    print(f"  {edge['sender']} → {edge['receiver']}  "
          f"Q={edge['q']:.3f}  K={edge['k']:.3f}  V={edge['v']:.3f}")
```

### Bootstrap Faithfulness CIs

```python
# 95% confidence intervals via nonparametric bootstrap
boot = gb.bootstrap_metrics(
    prompts=[
        ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
        ("When Alice and Bob entered the room, Bob handed the key to", " Alice", " Bob"),
        # ... recommended n >= 20 for stable CIs
    ],
    n_boot=500,
)
print(boot["sufficiency"])
# {"mean": 0.82, "std": 0.06, "ci_lo": 0.71, "ci_hi": 0.91, "n": 2}
```

### Cross-model Circuit Alignment (FCAS)

```python
# Are the same circuits present across model sizes?
model_sm = HookedTransformer.from_pretrained("gpt2")
model_md = HookedTransformer.from_pretrained("gpt2-medium")

gb_sm = GlassboxV2(model_sm)
gb_md = GlassboxV2(model_md)

r_sm = gb_sm.analyze("When Mary and John went to the store, John gave a drink to", " Mary", " John")
r_md = gb_md.analyze("When Mary and John went to the store, John gave a drink to", " Mary", " John")

fcas = gb_sm.functional_circuit_alignment(r_sm["top_heads"], r_md["top_heads"], top_k=5)
print(f"FCAS GPT-2-small ↔ GPT-2-medium: {fcas['fcas']:.3f}  (z={fcas['z_score']:.2f})")
# FCAS GPT-2-small ↔ GPT-2-medium: 0.835  (z=4.21)
```

---

## CLI

```bash
# Install
pip install glassbox-mech-interp

# Analyze a prompt
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
#   ...
```

---

## Benchmarks

### IOI (Indirect Object Identification) — Wang et al. (2022)

Evaluated on the canonical IOI task across the GPT-2 family.

| Model | Layers | Heads | Suff.* | Comp. | F1 | Glassbox | ACDC | Speedup |
|-------|--------|-------|--------|-------|----|----------|------|---------|
| GPT-2 small | 12 | 12 | 80.0% | 37.2% | 48.8% | **1.2s** | 43.2s | **37×** |
| GPT-2 medium | 24 | 16 | 35.1% | 23.7% | 27.9% | **4.9s** | 115.2s | **24×** |
| GPT-2 large | 36 | 20 | 18.2% | 14.2% | 15.9% | **14.3s** | 216.0s | **15×** |

*Sufficiency is a first-order Taylor approximation. Exact causal sufficiency (requiring full ablation runs over non-circuit heads) is higher — see the [arXiv paper](https://arxiv.org/abs/2603.09988).

### Cross-model Circuit Alignment (FCAS)

| Model pair | FCAS | z-score |
|-----------|------|---------|
| GPT-2 small ↔ GPT-2 medium | 0.835 | 4.21 |
| GPT-2 small ↔ GPT-2 large | 0.783 | 3.67 |
| GPT-2 medium ↔ GPT-2 large | 0.833 | 4.18 |

High FCAS confirms the IOI circuit is structurally conserved across scale (Wang et al. 2022).

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

## Installation

```bash
# Core (no extra deps beyond PyTorch + TransformerLens)
pip install glassbox-mech-interp

# With SAE feature attribution support
pip install glassbox-mech-interp sae-lens

# Full development install
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, TransformerLens ≥ 1.0

---

## Dashboard

```bash
pip install glassbox-mech-interp gradio matplotlib
git clone https://github.com/designer-coderajay/Glassbox-AI-2.0-Mechanistic-Interpretability-tool
cd Glassbox-AI-2.0-Mechanistic-Interpretability-tool
python dashboard/app.py
```

This opens a Gradio interface at `http://localhost:7860`. Three tabs: **Circuit Analysis** (attribution heatmap + faithfulness report), **Logit Lens**, and **Attention Patterns**.

Or use the hosted version — no install needed: **[huggingface.co/spaces/designer-coderajay/Glassbox-ai](https://huggingface.co/spaces/designer-coderajay/Glassbox-ai)**

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
| `bootstrap_metrics(prompts, n_boot, alpha)` | O(3N) | Bootstrap 95% CI on Suff/Comp/F1. |
| `functional_circuit_alignment(heads_a, heads_b, top_k, n_null)` | O(1) | Cross-model FCAS with null distribution and z-score. |

### `SAEFeatureAttributor(model)` — requires `sae-lens`

| Method | Description |
|--------|-------------|
| `attribute(tokens, target, distractor, layers)` | Residual-stream SAE decomposition. Returns feature activations + LD contributions. |
| `attribute_circuit_heads(circuit, tokens, target, distractor)` | Per-head SAE attribution (linear approximation). Links each circuit head to sparse features. |

### `HeadCompositionAnalyzer(model)`

| Method | Description |
|--------|-------------|
| `q_composition_score(sl, sh, rl, rh)` | Q-composition between head (sl,sh) → (rl,rh). |
| `k_composition_score(sl, sh, rl, rh)` | K-composition. |
| `v_composition_score(sl, sh, rl, rh)` | V-composition. |
| `composition_matrix(senders, receivers, kind)` | Full score matrix. kind = "q", "k", or "v". |
| `full_circuit_composition(circuit, kind, min_score)` | All pairwise scores within a circuit. |
| `all_composition_scores(circuit, min_score)` | Q + K + V scores in one call. |

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

---

## Related Tools

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) — mechanistic interpretability library Glassbox is built on
- [sae-lens](https://github.com/jbloomAus/SAELens) — pretrained Sparse Autoencoders (required for SAE feature attribution)
- [ACDC](https://github.com/ArthurConmy/Automatic-Circuit-DisCovery) — automated circuit discovery (baseline we benchmark against: Glassbox is 15–37× faster)
- [Neuronpedia](https://www.neuronpedia.org/) — SAE feature browser (linked from SAE attribution output)

---

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">
Built by <a href="mailto:mahale.ajay01@gmail.com">Ajay Pravin Mahale</a> · MSc 2026 · Made in Germany<br>
<strong>Glassbox AI — see inside every prediction</strong>
</div>
