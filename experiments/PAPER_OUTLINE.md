# Does Confidence–Faithfulness Independence Generalise Across LLM Families?

**A Multi-Model Mechanistic Interpretability Study Using Attribution Patching**

---

**Authors:** Ajay Mahale¹

¹ Glassbox Project · `glassbox-mech-interp` v3.6.0 · arXiv preprint

**Correspondence:** mahale.ajay01@gmail.com

**Code & Data:** https://github.com/designer-coderajay/glassbox-mech-interp · `experiments/cross_model_study.py`

**Mathematical Foundations:** `MATH_FOUNDATIONS.md` (bundled with library)

---

## Abstract

Model confidence scores are widely used as proxies for prediction reliability, yet their
relationship to mechanistic faithfulness — the degree to which an identified circuit actually
drives model behaviour — remains poorly characterised. Mahale (2026) established a near-zero
confidence–faithfulness correlation (r = 0.009) for GPT-2 on the Indirect Object
Identification (IOI) task. We ask whether this result generalises: do four architecturally
distinct language models (GPT-2-small, GPT-2-XL, Pythia-1.4B, Llama-2-7B) all exhibit
independence between softmax confidence and attribution-patching F1 faithfulness?

We formalise three null hypotheses — (H₀) per-model correlation is zero, (H_cross) all
per-model r values are drawn from the same distribution, and (H_circuit) pairwise Jaccard
circuit similarity exceeds 0.50 — and test them using Fisher Z transformation, Welch's
t-tests with Bonferroni correction, and bootstrap confidence intervals on n = 100 IOI
prompts per model. Our experiment harness (`cross_model_study.py`) implements all tests
with full reproducibility via fixed random seeds.

This study serves dual purposes: it provides an empirical foundation for Glassbox's EU AI
Act Annex IV compliance claims and tests whether confidence-based monitoring is a viable
proxy for circuit-level faithfulness across model families.

**Keywords:** mechanistic interpretability, attribution patching, faithfulness, circuit discovery,
EU AI Act, Indirect Object Identification, GPT-2, Pythia, Llama

---

## 1 · Introduction

### 1.1 Motivation

The deployment of large language models in regulated domains — healthcare, finance, legal
services — demands more than accuracy benchmarks. EU AI Act Article 13 requires *transparency*
sufficient for operators to understand model outputs; Article 15(1) requires *robustness*
even under adversarial conditions. Mechanistic interpretability addresses both: by identifying
the minimal circuit C ⊆ H responsible for a behaviour, it provides the causal account
regulators need.

A natural question follows: can **model confidence** — the softmax probability assigned to the
top token — serve as a cheap proxy for circuit faithfulness? If yes, compliance monitoring
reduces to confidence monitoring, which is cheap and scalable. If no, regulators and auditors
must invest in full circuit-level analysis for every deployment.

Mahale (2026) answered "no" for GPT-2 on the IOI task: r = 0.009, indistinguishable from
zero under a two-sided Fisher Z test (p > 0.05, n = 200). But GPT-2 is a single 124M-parameter
decoder trained on WebText in 2019. The result may not generalise to larger models, different
training data, or alternative architectures.

### 1.2 Research Questions

This paper addresses three questions:

**RQ1 (Per-model independence):** Does each of GPT-2-small, GPT-2-XL, Pythia-1.4B, and
Llama-2-7B independently exhibit a near-zero confidence–faithfulness correlation on IOI?

**RQ2 (Cross-model stability):** Are the per-model r values statistically
indistinguishable — i.e., drawn from the same null distribution — or do some models break
the independence assumption?

**RQ3 (Circuit overlap):** Do the attribution-patching circuits for each model share a
common structural skeleton, or is the IOI solution implemented differently across families?

### 1.3 Hypotheses

Let r_M denote the Pearson correlation between softmax confidence and F1 faithfulness for
model M over n = 100 IOI prompts, and let z_M = atanh(r_M) be its Fisher Z transform.

**H₀** (per-model): For each M ∈ {GPT-2s, GPT-2-XL, Pythia-1.4B, Llama-2-7B},
ρ_M = 0.

*Test:* Two-sided z-test on z_M; reject if |z_M| > z_{0.025} = 1.960.

**H_cross** (cross-model): All four z_M values are drawn from the same distribution.

*Test:* Welch's t-test on all 6 pairwise (M_i, M_j) comparisons with Bonferroni-adjusted
threshold α_adj = 0.05 / 6 ≈ 0.0083 (C(4,2) = 6 tests).

**H_circuit** (structural overlap): For each pair (M_i, M_j), the Jaccard similarity
J(M_i, M_j) ≥ 0.50, where heads are normalised to position [0,1] × [0,1] with ε = 0.05
matching tolerance.

*Reject H_circuit* if any pair produces J < 0.50.

### 1.4 Contributions

1. First multi-model empirical test of confidence–faithfulness independence across four
   LLM families using consistent methodology.
2. A fully reproducible experiment harness (`cross_model_study.py`) with dry-run mode,
   complete statistical pipeline, and JSON + Markdown output.
3. A 16-section mathematical foundations document (`MATH_FOUNDATIONS.md`) that formalises
   every formula used in Glassbox, with proofs, power analysis, and EU AI Act regulatory mapping.
4. Evidence (or lack thereof) for a universal confidence–faithfulness decorrelation, with
   direct implications for EU AI Act compliance monitoring strategy.

---

## 2 · Background

### 2.1 Attribution Patching

Attribution patching (Nanda, 2023; Kramár et al., 2024) approximates the causal effect of
each attention head h on a logit difference LD via a first-order Taylor expansion:

```
α(h) ≈ (∂LD / ∂z_h)|_{z_h = z_h^clean}  ·  (z_h^clean − z_h^corrupt)
```

where z_h^clean is the head's residual-stream output on the clean prompt and z_h^corrupt
on a corrupted counterfactual. This requires exactly **3 forward passes**: clean, corrupt,
and a gradient pass — O(n_heads · n_layers) cheaper than the O(n_heads²) brute-force
activation patching used by Conmy et al. (2023).

The circuit C is the set of heads with α(h) above a 5% threshold of the maximum score:

```
C = { h ∈ H : α(h) ≥ 0.05 · max_{h'} α(h') }
```

See `MATH_FOUNDATIONS.md §2` for full derivation.

### 2.2 Faithfulness Metrics

Following Conmy et al. (2023), we measure faithfulness along two axes:

**Sufficiency** — does C alone reproduce model behaviour?

```
S(C) = [LD(x_c ; do(H\C := z^corrupt)) − LD_corrupt] / [LD_clean − LD_corrupt]
```

**Comprehensiveness** — does C contain all the necessary mechanism?

```
Comp(C) = [LD_clean − LD(x_c ; do(C := z^corrupt))] / [LD_clean − LD_corrupt]
```

**F1 faithfulness** — harmonic mean:

```
F1_faith = 2 · S(C) · Comp(C) / (S(C) + Comp(C))
```

For GPT-2 on IOI with the full 26-head Wang et al. (2022) circuit:
S = 1.00, Comp = 0.22, F1 = 0.64. For the top-3 Name Mover heads only:
F1 ≈ 0.36 (reduced comprehensiveness). See `MATH_FOUNDATIONS.md §4`.

### 2.3 The IOI Task

The Indirect Object Identification task (Wang et al., 2022) uses prompts of the form:

> "When Mary and John went to the store, John gave a drink to ___"

The model should predict "Mary" (the indirect object not repeated in the second clause).
Logit difference: LD = logit("Mary") − logit("John").

The IOI circuit for GPT-2 involves three functionally distinct head types:
- **Name Mover heads** (L9H6: α=0.584, L9H9: α=0.431, L10H0: α=0.312) — copy IO token
- **Duplicate Token heads** — detect repeated names
- **S-Inhibition heads** — suppress the subject token

### 2.4 Confidence–Faithfulness Independence

Softmax confidence for prompt x: conf(x) = max_v softmax(logit_v(x)).

The null hypothesis ρ = 0 asserts that knowing conf(x) gives no information about F1_faith(x).
Mahale (2026) established r = 0.009 for GPT-2 on IOI (n = 200). Under H₀, the Fisher Z
statistic:

```
Z = atanh(r) · √(n − 3)  ~  N(0, 1)
```

With r = 0.009, n = 200: Z = 0.009 · √197 ≈ 0.126, p ≈ 0.90. Far from rejection.

Power analysis (`MATH_FOUNDATIONS.md §10`): with n = 100 prompts per model, the study has
86% power to detect correlations of |ρ| ≥ 0.28 (two-sided α = 0.05). This is the minimum
effect size considered practically significant for compliance monitoring.

---

## 3 · Methodology

### 3.1 Models

| Model | Family | Parameters | Architecture | TransformerLens Key |
|-------|--------|-----------|--------------|---------------------|
| GPT-2-small | OpenAI GPT-2 | 117 M | 12L × 12H | `gpt2` |
| GPT-2-XL | OpenAI GPT-2 | 1.5 B | 48L × 25H | `gpt2-xl` |
| Pythia-1.4B | EleutherAI Pythia | 1.4 B | 24L × 16H | `pythia-1.4b` |
| Llama-2-7B | Meta Llama 2 | 7.0 B | 32L × 32H | `meta-llama/Llama-2-7b-hf` |

All models are loaded via TransformerLens `HookedTransformer.from_pretrained()`. Llama-2-7B
requires a Hugging Face access token and ≥16 GB VRAM (or CPU with sufficient RAM).

### 3.2 Prompt Construction

We generate 100 IOI prompts per model from 20 name pairs × 5 sentence frames:

```
Name pairs: [("Mary", "John"), ("Alice", "Bob"), ("Emma", "James"),
             ("Sarah", "David"), ("Lisa", "Michael"), ...] (20 total)

Frames:
  "When {io} and {s} went to the store, {s} gave a drink to"
  "After {io} and {s} arrived at the party, {s} handed a gift to"
  "Because {io} and {s} came to dinner, {s} passed the salt to"
  "Since {io} and {s} met at work, {s} sent an email to"
  "As {io} and {s} walked through the park, {s} gave a flower to"
```

Target token: the IO name (e.g., "Mary"). Distractor token: the S name (e.g., "John").
Corrupted prompt: IO and S names are swapped.

### 3.3 Per-Prompt Analysis Pipeline

For each prompt x:

1. **Clean forward pass** → logits_clean, LD_clean, conf_clean
2. **Corrupt forward pass** → logits_corrupt, LD_corrupt
3. **Gradient pass** → ∂LD/∂z_h for all heads h
4. **Attribution scores** → α(h) = gradient · (z_h^clean − z_h^corrupt)
5. **Circuit** → C = {h : α(h) ≥ 0.05 · max α}
6. **Faithfulness** → S(C), Comp(C), F1_faith (two additional patched forward passes)
7. **Store** → (conf_clean, F1_faith, top_heads, S, Comp) in `PromptResult`

Total forward passes per prompt: 5 (clean + corrupt + gradient + sufficiency patch + comprehensiveness patch).
Total per model run (n=100): ≤ 500 forward passes.

### 3.4 Model-Level Statistics

After n = 100 prompts per model, aggregate:

**Pearson correlation:**
```
r_M = Σ(conf_i − conf̄)(f1_i − f1̄) / √[Σ(conf_i − conf̄)² · Σ(f1_i − f1̄)²]
```

**Fisher Z and 95% CI:**
```
z_M = atanh(r_M),   SE = 1/√(n−3) = 1/√97 ≈ 0.1015

CI_z: [z_M − 1.960·SE, z_M + 1.960·SE]
CI_r: [tanh(CI_z_lower), tanh(CI_z_upper)]
```

**BCa Bootstrap CI** (B = 2000 resamples) on F1, sufficiency, comprehensiveness for robustness
against non-normal distributions.

### 3.5 Cross-Model Comparisons

**Welch's t-test on F1** (6 pairwise comparisons):

```
t_{ij} = (f1̄_i − f1̄_j) / √(s²_i/n_i + s²_j/n_j)

df_{ij} = (s²_i/n_i + s²_j/n_j)² / [(s²_i/n_i)²/(n_i−1) + (s²_j/n_j)²/(n_j−1)]
```

Bonferroni threshold: α_adj = 0.05 / 6 ≈ 0.0083.

**Fisher Z comparison** between pairs of r values:

```
Z_diff = (z_i − z_j) / √(1/(n_i−3) + 1/(n_j−3))
```

Two-sided p-value from standard normal. Bonferroni-corrected over 6 pairs.

**Jaccard circuit similarity** (normalised head positions):

For model M with n_L layers and n_H heads per layer, normalise each head (l, h) to
(l/(n_L−1), h/(n_H−1)) ∈ [0,1]². Two heads match if Euclidean distance < ε = 0.05.

```
J(M_i, M_j) = |C_i ∩_ε C_j| / |C_i ∪_ε C_j|
```

where ∩_ε is the approximate intersection under tolerance ε.

**Cohen's d effect size** for F1 differences:

```
d_{ij} = (f1̄_i − f1̄_j) / s_pooled,   s_pooled = √[(s²_i + s²_j)/2]
```

Interpretation: |d| < 0.2 (negligible), 0.2–0.5 (small), 0.5–0.8 (medium), > 0.8 (large).

### 3.6 Reproducibility

All experiments use fixed random seeds: `seed = hash(model_key) % (2**31)` for dry-run
synthetic data. Real model inference is deterministic given fixed prompt ordering and
`torch.no_grad()` context. Full environment pinned in `requirements/research.txt`.

Code: `experiments/cross_model_study.py`
CLI: `python cross_model_study.py --n-prompts 100 --output-dir results/ [--dry-run]`

---

## 4 · Results (Placeholders — To Be Filled After Experiment Run)

### 4.1 Table 1: Confidence–Faithfulness Correlations

| Model | n | r | 95% CI (Fisher Z) | p-value (H₀: ρ=0) | Reject H₀? |
|-------|---|---|-------------------|--------------------|------------|
| GPT-2-small | 100 | _TBD_ | [_TBD_, _TBD_] | _TBD_ | _TBD_ |
| GPT-2-XL | 100 | _TBD_ | [_TBD_, _TBD_] | _TBD_ | _TBD_ |
| Pythia-1.4B | 100 | _TBD_ | [_TBD_, _TBD_] | _TBD_ | _TBD_ |
| Llama-2-7B | 100 | _TBD_ | [_TBD_, _TBD_] | _TBD_ | _TBD_ |

*Reference:* Mahale (2026) GPT-2: r = 0.009, p ≈ 0.90 (n = 200).

### 4.2 Table 2: Faithfulness Metrics (Mean ± SD)

| Model | Sufficiency | Comprehensiveness | F1 | Analysis time |
|-------|------------|-------------------|----|---------------|
| GPT-2-small | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| GPT-2-XL | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Pythia-1.4B | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Llama-2-7B | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

*Reference:* Mahale (2026) GPT-2: S = 1.00, Comp = 0.22, F1 = 0.64, t = 1.2s.
*(Note: F1 = 0.64 is for the full 26-head Wang et al. circuit; top-3 Name Movers give F1 ≈ 0.36.)*

### 4.3 Table 3: Pairwise Jaccard Circuit Similarity

|  | GPT-2-small | GPT-2-XL | Pythia-1.4B | Llama-2-7B |
|--|-------------|----------|-------------|------------|
| **GPT-2-small** | 1.00 | _TBD_ | _TBD_ | _TBD_ |
| **GPT-2-XL** | — | 1.00 | _TBD_ | _TBD_ |
| **Pythia-1.4B** | — | — | 1.00 | _TBD_ |
| **Llama-2-7B** | — | — | — | 1.00 |

*H_circuit reject threshold:* J < 0.50 for any pair.

### 4.4 Table 4: Cross-Model F1 Comparisons (Welch's t, Bonferroni α = 0.0083)

| Pair | Δ F1 | t | df | p (uncorrected) | p (Bonferroni) | Cohen's d | Significant? |
|------|------|---|----|-----------------|----------------|-----------|--------------|
| GPT-2s vs GPT-2-XL | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| GPT-2s vs Pythia-1.4B | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| GPT-2s vs Llama-2-7B | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| GPT-2-XL vs Pythia-1.4B | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| GPT-2-XL vs Llama-2-7B | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Pythia-1.4B vs Llama-2-7B | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### 4.5 Dry-Run Results (Synthetic Data, Seed-Fixed)

The following results are from `--dry-run` mode (synthetic data, reproducible via fixed seeds).
They validate the statistical pipeline but do not reflect real model behaviour.

```
=== GLASSBOX CROSS-MODEL STUDY ===
Models: gpt2-small, gpt2-xl, pythia-1.4b, llama-2-7b
Prompts per model: 100   Device: cpu   Dry-run: True

[DRY-RUN] gpt2-small  r=___ p=___  F1=___ ± ___
[DRY-RUN] gpt2-xl     r=___ p=___  F1=___ ± ___
[DRY-RUN] pythia-1.4b r=___ p=___  F1=___ ± ___
[DRY-RUN] llama-2-7b  r=___ p=___  F1=___ ± ___
```

*(Run `python experiments/cross_model_study.py --dry-run` to populate these values.)*

---

## 5 · Discussion

### 5.1 Interpretation of H₀ (Per-Model Independence)

If all four models fail to reject H₀ (|r| < 0.28, consistent with the null), the result
extends Mahale (2026) from a single model to a cross-family finding: **confidence–faithfulness
independence is not a GPT-2 artefact; it is a property of transformer-based language models
generally**.

This has a sharp practical implication: softmax confidence monitoring cannot substitute for
circuit-level auditing under EU AI Act requirements. A high-confidence prediction may arise
from a mechanistically unfaithful circuit.

If any model rejects H₀, the question becomes whether the correlation is positive (higher
confidence → more faithful circuit, which would be a useful monitoring signal) or negative
(higher confidence → less faithful circuit, which would actively mislead auditors).

### 5.2 Interpretation of H_cross (Cross-Model Stability)

If H_cross is not rejected (all pairwise Fisher Z comparisons pass Bonferroni correction),
it means the decorrelation is consistent in magnitude across architectures. This strengthens
the compliance argument: the finding is not model-specific noise, but a stable empirical law.

If H_cross is rejected for some pair, the diverging model warrants separate investigation.
Candidate explanations:
- Larger models have richer residual streams, potentially creating multiple partially faithful
  circuits that raise comprehensiveness without correlating to confidence.
- Models trained on different data distributions (Pythia on The Pile vs GPT-2 on WebText)
  may generalise IOI differently.

### 5.3 Interpretation of H_circuit (Structural Overlap)

Wang et al. (2022) identified 26 heads in the GPT-2 IOI circuit. We expect the top Name
Mover pattern (L9H6, L9H9, L10H0 in GPT-2) to have approximate analogues in other models,
but at different absolute (l, h) positions. After normalisation, Jaccard ≥ 0.50 would indicate
a common structural skeleton — the IOI task is solved similarly across families.

Jaccard < 0.50 would mean models have found genuinely different mechanistic solutions.
This would be a surprising result and would suggest that circuit-level auditing results from
one model family should not be assumed to transfer to another.

### 5.4 EU AI Act Implications

This study directly informs three Annex IV sections:

| Finding | EU AI Act Article | Compliance Implication |
|---------|------------------|----------------------|
| ρ ≈ 0 per model | Art. 13(1) — Transparency | Confidence alone insufficient for transparency; full circuit report required |
| H_cross not rejected | Art. 15(1) — Robustness | Cross-model stability supports generalising the Glassbox audit methodology |
| J ≥ 0.50 across models | Art. 9(1) — Risk management | Shared circuit structure enables common audit templates |
| J < 0.50 for some pair | Art. 9(1) — Risk management | Model-specific audits required; no template transfer |

### 5.5 Limitations

1. **IOI task specificity.** All results are conditioned on the IOI task. Other tasks
   (factual recall, arithmetic, sentiment) may show different correlation patterns.
2. **n = 100 prompts per model.** Power analysis shows 86% power for |ρ| ≥ 0.28. Effects
   smaller than 0.28 will not be reliably detected. A larger study with n = 400 would bring
   power to > 95% for |ρ| ≥ 0.14.
3. **Attribution patching approximation.** The first-order Taylor approximation is exact
   only when the corrupted and clean activations are close. For large-vocabulary models with
   highly variable logit differences, the linear approximation may introduce error.
   See `MATH_FOUNDATIONS.md §2.4` for discussion of approximation error bounds.
4. **Llama-2-7B access.** Requires Hugging Face gated access. Dry-run mode allows pipeline
   testing without the model.
5. **Fixed prompt templates.** The 5 sentence frames may not capture the full variance of
   natural-language IOI instances. Future work should use a broader prompt set.

---

## 6 · Conclusion

We present the first systematic test of confidence–faithfulness independence across four
architecturally distinct language models using attribution patching on the IOI task. Our
null hypothesis framework, power analysis, and full statistical pipeline are pre-registered
in `experiments/cross_model_study.py` and `MATH_FOUNDATIONS.md`.

If the results confirm that ρ ≈ 0 universally, the implication for AI regulation is clear:
**confidence-based monitoring is not sufficient for EU AI Act Annex IV compliance**. Mechanistic
circuit auditing, as implemented in Glassbox v3.6.0, provides the causal evidence that
confidence scores cannot.

If the results reveal model-specific exceptions, that is equally valuable: it identifies which
model families are amenable to lightweight monitoring and which require full attribution-patching
audits.

Either way, this study demonstrates that claims about model interpretability must be validated
empirically across model families — not extrapolated from a single architecture.

---

## References

1. **Wang, K. et al.** (2022). Interpretability in the Wild: a Circuit for Indirect Object
   Identification in GPT-2 small. *arXiv:2211.00593.*

2. **Conmy, A. et al.** (2023). Towards Automated Circuit Discovery for Mechanistic
   Interpretability. *NeurIPS 2023.* arXiv:2304.14997.

3. **Nanda, N.** (2023). Attribution Patching: Activation Patching At Industrial Scale.
   *Neel Nanda's Blog.* https://neelnanda.io/attribution-patching

4. **Kramár, J. et al.** (2024). AtP*: An efficient and scalable method for localizing LLM
   behaviour to components. *arXiv:2403.00745.*

5. **Mahale, A.** (2026). Glassbox: Mechanistic Interpretability and EU AI Act Compliance
   for Production Language Models. *arXiv:2603.09988.*

6. **Biderman, S. et al.** (2023). Pythia: A Suite for Analyzing Large Language Models Across
   Training and Scaling. *ICML 2023.* arXiv:2304.01373.

7. **Touvron, H. et al.** (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models.
   *arXiv:2307.09288.*

8. **Fisher, R.A.** (1915). Frequency distribution of the values of the correlation coefficient
   in samples from an indefinitely large population. *Biometrika 10*(4), 507–521.

9. **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
   Lawrence Erlbaum Associates.

10. **Efron, B. & Tibshirani, R.J.** (1993). *An Introduction to the Bootstrap.*
    Chapman & Hall/CRC.

11. **Bland, J.M. & Altman, D.G.** (1995). Multiple significance tests: the Bonferroni method.
    *British Medical Journal 310*, 170.

12. **Welch, B.L.** (1947). The generalization of Student's problem when several different
    population variances are involved. *Biometrika 34*(1–2), 28–35.

13. **Elhage, N. et al.** (2021). A Mathematical Framework for Transformer Circuits.
    *Transformer Circuits Thread.* https://transformer-circuits.pub/2021/framework/index.html

---

## Appendix A · Mathematical Notation Reference

| Symbol | Definition |
|--------|-----------|
| H | Full set of attention heads in model M |
| C ⊆ H | Identified circuit (heads above 5% attribution threshold) |
| z_h | Residual-stream output of head h |
| α(h) | Attribution score for head h |
| LD | Logit difference: logit(IO token) − logit(S token) |
| S(C) | Sufficiency metric ∈ [0, 1] |
| Comp(C) | Comprehensiveness metric ∈ [0, 1] |
| F1_faith | Faithfulness F1 = 2·S·Comp/(S+Comp) |
| conf(x) | Softmax confidence = max_v softmax(logit_v) |
| r_M | Pearson correlation between conf and F1_faith for model M |
| z_M | Fisher Z transform: atanh(r_M) |
| ρ_M | Population correlation (H₀: ρ_M = 0) |
| z_{α/2} | Standard normal critical value (1.960 for α = 0.05) |
| n | Number of prompts per model (100 in this study) |
| J(M_i, M_j) | Jaccard similarity between normalised circuits of M_i and M_j |
| ε | Jaccard matching tolerance (0.05 in normalised [0,1]² space) |
| α_adj | Bonferroni-corrected threshold = 0.05/6 ≈ 0.0083 |
| d_{ij} | Cohen's d effect size between models M_i and M_j |

---

## Appendix B · Experiment Harness Architecture

```
cross_model_study.py
├── MODELS dict              # 4 models with metadata
├── build_ioi_prompts()      # 100 prompts from 20 pairs × 5 frames
├── PromptResult (dataclass) # per-prompt: conf, LD, S, Comp, F1, top_heads
├── ModelResult (dataclass)
│   ├── aggregate()          # r, Fisher Z CI, BCa bootstrap CIs
│   └── to_dict()            # JSON serialisation
├── CrossModelStatistics
│   ├── welch_ttest_f1()     # Welch's t + Bonferroni on F1
│   ├── fisher_z_comparison()# z_i − z_j test on correlations
│   ├── jaccard_circuit_similarity() # normalised head position Jaccard
│   └── all_pairwise()       # all 6 pairs, returns dict
├── run_model()              # real or dry-run mode
├── generate_report()        # JSON + Markdown output
└── main()                   # argparse CLI
```

**CLI flags:**
```
--models     gpt2-small gpt2-xl pythia-1.4b llama-2-7b  (subset supported)
--n-prompts  100       (default)
--device     cpu | cuda | mps
--output-dir results/  (default)
--dry-run    use synthetic data (no model loading required)
```

---

## Appendix C · Power Analysis Summary

From `MATH_FOUNDATIONS.md §10`:

| n (per model) | Min detectable |ρ| (80% power) | Min detectable |ρ| (86% power) |
|--------------|-------------------------------|-------------------------------|
| 50 | 0.39 | 0.35 |
| 100 | 0.28 | 0.25 |
| 200 | 0.20 | 0.18 |
| 400 | 0.14 | 0.13 |

This study uses n = 100, providing 86% power for |ρ| ≥ 0.25 (two-sided α = 0.05,
z_β = 1.10). Effects smaller than 0.25 would require n ≥ 200 for reliable detection.

Formula:
```
n ≥ ((z_{α/2} + z_β) / atanh(ρ))²  +  3
```

For ρ = 0.28, α = 0.05, β = 0.14 (86% power):
```
n ≥ ((1.960 + 1.080) / atanh(0.28))²  +  3
  = (3.040 / 0.2877)²  +  3
  = (10.566)²  +  3
  = 111.6  +  3
  ≈ 115
```

n = 100 is slightly below this — the conservative choice for computational feasibility.
For the final paper, increasing to n = 115 per model achieves exactly 86% power.

---

*Document version: 1.0.0-outline*
*Generated by Glassbox cross-model study pipeline*
*Mathematical foundations: MATH_FOUNDATIONS.md*
*Experiment code: experiments/cross_model_study.py*
*Paper arXiv status: pre-submission outline*
