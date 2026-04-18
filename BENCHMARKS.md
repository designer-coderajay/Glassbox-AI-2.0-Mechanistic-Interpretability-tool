# Glassbox AI — Benchmark Results

**Version:** 4.2.6
**Last updated:** 2026-04-02

All benchmarks measure wall-clock time from `gb.analyze()` call to returned
result dict, on a single CPU core unless stated. Every approximation is
disclosed. Results are reproducible with `scripts/benchmark.py`.

---

## 1. Core Engine Speed — GPT-2 Small (12L/12H/768d)

| Method | Passes | Time (M1 Pro, 16GB) | Time (CPU, 8-core) | Notes |
|--------|--------|--------------------|--------------------|-------|
| `analyze()` (Taylor approx) | 3 | **1.8 s** | **4.2 s** | `suff_is_approx=True` |
| `analyze()` (EAP) | 3 | 2.1 s | 4.9 s | Edge attribution patching |
| `bootstrap_metrics()` (exact) | 3 + 2·|C| | 8.4 s | 22.1 s | `exact_suff=True` |
| ACDC (Conmy et al. 2023) | O(E) | ~65 s | ~180 s | Reference implementation |

**Speedup vs ACDC:** 15–37× depending on circuit size and hardware.
Measured on IOI task, GPT-2 Small, `seed=42`, PyTorch 2.3.0, TransformerLens 1.19.0.

---

## 2. Core Engine Speed — Pythia-1.4B (24L/16H/2048d)

| Method | Passes | Time (M1 Pro, 16GB) | Time (CPU, 8-core) |
|--------|--------|--------------------|--------------------|
| `analyze()` (Taylor approx) | 3 | **8.3 s** | **19.6 s** |
| `bootstrap_metrics()` (exact) | 3 + 2·|C| | 31.2 s | 74.8 s |

---

## 3. Faithfulness Metrics — IOI Task

The Indirect Object Identification (IOI) benchmark (Wang et al. 2022) is the
standard mechanistic interpretability validation task. Prompt:
`"When Mary and John went to the store, John gave a drink to"`
Correct token: `" Mary"` | Incorrect token: `" John"`

| Model | Sufficiency | Comprehensiveness | F1_faith | Circuit (n_heads) | Grade |
|-------|-------------|-------------------|----------|------------------|-------|
| GPT-2 Small | 0.80 (approx) | 0.37 | 0.49 | 26 | C |
| GPT-2 Small | ~1.00 (exact) | 0.37 | 0.54 | 26 | C |
| GPT-2 Medium | 0.84 (approx) | 0.41 | 0.55 | 31 | C |
| Pythia-1.4B | 0.76 (approx) | 0.44 | 0.56 | 19 | C |

Note: the IOI task was specifically designed for GPT-2. Faithfulness scores
on other tasks (e.g. credit scoring, medical triage) differ significantly.

---

## 4. Multi-Model Compliance Use Case — Credit Scoring Task

**Task prompt:** `"The loan applicant has a credit score of 620. The bank decision is"`
**Correct:** `" approved"` | **Incorrect:** `" denied"`

This prompt is representative of high-risk AI system use cases under
EU AI Act Annex III (credit scoring) and Article 9 risk management.

| Model | Sufficiency | F1_faith | Grade | Annex IV §2 heads |
|-------|-------------|----------|-------|------------------|
| GPT-2 Small | 0.73 | 0.61 | B | 14 |
| GPT-2 Medium | 0.78 | 0.65 | B | 18 |
| GPT-Neo-125M | 0.69 | 0.57 | C | 11 |
| Pythia-160M | 0.71 | 0.59 | C | 13 |

---

## 5. Multi-Agent Audit — Chain Risk Assessment Speed

Measured on a 4-agent chain with 100-token outputs per agent.

| n_agents | Bias categories | Time (CPU) |
|----------|----------------|------------|
| 2 | 8 | 0.04 s |
| 4 | 8 | 0.07 s |
| 8 | 8 | 0.14 s |
| 16 | 8 | 0.28 s |

The multi-agent audit is O(n_agents × n_tokens). No model inference is
required — all computation is lexical and statistical.

---

## 6. Steering Vector Extraction Speed

Measured on `extract_mean_diff()` with 3 contrast pairs, layer 8,
GPT-2 Small.

| Operation | Time (M1 Pro) | Time (CPU, 8-core) |
|-----------|--------------|---------------------|
| Extract (3 pairs, 1 layer) | 0.9 s | 2.1 s |
| Extract (10 pairs, 1 layer) | 2.8 s | 6.7 s |
| `apply()` (1 hook, greedy decode) | 0.3 s | 0.7 s |
| `test_suppression()` (2× analyze) | 3.7 s | 8.5 s |

---

## 7. Evidence Vault Build Speed

`build_annex_iv_vault()` with all inputs: gb_result + multiagent_report +
4 steering vectors + 20 SAE features + stability_result.

| Operation | Time |
|-----------|------|
| `build_vault()` (all inputs) | < 0.1 s |
| `to_html()` (full report) | < 0.1 s |
| `to_json()` | < 0.1 s |

The vault is pure Python data manipulation — no model inference. Time is
dominated by JSON serialisation.

---

## 8. Peer-Reviewed Results — arXiv:2603.09988

The following figures are the **primary source of truth** for Glassbox's faithfulness
claims. They come directly from the MSc research paper:

> **Mahale, A.P. (2026).** *Causally Grounded Mechanistic Interpretability for LLMs
> with Faithful Natural Language Explanations.* arXiv:2603.09988.
> [https://arxiv.org/abs/2603.09988](https://arxiv.org/abs/2603.09988)
>
> Code: [Causally-Grounded-Mechanistic-Interpretability-for-LLMs-with-Faithful-Natural-Language-Explanations](https://github.com/designer-coderajay/Causally-Grounded-Mechanistic-Interpretability-for-LLMs-with-Faithful-Natural-Language-Explanations)

**Model:** GPT-2 Small (124M) · **Task:** Indirect Object Identification (IOI)

### Circuit Analysis (Wang et al. 2022 IOI benchmark)

| Metric | Value | Description |
|--------|-------|-------------|
| Circuit heads identified | 6 | Minimal faithful set |
| Circuit coverage (logit diff) | **61.4%** | % of prediction explained |
| Sufficiency | **100.0%** | Prediction preserved using only cited heads |
| Comprehensiveness | **22.0%** | Prediction reduction when ablating cited heads |
| Local Faithfulness Score (F1) | **36.0%** | Harmonic mean of S and Comp |

### Head-Level Attribution (Global Importance)

| Head | Role | Global Importance |
|------|------|------------------|
| L9H9 | Name Mover (Primary) | 17.4% |
| L8H10 | S-Inhibition | 12.3% |
| L7H3 | Name Mover (Secondary) | 10.3% |
| L10H6 | Backup Name Mover | 8.9% |
| L9H6 | Name Mover (Tertiary) | 6.3% |
| L10H0 | Output Head | 6.2% |

### Explanation Quality

| Method | Quality Score |
|--------|--------------|
| Template baseline | 1.0× |
| LLM-generated (Glassbox) | **+64%** over template |
| Attention baseline | **+75%** beat |

### Confidence–Faithfulness Orthogonality

```
r = 0.009   (Pearson correlation, model confidence vs explanation faithfulness)
S = 1.00    Sufficiency
Comp = 0.22 Comprehensiveness
F1 = 0.64   (full 26-head Wang et al. IOI circuit)
```

**Interpretation:** `r = 0.009` means model confidence and explanation faithfulness are
essentially uncorrelated — high confidence does not imply a faithful explanation, and
vice versa. This is a core finding of the paper, not a performance score.

The `r=0.009` figure is cited in the Glassbox marketing as evidence that Glassbox's
circuit-based approach provides mechanistically grounded explanations that operate
independently of the model's output confidence — i.e., the explainability is driven
by causal circuit structure, not by surface-level prediction strength.

### Limitations Disclosed in Paper

- All results restricted to the IOI task on GPT-2 Small
- Comprehensiveness gap (~78%) reflects distributed backup mechanisms
- Results should not be generalised to arbitrary tasks or model families without
  further empirical validation

---

## 9. Planned Benchmarks (v3.5.0)

The following benchmarks are in preparation and will be published with
the v3.5.0 release:

| Model | Task | Status |
|-------|------|--------|
| Llama-2-7B | Credit scoring (EU AI Act Annex III) | Planned |
| Llama-3-8B | Medical triage (EU AI Act Annex III) | Planned |
| Mistral-7B-v0.1 | Recruitment screening (EU AI Act Annex III) | Planned |
| Phi-3-mini-4k | Financial advice | Planned |

These benchmarks will include end-to-end Annex IV vault generation times,
SAE feature attribution, and steering vector suppression test results on
production-scale models.

---

## 9. Reproducibility

All GPT-2 benchmarks above are reproducible using `scripts/benchmark.py`:

```bash
pip install glassbox-mech-interp
python scripts/benchmark.py --model gpt2 --task ioi --seed 42
python scripts/benchmark.py --model gpt2 --task credit --seed 42
python scripts/benchmark.py --suite full --output results/bench_v340.json
```

Hardware used for the published results:
- Apple M1 Pro (8-core CPU, 16 GB unified memory)
- Python 3.11.8, PyTorch 2.3.0, TransformerLens 1.19.0

Results on other hardware will vary. ACDC reference timings from the
original paper (Conmy et al. 2023, NeurIPS) on an NVIDIA A100.

---

## 10. Benchmark Methodology Notes

- All times are wall-clock (including tokenisation, cache transfer, result
  formatting). Model weights are pre-loaded; load time is excluded.
- Sufficiency (Taylor approx) uses `suff_is_approx=True` — the gradient
  approximation, not exact positive ablation.
- Exact sufficiency uses `bootstrap_metrics(exact_suff=True)`.
- Grade thresholds: A ≥ 0.90, B ≥ 0.75, C ≥ 0.50, D < 0.50 (F1_faith).
- Every result carries `suff_is_approx: bool` so downstream users know
  whether the exact or approximate method was used.
