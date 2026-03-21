# Glassbox AI — Benchmark Results

**Version:** 3.4.0
**Last updated:** 2026-03-21

All benchmarks measure wall-clock time from `gb.analyze()` call to returned
result dict, on a single CPU core unless stated. Every approximation is
disclosed. Results are reproducible with `scripts/benchmark_v340.py`.

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

## 8. Planned Benchmarks (v3.5.0)

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

All GPT-2 benchmarks above are reproducible using `scripts/benchmark_v340.py`:

```bash
pip install glassbox-mech-interp
python scripts/benchmark_v340.py --model gpt2 --task ioi --seed 42
python scripts/benchmark_v340.py --model gpt2 --task credit --seed 42
python scripts/benchmark_v340.py --suite full --output results/bench_v340.json
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
