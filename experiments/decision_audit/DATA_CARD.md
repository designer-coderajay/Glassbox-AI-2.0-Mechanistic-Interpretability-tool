# Data Card — Synthetic Credit Decisions (self-owned)

**Provenance ID:** D1 (see `docs/PLAN_DECISION_AUDIT.md` §3)
**Created:** 2026-06-27 · **Owner:** Ajay Pravin Mahale · **Seed:** 42

## What this is
A fully **synthetic**, self-generated credit-application dataset used as the
ground-truth target for the faithfulness experiment. Each row has seven features
and an approve/deny label produced by a **known, deterministic rule** (see
`credit_rule.py`). There is **no real or anonymised personal data** of any kind.

## Why it exists
Because we define the decision rule, we can later verify that the circuit Glassbox
discovers actually **implements that rule** (Phase 1.7) — faithfulness validated
against ground truth, not just measured. It is also 100% legally clean: we own it.

## License / rights
**Self-generated — we own it outright.** No third-party data, no PII, no upstream
licence to satisfy. Free to use, publish, and redistribute.

## The rule (single source of truth)
`APPROVE  iff  credit_score ≥ 640  AND  dti < 0.40  AND  num_defaults == 0` — else DENY.

- **Rule-relevant features:** `credit_score`, `dti`, `num_defaults`
- **Distractor features (not in the rule):** `annual_income`, `employment_years`, `age`, `loan_amount`
  Included deliberately so Phase 1.7 can check the circuit responds to the relevant
  features and ignores the distractors.

## Schema
| Feature | Type | Range (generated) | In rule? |
|---|---|---|---|
| credit_score | int | 300–850 (~N(680,80)) | ✅ |
| dti | float | 0.00–0.80 (~N(0.35,0.15)) | ✅ |
| num_defaults | int | 0–3 (P0=0.72) | ✅ |
| annual_income | int | 15k–250k | distractor |
| employment_years | float | 0–40 | distractor |
| age | int | 18–75 | distractor |
| loan_amount | int | 1k–80k | distractor |
| label | str | `approved` / `denied` | = rule(features) |

## Splits & balance (measured)
- **train:** 2,000 rows · approved rate **0.498**
- **test:** 500 rows · approved rate **0.506**
- Natural (unbalanced) approval rate in the raw pool: **0.312**; balanced to ~50/50
  by subsampling so a model cannot win by guessing the majority class.

## Integrity checks (passed)
- Reproducible: identical file hashes on re-run from the seed.
- Ground truth: **100%** of 2,500 labels exactly match `credit_rule.decide()` (0 mismatches).

## Files
`data/train.csv`, `data/test.csv` (tabular) · `data/train.jsonl`, `data/test.jsonl`
(`{"prompt","completion"}` for LLM fine-tuning + Glassbox) · `data/stats.json`.

## Prompt format (for fine-tuning + audit)
`"Credit application — score: …, debt-to-income: …, prior defaults: …, annual income: $…, age: …, employment: … yrs, loan: $…. Decision:"` → ` approved` / ` denied`

## Limitations (stated honestly)
- A demonstration dataset, **not** a real underwriting distribution. Feature
  correlations are simplified and labels are noise-free by design.
- A model trained on this is a **demonstration model**, not a production underwriter.
- Accuracy on this data ≠ a faithful explanation — that is exactly what Phase 1.6/1.7 test.

## Reproduce
```bash
cd experiments/decision_audit && python3 generate_credit_data.py
```
