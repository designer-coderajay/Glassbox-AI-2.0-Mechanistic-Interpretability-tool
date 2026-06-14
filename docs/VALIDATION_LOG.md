# Glassbox — Validation Log

A reproducible record of what has actually been run, on what hardware, with what
result. Only runs that happened are listed. "Not yet validated" means exactly
that — no claim is made beyond these rows.

Harnesses: `scripts/validate_auditable_gpt2.py` (single-model gate),
`scripts/validate_models_matrix.py` (multi-model matrix). Probe: IOI
("When Mary and John went to the store, John gave a drink to" → " Mary" vs
" John"), a known cross-model capability used to test that the pipeline works on
an architecture — not a business decision.

---

## Run 1 — multi-architecture + billion-scale matrix

- **Date:** 2026-06-14
- **Hardware:** Apple Silicon Mac (arm64), **CPU only**
- **Stack:** torch 2.10.0, transformer-lens 2.17.0, transformers 4.57.6,
  glassbox-mech-interp 4.5.0 (`.venv-torch`)
- **What ran:** for each model — head-level `AuditableModel` adapter +
  `run_conformance` (determinism + patch-identity on real forward hooks), then a
  full `analyze()` on the IOI probe (circuit + ERASER faithfulness).

| Model | Family | Arch | Conformance | Circuit | F1 | Grade | Time |
|---|---|---|---|---|---|---|---|
| distilgpt2 | GPT-2 | 6L×12H | **PASS** | 1 | 1.00 | A | 160s* |
| gpt2 | GPT-2 | 12L×12H | **PASS** | 1 | 0.704 | B | 7s |
| EleutherAI/pythia-70m | Pythia / GPT-NeoX | 6L×8H | **PASS** | 17 | 0.00 | D¹ | 76s |
| EleutherAI/pythia-160m | Pythia / GPT-NeoX | 12L×12H | **PASS** | 1 | 0.97 | A | 178s* |
| EleutherAI/gpt-neo-125M | GPT-Neo | 12L×12H | **PASS** | 1 | 1.00 | A | 119s* |
| facebook/opt-125m | OPT | 12L×12H | **PASS** | 1 | 0.554 | C | 128s* |
| EleutherAI/pythia-1b | Pythia / GPT-NeoX | 16L×8H | **PASS** | 1 | 0.867 | A | **1335s** |

\* Time includes the first-time model download. ¹ pythia-70m: the tool emitted
*"clean logit diff −0.974 (≤0)… the model prefers the distractor… treat results
as unreliable"* — i.e. the model cannot perform IOI, and Glassbox correctly
flagged the result as unreliable rather than reporting a misleading circuit. This
is intended honest behaviour, not a gate failure (conformance still PASSED).

**Result:** conformance gate **7/7 PASS** across **4 architecture families**
(GPT-2, Pythia/GPT-NeoX, GPT-Neo, OPT) and **up to ~1B parameters**
(`pythia-1b` ≈ 1.0B actual; TransformerLens's internal heuristic over-reported
~2.1B). The full analyze→circuit→faithfulness pipeline ran end to end on every
model, and self-flagged the one model (pythia-70m) that cannot perform the probe.

**Cost note:** the 1B audit took **~22 minutes on CPU**. This confirms Glassbox
is an **offline / sampled** audit tool (not real-time) and that **GPU is required
for speed and for anything larger**.

---

## Run 2 — GPU (Google Colab T4) + first GQA architecture

- **Date:** 2026-06-14
- **Hardware:** Google Colab, **NVIDIA Tesla T4 (16 GB)**, CUDA
- **Stack:** torch 2.10.0+cu128, transformer-lens 2.17.0, transformers 4.57.6
  (note: `torchvision` had to be uninstalled — Colab's build mismatched torch and
  broke `transformers`' lazy import; not used by Glassbox).

| Model | Family | Arch | Conformance | F1 | Grade | Time |
|---|---|---|---|---|---|---|
| EleutherAI/pythia-1b | Pythia / GPT-NeoX | 16L×8H | **PASS** | 0.867 | A | **25.7s** |
| **Qwen/Qwen2-0.5B** | **Qwen2 (GQA)** | 24L×14H | **PASS** | 0.357 | D¹ | 17.5s |
| EleutherAI/pythia-1.4b | Pythia / GPT-NeoX | 24L×16H | **PASS** | 0.54 | C | 62.8s |

¹ Qwen2-0.5B: conformance PASS is the signal that the **GQA** read/patch round-trip
is correct; the low IOI F1 is the probe being weak on a 0.5B model, honestly
reported (not a gate failure).

**Two findings:**
1. **GPU speedup** — the same pythia-1b audit went from **~22 min (CPU) to ~26 s
   (T4)**, ~50×. Confirms GPU makes per-decision audits practical.
2. **GQA architecture validated** — `Qwen2-0.5B` (grouped-query attention) passes
   the conformance gate. This closes the largest "supported but unproven" gap at
   small scale.

**T4 ceiling:** `pythia-2.8b` (fp32 ≈ 11 GB weights + backward) was interrupted —
out of memory territory on a 16 GB T4 in fp32. 2.8B+ needs a larger GPU or
bf16/fp16 loading; 7B needs an A100-class card.

---

## Run 3 — fp16 push on the T4 (hardware ceiling reached)

- **Date:** 2026-06-14, Colab **T4 16 GB**, `--dtype float16`.

| Model | Conformance | Faithfulness | Notes |
|---|---|---|---|
| EleutherAI/pythia-2.8b | **PASS** (50.7s) | F1 0.347 — **NOT trustworthy** | fp16: TL advises `from_pretrained_no_processing`; attribution gradients degrade in fp16. Gate passes; the F1 is a precision artifact, not a result. |
| EleutherAI/pythia-6.9b | — | — | Loaded (~14 GB fp16) but the audit's backward pass **OOM'd on the 16 GB T4** (interrupted). 6.9B+ needs ≥24 GB (L4) / 40 GB (A100). |

**Conclusion — T4 ceiling:** conformance gate validated to **2.8B**; **trustworthy
faithfulness to ~1.4B** (fp32). fp16 buys memory but costs faithfulness accuracy,
and 6.9B will not fit on 16 GB regardless. Going larger *and* trustworthy requires
a bigger GPU + **bf16** (stable, unlike fp16) — an A100/L4.

---

## Run 4 — A100/L4 bf16: 7B + GQA (Mistral)

- **Date:** 2026-06-14, Colab **A100/L4-class GPU**, `--dtype bfloat16`.

| Model | Family | Arch | Conformance | F1 (bf16) | Time |
|---|---|---|---|---|---|
| EleutherAI/pythia-2.8b | Pythia / GPT-NeoX | 32L×32H | **PASS** | 0.313 (D)† | 41.5s |
| EleutherAI/pythia-6.9b | Pythia / GPT-NeoX | 32L×32H | **PASS** | 0.324 (D)† | 73.4s |
| **mistralai/Mistral-7B-v0.1** | **Mistral (GQA)** | 32L×32H | **PASS** | 0.363 (D)† | 67.8s |

† **F1 not trustworthy at this precision/scale.** All three graded D, *including
Mistral-7B which is known to perform IOI well* — a strong indicator the
faithfulness numbers are degraded, not real. Two candidate causes, not yet
separated: (1) **bf16** loses precision in the attribution backward pass (TL warns
`from_pretrained_no_processing`); (2) **method scaling** — minimal-circuit pruning
returned 1–2 heads, which may be too few to be *sufficient* in a 7B model. The
**conformance gate is forward-only (no gradients), so its PASS is unaffected by
precision and IS trustworthy.**

**Headline:** conformance gate **validated to 7B and on a real GQA model
(Mistral-7B)** — the previously-unproven architecture+scale claim. Faithfulness
*quality* at 2.8B–7B remains **open** pending an fp32 comparison (below).

**Decisive test — ANSWERED (Run 5).** `pythia-2.8b` re-run in **fp32**:
**F1 0.34 (D), circuit = 1 head** — essentially identical to bf16's 0.313.
**Precision is ruled out.** The low faithfulness at scale is a **method-scaling
limitation**, not numerical.

Evidence (all `circuit = 1`, fp32): pythia-1b F1 **0.87 (A)** → pythia-1.4b
**0.54 (C)** → pythia-2.8b **0.34 (D)**. The minimal-circuit pruning returns a
**single head regardless of model size**, and one head becomes insufficient as the
model grows (IOI spreads across more heads in larger models → sufficiency falls →
F1 falls).

**Conclusion — honest cap on the *faithfulness* claim:** with the current
`minimum_faithful_circuit` pruning, trustworthy faithfulness holds to **~1–1.4B**;
beyond that the circuit is under-sized and F1 is not meaningful. The
**conformance gate is unaffected and validated to 7B + GQA.** The fix is a
**scale-aware circuit selection** (keep adding heads until *measured* sufficiency
reaches a target, instead of stopping at ~1 head) — a real method change requiring
GPU re-validation, not a tuning tweak. Tracked as the top research item.

---

## What is NOT yet validated (no claims made here)

- **Trustworthy faithfulness above ~1.4B** — the gate passes to 7B, but F1 at
  2.8B–7B (bf16) is degraded/inconclusive; fp32 comparison needed to separate
  precision from method-scaling. No faithfulness claim is made above 1.4B.
- **Other GQA families** — validated on **Qwen2-0.5B** and **Mistral-7B** (gate).
  Llama-3 / Gemma use the same mechanism and *likely* work but were not run (need
  an HF token).
- **13B+ frontier scale** — needs a GPU (40–80GB); not yet run.
- **70B–200B** — requires multi-GPU sharding via the native-HF/distributed
  backend, which is built as an interface but **not yet validated**. No 200B run
  has been performed; no 200B claim is made.
- **GPU batching / production throughput** — unproven.
- **Closed APIs** (GPT-4, Claude, Gemini) — out of scope by construction (no
  activations/gradients available).

---

## How to reproduce

```bash
pip install "torch==2.10.0" "transformer-lens==2.17.0" "transformers==4.57.6" scipy
pip install -e . --no-deps
python scripts/validate_models_matrix.py            # the small-model matrix
python scripts/validate_models_matrix.py --models EleutherAI/pythia-1b --device cpu --max-units 12
# On a GPU box, add --device cuda and larger --models (e.g. a small GQA model with an HF token).
```
