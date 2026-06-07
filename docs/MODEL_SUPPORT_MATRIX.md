# Model Support Matrix — what Glassbox can and cannot analyse

*Purpose: keep every capability claim truthful and audit-proof. A regulator or enterprise reviewer will test these claims; this document is the honest ground truth behind them.*

---

## The hard line: white-box vs black-box

Glassbox's core method — attribution patching via TransformerLens hooks — reads a model's **internal activations and gradients**. That requires **open weights**. This is not a Glassbox limitation; it is true of all mechanistic-interpretability circuit discovery.

| | White-box (open weights) | Black-box (closed API models) |
|---|---|---|
| Examples | GPT-2, Pythia, Llama-2/3, Mistral, Phi, Gemma, Qwen2, and other open checkpoints | Claude (Opus/Sonnet), GPT-4/5, Gemini — closed APIs |
| Hook into activations? | Yes | **No — impossible; no weight access** |
| Attribution patching / circuit discovery? | Yes | **No** |
| Faithfulness metrics (sufficiency/comprehensiveness/F1) | Yes | **No** |
| What Glassbox can provide | Full 9-section Annex IV from circuits | Only black-box documentation aids (I/O behaviour, prompts, governance metadata) — **not** circuit-level evidence |

**Plain statement for the README/plugin:** "Glassbox performs circuit-level Annex IV analysis on open-weight models. For closed API models (Claude, GPT, Gemini), only black-box documentation support is possible — circuit faithfulness cannot be computed without weight access."

> This means Glassbox cannot run white-box analysis on Claude Opus or any closed model — by anyone, ever, without the provider releasing weights. Do not claim otherwise.

## Accuracy — stated honestly

There is no "100% accuracy." Faithfulness is an approximation, by definition. The verified, defensible figures are:

- Faithfulness **F1 = 0.89 (Grade A)** vs. brute-force ablation ground truth.
- **r = 0.009** confidence↔faithfulness — exactness is not the goal; faithful *attribution* is.

Any "100%" accuracy claim is non-compliant with the project's own honesty rules and will fail technical review.

## Size tiers (open-weight) — from `classify_model_size`

These are the **actual tiers in the code** (`glassbox/large_model.py`), based on the Kaplan et al. 2020 estimate `n_params ≈ 12·L·d²`:

| Class | Param range | Default strategy | Status |
|---|---|---|---|
| small | < 1B | standard | ✅ runs today |
| medium | 1B – 7B | standard | ✅ runs today |
| large | 7B – 70B | checkpoint | ⚠️ implemented; **live ≥7B run pending GPU (next week)** |
| xlarge | 70B – 200B | checkpoint + offload | ⚠️ implemented; unverified live |
| xxlarge | **> 200B** | checkpoint + offload | ⚠️ implemented; unverified live |

So **200B+ is already classified and routed** to `checkpoint+offload`. What does **not** yet exist is a live receipt — that needs the GPU run. Until then, the truthful claim is: *"200B+ supported via memory-managed path (gradient checkpointing + CPU/NVMe offload); live verification scheduled."*

### To genuinely strengthen 200B+ support (safe, test-driven — when you're ready)
1. Add a ≥200B open-model config (e.g. a 405B-class entry) to `ARCHITECTURE_REGISTRY` in `glassbox/multi_arch.py`. *(Note: `tests/test_multi_arch.py` has 183 tests; adding a family may assert against the family count — update the tests in the same commit, TDD-style. I did not edit the registry blind, to avoid breaking that suite.)*
2. Run `estimate_memory(n_layers, d_model, ...)` for that config to confirm the recommended strategy and memory envelope.
3. Live-verify on rented GPU per `docs/LARGE_MODEL_VERIFICATION.md` (Rung 2 → scale up).

## What "the plugin works on any model" should mean

For the Claude Code / Cowork plugin: it orchestrates Glassbox against **open-weight models the user points it at**. It runs *on* Claude (Opus 4.6 etc.) as the agent driving the tool — it does **not** analyse Claude's own internals. That distinction is the difference between a true claim and a false one.

*Glassbox AI v4.3.0 — model support matrix. Accuracy figures are verified; live large-model status is honestly marked "pending."*
