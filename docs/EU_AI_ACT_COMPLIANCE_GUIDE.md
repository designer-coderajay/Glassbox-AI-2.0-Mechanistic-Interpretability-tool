# How to Get Your EU AI Act Annex IV Compliance Proof

**A 7-step guide — from `pip install` to a regulator-ready evidence package in under 2 seconds.**

> **Scope & disclaimer.** Glassbox generates the *technical* portion of Annex IV documentation from a model's internal circuits. It is engineering tooling, not legal advice. Whether a specific system is "high-risk," and whether machine-generated documentation satisfies a given national authority or notified body, is a legal determination you should confirm with qualified counsel. Article references below point to the EU AI Act (Regulation (EU) 2024/1689) for orientation, not as legal opinion.

---

## Why this matters

From **August 2, 2026**, providers of high-risk AI systems placed on the EU market must maintain technical documentation as set out in **Annex IV**. Non-compliance penalties under **Article 99** reach **up to €35,000,000 or 7% of total worldwide annual turnover**, whichever is higher.

Writing Annex IV by hand takes weeks of combined ML + legal effort per model. Annex IV of Regulation (EU) 2024/1689 has **9 numbered points** ([primary source](https://artificialintelligenceact.eu/annex/4/)). Glassbox produces the full **9-section Annex IV structure** automatically from the model's own circuits — with one honest caveat: **Section 8 is the EU Declaration of Conformity (Article 47), a legal attestation the provider must sign.** Glassbox supplies the structured reference/placeholder for it, not the signed declaration itself. So the accurate claim is "generates the full 9-section structure; 8 of 9 are produced from the model, 1 is a provider-signed legal reference."

| Capability | Verified figure |
|---|---|
| Annex IV sections (full structure) | 9 of 9 (§8 is a provider-signed legal reference) |
| Time to full audit (GPT-2 Small) | < 2 seconds |
| Circuit discovery vs. ACDC baseline | 37× faster (1.2s vs 43.2s) |
| Faithfulness F1 | 0.89 (Grade A) |
| Confidence ↔ faithfulness correlation | r = 0.009 (orthogonal — see Step 6) |

*These numbers are the locked ground truth. Do not round or embellish them in any customer-facing material.*

---

## Step 1 — Install

```bash
pip install glassbox-mech-interp
glassbox-ai doctor          # verifies torch + transformer_lens are correctly installed
glassbox-ai version
```

No network calls are required at audit time — the engine runs fully local, which is what air-gapped regulated environments need.

## Step 2 — Confirm your system is in scope

A system is generally high-risk if it falls under **Article 6** in combination with **Annex III** use cases. The domains that matter most for Glassbox users:

- **Credit & creditworthiness** (Annex III §5(b)) — loan scoring, limit decisions
- **Medical / safety triage** (Annex III §5, and medical-device overlap)
- **Employment** (Annex III §4) — CV screening, candidate ranking
- **Biometrics** (Annex III §1)
- **Essential services & critical infrastructure** (Annex III §2, §5)

If your LLM makes or materially informs a decision in one of these, you almost certainly need Annex IV documentation. (Confirm the classification with counsel — this is the one step Glassbox cannot do for you.)

## Step 3 — Run a circuit audit on your real prompt

Glassbox is **not** limited to IOI / name-swap tasks. The any-prompt engine auto-selects a corruption strategy for any domain (credit, medical, HR, legal, factual), each backed by a published method.

```python
from transformer_lens import HookedTransformer
from glassbox import GlassboxV2

model = HookedTransformer.from_pretrained("gpt2")   # or your fine-tuned model
gb    = GlassboxV2(model)

result = gb.analyze(
    prompt              = "Loan application. Annual income: €42,000. Decision:",
    correct             = " Approved",
    incorrect           = " Denied",
    corruption_strategy = "auto",     # auto | name_swap | random_token | antonym | semantic_negation
)

print(result["faithfulness"])          # sufficiency / comprehensiveness / F1
print(result["corruption_metadata"])   # which strategy was selected, and why (audit trail)
```

`corruption_strategy="auto"` resolves in priority order: `name_swap` → `antonym` → `semantic_negation` → `random_token` (universal fallback). You can inspect or force the choice with `auto_corrupt()`:

```python
from glassbox import auto_corrupt
corrupted, strategy, rationale = auto_corrupt(
    "Loan application for €42,000. Decision:", " Approved", " Denied"
)
```

## Step 4 — Generate the Annex IV evidence vault

```python
from glassbox import build_annex_iv_vault

vault = build_annex_iv_vault(
    gb_result   = result,
    model_name  = "your-credit-model-v3",
    provider    = "Acme Bank AG",
    output_json = "reports/annex-iv.json",
    output_html = "reports/annex-iv.html",
)

print(vault.to_dict()["compliance_summary"])
```

The vault is **deterministic** — the same input produces the same output, with a SHA-256 integrity hash so an auditor can verify the file was not altered after generation.

### How the 9 official Annex IV sections map to your model

(Official point names from Annex IV; "Glassbox source" reflects `glassbox/compliance.py`.)

| Annex IV point (official) | Glassbox source |
|---|---|
| 1. General description of the AI system | `model_info()` + provider metadata |
| 2. Detailed description of elements & development process | circuit + attribution method record |
| 3. Monitoring, functioning & control | faithfulness metrics (sufficiency, comprehensiveness, F1) |
| 4. Appropriateness of the performance metrics | benchmark + circuit accuracy *(note: code labels its Section 4 "data governance" — review this mapping with counsel)* |
| 5. Risk management system (Article 9) | `risk_register` failure-mode catalogue |
| 6. Relevant changes through the lifecycle | `CircuitDiff` (cross-version circuit comparison) |
| 7. Harmonised standards applied (Article 40) | method citations (attribution patching, ACDC, DAS) |
| 8. Copy of EU Declaration of Conformity (Article 47) | **provider-signed legal reference — not auto-generated** |
| 9. Post-market monitoring plan (Article 72) | `CircuitDiff` CI gate |

## Step 5 — Audit large models (7B → 70B+)

Attribution patching needs 3 forward passes with cached activations — infeasible for 70B models at full precision. Use the drop-in memory-managed path:

```python
from glassbox import estimate_memory, analyze_large
import torch

# Plan before you load
est = estimate_memory(n_layers=80, d_model=8192, seq_len=256, dtype=torch.bfloat16)
print(est.recommend_strategy, est.warnings)

# Audit — auto-selects standard / checkpoint / checkpoint+offload
result = analyze_large(
    gb, prompt, correct=" Approved", incorrect=" Denied",
    strategy="auto", dtype=torch.bfloat16,
)
```

Or from the CLI:

```bash
glassbox-ai estimate-memory --n-layers 80 --d-model 8192 --seq-len 256 --dtype bfloat16
```

> **Honesty note for your own claims:** the large-model path is implemented and unit-tested, with an end-to-end live verification runbook in `docs/LARGE_MODEL_VERIFICATION.md`. State "supported, verification in progress" until you have published a live run on a real ≥7B model. Do not claim "tested on 70B" without the receipt.

## Step 6 — Read the result correctly (the r = 0.009 finding)

The single most important interpretive point for an auditor: **model confidence is not a proxy for circuit faithfulness.** Across the benchmark, the correlation between the two is **r = 0.009** — effectively zero. A model can be highly confident on an output whose internal circuit poorly explains it. This is a *finding*, not a bug: it means a compliance reviewer cannot rely on confidence scores as evidence of correct internal reasoning, which is precisely why circuit-level evidence is needed.

## Step 7 — Put it in CI (the part that makes it stick)

Add a compliance gate so every model change re-generates and diffs the evidence — this is the **Article 72** post-market monitoring story, and operationally it's what keeps the documentation current instead of stale-at-launch.

```yaml
# .github/workflows/compliance.yml  (sketch)
- run: glassbox-ai analyze --prompt "..." --correct " Approved" --incorrect " Denied"
- run: python scripts/docker_scan.py --model your-model --out annex-iv.pdf
- uses: actions/upload-artifact@v4
  with: { name: annex-iv-report, path: annex-iv.pdf }
```

For a fully standalone, Python-free run, the Docker scanner produces the report in one command:

```bash
docker build -f scan.Dockerfile -t glassbox-scan .
docker run --rm -v "$PWD/out:/out" glassbox-scan --model gpt2 --out /out/annex-iv.pdf
```

---

## At a glance

```
pip install  →  classify (Art. 6 + Annex III)  →  gb.analyze(corruption="auto")
            →  build_annex_iv_vault()  →  [large model? analyze_large()]
            →  read r=0.009 caveat  →  wire into CI (Art. 72)
```

One function call. All nine Annex IV sections (eight from the model, one provider-signed). Under two seconds. Deterministic, hashed, air-gapped.

*Glassbox AI v4.3.0 · `glassbox-mech-interp` (MIT) · arXiv:2603.09988 · Not legal advice.*
