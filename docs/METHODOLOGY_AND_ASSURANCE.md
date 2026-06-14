# Glassbox — Methodology & Assurance

*What the numbers mean, what is mathematically guaranteed, what is measured, what is
only a hypothesis, and how it maps to the EU AI Act. Written to be handed to a
technical buyer, an auditor, or a regulator.*

Version aligned with `glassbox-mech-interp` 4.5.0. Paper: arXiv 2603.09988 (Mahale, 2026).

---

## 0. One-paragraph summary

Glassbox takes a model's **decision** (a choice between two outcomes), finds the
small set of attention heads that causally drive it, and then **measures** how
well that set explains the decision. It does not ask you to trust the
explanation — it tests it by re-running the model with parts ablated, and reports
the result, including when the explanation is poor. The same evidence is written
into an EU AI Act Annex IV technical-documentation file. The headline empirical
finding that motivates all of this: a model's **confidence is essentially
uncorrelated with the faithfulness of its explanation (r = 0.009)**, so
faithfulness must be measured, not assumed.

---

## 1. The three layers of assurance — and which is which

Be precise about what is *proven*, what is *measured*, and what is a *hypothesis*.
Only Layer 2 is a measurement; conflating the layers is how interpretability tools
mislead.

### Layer 1 — the circuit (an approximation, not a proof)

The exact causal effect of a component (attention head) on a decision is the
change in the logit-difference when that component's activation is replaced
("patched") with the activation from a contrasting input. Computing that exactly
for every head costs one forward pass per head.

Glassbox uses **attribution patching** — a first-order Taylor approximation:

```
effect(head) ≈ ( a_clean(head) − a_corrupt(head) ) · ∂(logit_diff)/∂a(head)
```

i.e. *activation difference × gradient*, obtained in **3 forward/backward passes
total** instead of one-per-head. This is the method of Nanda (2023) and Syed,
Rager & Conmy (2023). It is fast but **linear**: its error grows where the loss
surface is nonlinear between the clean and corrupted activations. Glassbox
therefore also provides **second-order (Hessian) error bounds** to *bound* that
approximation error rather than ignore it.

**Status:** the circuit is a ranked **hypothesis** with a known error term — not a
theorem about the model's "true reason."

### Layer 2 — faithfulness (this is the measured assurance)

The circuit hypothesis is then **tested empirically** by re-running the model:

- **Sufficiency** — keep only the circuit, ablate the rest. Does the decision
  survive? High = the circuit alone reproduces the decision.
- **Comprehensiveness** — remove the circuit, keep the rest. Does the decision
  break? High = the circuit is necessary.
- **F1** — harmonic mean of the two.

These are the **ERASER faithfulness metrics** (DeYoung et al., 2020). They are
*computed by ablation and re-running the model* — reproducible and falsifiable,
not asserted. **When they are low, the tool says the explanation is incomplete**
(see the worked example below, where gpt2 scores sufficiency 0.48 and is graded C
/ NON-COMPLIANT). That self-honesty is the core trust mechanism.

### Layer 3 — causal-abstraction certificate (the strongest, narrowest claim)

The strongest statement is not "these heads correlate with the decision" but
"this circuit *implements* the declared decision rule." That is tested with
**interchange interventions** and scored as **interchange-intervention accuracy
(IIA)** — the causal-abstraction line of work (Geiger et al.). Glassbox computes
the certificate; generating the interchange trials at production scale is open
research, so it is offered as the top **evidence tier (A)**, not claimed by
default.

---

## 2. What is NOT guaranteed (state this to every buyer)

- It does **not** prove the circuit is the *unique* cause. Models have backup
  mechanisms — multiple sufficient circuits (the IOI "backup name-mover"
  phenomenon is the textbook case).
- Faithfulness is an empirical claim about **the inputs you tested**, not a
  theorem over all inputs. This is why Glassbox reports **bootstrap confidence
  intervals** over a distribution of prompts rather than a single number.
- A metric like "sufficiency 0.48" is a **defined quantity**, not "48% true."
- **High faithfulness ≠ a good or fair decision.** Faithfulness measures whether
  the explanation matches the model's actual computation — not whether the
  decision itself is correct, lawful, or unbiased. Keep these separate.

---

## 3. Why you can trust it (structural, not "trust the vendor")

1. Every number is **computed by re-running the model** — reproducible by the auditor.
2. The methods are from the **peer-reviewed literature**, not invented:
   ERASER faithfulness (DeYoung et al., 2020); attribution patching (Nanda, 2023;
   Syed et al., 2023); ACDC baseline (Conmy et al., NeurIPS 2023); causal
   abstraction / IIA (Geiger et al.).
3. A **conformance suite** gates the implementation on each model (determinism,
   patch-identity) before any number is believed.
4. An **evidence-tier label (A–D)** forces honest scoping of every result.
5. It is **open source** — the regulator can run it themselves.
6. It is honest about failure: it returns NON-COMPLIANT when a model's decision
   cannot be faithfully explained, instead of fabricating a clean story.

*Empirical basis for the whole approach:* confidence–faithfulness correlation
**r = 0.009** — a model's own confidence tells you nothing about whether its
explanation is real.

---

## 4. Scope and limits (the honest boundary)

- **White-box only.** The method needs weights, activations, and gradients. It
  works on **open-weight models you can load** (GPT-2 family verified;
  Llama / Mistral / Phi via adapters). It does **not** work through a closed API
  (GPT-4, Claude, Gemini) — you cannot extract activations or gradients from an
  API, so those get only the weaker **black-box behavioral tier**, not circuits.
- **Decisions, not free-form chat.** The audit needs a decision framed as a
  contrast between outcomes (approve/deny, hire/reject, flag/clear). Free-form
  generation has no contrast to attribute against.
- **Offline / sampled, not per-request.** At ~3 passes + gradients per analysis,
  this audits representative decisions to produce documentation and periodic
  monitoring — it is not a real-time monitor on billions of production requests.
- **Verified scope today** (see `VALIDATION_LOG.md`): conformance gate PASS across
  **6 architecture families** (GPT-2, Pythia/GPT-NeoX, GPT-Neo, OPT, Qwen2-GQA,
  Mistral-GQA) **up to 12B**; and **faithfulness (suff + comp, both rigor-controlled)
  validated 82M → 12B across 9 model families** including every major GQA family
  (Llama-3, Mistral, Gemma-2, Qwen2/2.5, Yi, Phi-3), via scale-aware circuit
  selection — comprehensiveness is specificity-checked against a random same-size
  circuit at every scale (comp ~1.0 vs random 0.0–0.29).
  **Not yet validated:** 13B–200B (needs a multi-GPU cluster + the unproven
  distributed backend; a single 80 GB GPU tops out ~13–30B for gradient-based
  attribution), production throughput, and closed APIs.

---

## 5. EU AI Act mapping — and the honest legal limit

Glassbox produces **evidence and documentation** toward specific obligations:

| Obligation | What Glassbox contributes |
|---|---|
| **Art. 11 + Annex IV** (technical documentation) | Auto-fills 8 of 9 sections from the circuit + faithfulness evidence; §8 (Declaration of Conformity) is left for human sign-off. |
| **Art. 13** (transparency) | The faithfulness-graded circuit is measured transparency evidence. |
| **Art. 15** (accuracy, robustness) | Robustness / multi-corruption tests + held-out validation. |
| **Art. 9 / Art. 72** (risk management, post-market monitoring) | Risk register + CUSUM drift detector. |
| NIST AI RMF / ISO 42001 | Same evidence cross-walked to those frameworks (theme level). |

**The legal limit — do not overstate.** The EU AI Act does **not** mandate
mechanistic interpretability and defines **no faithfulness threshold**. The
`overall_status: NON-COMPLIANT` produced by the tool is **Glassbox's own internal
gate, not a legal ruling** — never present it as a regulator's verdict. Conformity
is determined by the provider's conformity assessment (self-assessment or a
notified body) against harmonised standards that are still being written.
Glassbox's output is a strong **input** to that process; it does not by itself
make anyone compliant, and it is **not legal advice or certification**.

Enforcement dates: high-risk obligations apply from **2 August 2026** under
current law; the **Digital Omnibus** (provisionally agreed 7 May 2026, pending
formal adoption) would defer the Annex III timeline to **2 December 2027** — plan
against both until it is adopted. Documentation non-compliance penalty: up to
**€15M or 3% of global annual turnover** (Art. 99(4)).

---

## 6. Worked example (reproduces the honesty)

`examples/loan_decision_audit.py` runs a loan-approval-style decision on gpt2 end
to end. gpt2 cannot underwrite, so it scores **sufficiency ≈ 0.48, F1 ≈ 0.58,
Grade C**, and the report is marked **NON-COMPLIANT** — the correct outcome for a
model with no real decision circuit. The plumbing (real model → real circuit →
measured faithfulness → Annex IV file) is what is real; point it at a fine-tuned
decision model for a meaningful audit.

---

*All figures are from the reconciled benchmark set (`BENCHMARKS.md`) and the test
suite (932 passing, 71% coverage as of 4.5.0). Regulatory statements reflect a
reading of the EU AI Act and the pending Digital Omnibus as of June 2026 and
should be confirmed with qualified counsel. Citations name the established methods
this tool builds on; verify them against the primary sources before relying on
them in a formal filing.*
