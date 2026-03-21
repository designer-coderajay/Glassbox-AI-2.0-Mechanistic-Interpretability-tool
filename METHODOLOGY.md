# Glassbox AI — Proprietary Methodology

**Author:** Ajay Pravin Mahale (`mahale.ajay01@gmail.com`)
**Version documented:** 3.4.0
**Date:** 2026-03-21
**Status:** Proprietary — All rights reserved

> This document describes the intellectual property embodied in the Glassbox AI
> system: the specific architectural decisions, mathematical formulations, and
> legal-technical translation logic that constitute the novel contribution of
> this work. It is intended to support due diligence, patent prosecution, and
> regulatory review.

---

## 1. The Core Innovation: Regulatory-Native Circuit Analysis

Most mechanistic interpretability tools stop at the mathematical layer —
they identify circuits, compute faithfulness scores, and visualise attention
patterns. Glassbox makes a different architectural choice: **every
mathematical operation is co-designed with a specific provision of Regulation
(EU) 2024/1689 (EU AI Act)**.

This is not post-hoc annotation. The compliance mapping is load-bearing. It
determines which metrics are computed, at what thresholds, with what
terminology, and in which section of the Annex IV output document. The result
is a system where a single `gb.analyze()` call produces output that is
simultaneously:

1. A mathematically rigorous mechanistic interpretability result, and
2. A legally structured Annex IV technical documentation draft.

No other open-source tool, as of March 2026, implements this co-design.

---

## 2. Proprietary Mathematical Contributions

### 2.1 Taylor-Approximated Attribution Patching (O(3) passes)

**Prior art.** Conmy et al. (2023) introduced Automated Circuit Discovery
(ACDC), which requires O(E) forward passes where E is the number of edges in
the computation graph — scaling to thousands of passes on large models.

**Glassbox contribution.** We implement a first-order Taylor approximation to
the patching objective that reduces the full attribution pass to exactly 3
forward passes regardless of circuit size:

```
ΔL(e) ≈ (∂L/∂h_e) · (h_e^clean − h_e^corrupted)
```

Where:
- `ΔL(e)` is the estimated effect of edge `e` on the logit difference
- `∂L/∂h_e` is the gradient of the logit difference with respect to
  the head's output (computed once via backward pass)
- `h_e^clean − h_e^corrupted` is the activation difference under
  clean vs. mean-ablated inputs (computed in 2 forward passes)

This formulation enables circuit discovery on consumer hardware in seconds
rather than minutes. The approximation error is disclosed via the
`suff_is_approx` flag in every result.

**Regulatory mapping.** Article 15(1) requires that accuracy and performance
be declared with their metrics. The Taylor approximation disclosure
(`suff_is_approx: True/False`) directly satisfies this requirement.

### 2.2 Minimum Faithful Circuit (MFC) Algorithm

**Definition.** Given a set of candidate heads ranked by `|ΔL(e)|`, the MFC
is the smallest subset `C ⊆ H` such that:

```
sufficiency(C) ≥ τ_s  AND  comprehensiveness(C) ≥ τ_c
```

Where `τ_s = 0.70` and `τ_c = 0.60` are empirically derived thresholds
calibrated against the IOI benchmark (Wang et al. 2022).

**Glassbox contribution.** We implement a greedy forward selection with early
stopping that finds the MFC in O(|C|) additional passes after the initial
attribution ranking, rather than the exponential search that exact methods
require. The algorithm is:

1. Rank all heads by `|ΔL(e)|` descending.
2. Greedily add heads while sufficiency is below `τ_s`.
3. Prune heads whose removal does not reduce sufficiency below `τ_s`.
4. Verify comprehensiveness; if below `τ_c`, re-add the most recently pruned head.

**Regulatory mapping.** The MFC is the "documented computational structure"
required by Annex IV §2 and Article 11. Its size (n_heads) is a direct proxy
for model interpretability per Article 13(1).

### 2.3 Faithfulness F1 Score as a Compliance Gate

**Formulation.**
```
F1_faith = 2 · (sufficiency · comprehensiveness) / (sufficiency + comprehensiveness)
```

This is the standard harmonic mean applied to the faithfulness pair. The
insight is that neither metric alone is sufficient for compliance purposes:

- High sufficiency with low comprehensiveness means the circuit is sufficient
  but not necessary — the model has backup mechanisms, increasing risk of
  unpredictable behaviour under distribution shift.
- High comprehensiveness with low sufficiency means the circuit is necessary
  but not alone sufficient — the explanation is incomplete.

**Regulatory mapping.** We define a compliance gate at `F1_faith ≥ 0.65`,
derived from the Article 15(1) requirement for "appropriate level of accuracy."
This threshold appears in the CI/CD pipeline as `GLASSBOX_MIN_SUFFICIENCY`
and in the `AnnexIVEvidenceVault` as the §4 risk management threshold.

### 2.4 Exact Sufficiency via Positive Ablation

For high-stakes deployments, Glassbox supports exact sufficiency computation:

```
suff_exact = σ(logit_diff(circuit_only)) / σ(logit_diff(full_model))
```

Where "circuit_only" means all non-circuit heads are mean-ablated (corrupted),
and the circuit heads run on clean activations. This is the proper causal
intervention, not the Taylor approximation.

The `bootstrap_metrics(exact_suff=True)` method uses this formulation and
achieves ~100% sufficiency on the IOI benchmark with GPT-2 small, validating
the circuit identification methodology.

---

## 3. Proprietary Legal-Technical Translation Layer

This is the core IP. The following mappings are the proprietary contribution
of Glassbox AI and are not derived from any academic paper.

### 3.1 Faithfulness → Article 15(1)

| Metric | Threshold | Article | Compliance Meaning |
|--------|-----------|---------|-------------------|
| Sufficiency | ≥ 0.70 | Art. 15(1) | Circuit retains ≥70% of model's predictive accuracy |
| Comprehensiveness | ≥ 0.60 | Art. 15(1) | Ablating circuit causes ≥60% accuracy drop |
| F1_faith | ≥ 0.65 | Art. 15(1) | Balanced explainability gate for compliance sign-off |

The grade mapping (Excellent/Good/Marginal/Poor → A/B/C/D) provides a
single-letter compliance status analogous to credit ratings, enabling
non-technical stakeholders (compliance officers, notified bodies) to assess
the system's explainability standing.

### 3.2 Circuit Size → Article 13(1)

A smaller minimum faithful circuit is not just a mathematical curiosity —
it is a transparency indicator with a direct Article 13(1) interpretation.
Article 13(1) requires that AI systems be designed to allow informed use.
A 3-head circuit is more interpretable than a 47-head circuit; Glassbox
quantifies this explicitly and includes it in Annex IV §2 documentation.

### 3.3 Bias Signals → Article 10

| Bias Category | Protected Attribute | Article Reference |
|---------------|--------------------|--------------------|
| Gender | Sex / gender identity | Art. 10(2)(f), Art. 10(5) |
| Race/Ethnicity | Racial/ethnic origin | Art. 10(2)(f), Art. 10(5) |
| Nationality | National origin | Art. 10(2)(f) |
| Religion | Religious belief | Art. 10(2)(f), Art. 10(5) |
| Age | Age group | Art. 10(2)(f) |
| Disability | Physical/mental disability | Art. 10(2)(f), Art. 10(5) |
| Sexuality | Sexual orientation | Art. 10(2)(f), Art. 10(5) |
| Socioeconomic | Economic background | Art. 10(2)(f) |

These mappings implement Article 10(2)(f)'s requirement that providers
document "possible biases that are likely to affect health or safety or
lead to discrimination."

### 3.4 Steering Vectors → Article 9(2)(b)

A steering vector extracted via Representation Engineering (Zou et al. 2023)
constitutes a "technical measure to address risks" under Article 9(2)(b).
Glassbox is the first tool to formalise this connection: by exporting the
vector with provenance metadata (layer, method, n_contrast_pairs, timestamp)
and a quantified suppression test result, the vector becomes a documented
and auditable risk mitigation measure rather than an ad-hoc patch.

### 3.5 Multi-Agent Liability Scoring

The contamination score formula:

```
contamination(A→B) = |bias_tokens(B_output) ∩ bias_tokens(A_output)| / |bias_tokens(B_output)|
```

measures the fraction of B's bias signal that can be traced back to A's
output. This formalises a chain-of-causation argument for Article 9
system-level risk assessment in multi-agent deployments.

The responsibility score:

```
responsibility(agent_i) = (introduced_score + amplified_score · 1.5) / n_handoffs
```

weights amplification more than introduction, reflecting the legal principle
that an agent that takes a small bias and magnifies it bears greater
responsibility than an agent that originates it at a low level.

These formulas and their Article 9 mapping are original contributions of
Glassbox AI.

### 3.6 Annex IV Evidence Vault Structure

The `AnnexIVEvidenceVault` implements a specific mapping from interpretability
findings to the 7 sections of Annex IV:

| Vault Evidence Type | Annex IV Section | Articles |
|--------------------|-----------------|---------|
| faithfulness metrics | §2 (system description) | Art. 11, 15(1) |
| circuit heads | §2 (development process) | Art. 11, 13(1) |
| stability metrics | §3 (monitoring) | Art. 15(1), 72 |
| steering vectors | §4 (risk management) | Art. 9(2)(b), 9(5) |
| SAE feature activations | §4 (risk management) | Art. 10(2)(f), 10(5) |
| multi-agent liability | §4 (risk management) | Art. 9, 10(2)(f) |
| technical references | §6 (standards applied) | Art. 11 |
| conformity declaration | §7 (declaration of conformity) | Art. 11 |

The logic that decides which evidence goes into which section and which
articles are cited is the original proprietary contribution of Glassbox AI.

---

## 4. Architectural Decisions and Their Justification

### 4.1 No LLM Dependency in the Compliance Layer

Every component in the compliance pipeline (NaturalLanguageExplainer,
MultiAgentAudit bias detection, AnnexIVEvidenceVault) is rule-based and
deterministic. This is a deliberate architectural choice:

- **Auditability.** A rule-based system produces the same output for the
  same input every time. An LLM-based system does not. Regulators require
  reproducibility.
- **Circular trust.** Using an LLM to explain another LLM's compliance
  creates a trust dependency that cannot be independently verified.
- **Air-gap compatibility.** Enterprise deployments in regulated industries
  (banking, healthcare) often prohibit outbound API calls. A rule-based
  compliance layer works in fully air-gapped environments.

### 4.2 BSL 1.1 Dual Licensing

The Business Source License 1.1 with a 10-year change date (2036-03-21) is
a strategic choice that creates a commercial moat while maintaining
open-source visibility. Any company wishing to use Glassbox in a commercial
product must either:

1. Obtain a commercial licence from Ajay Pravin Mahale, or
2. Wait until 2036-03-21, when the code converts to Apache 2.0.

This mirrors the licensing strategies of CockroachDB, HashiCorp Vault, and
Elastic — all of which achieved multi-billion dollar valuations under BSL.

### 4.3 TransformerLens as the Foundation

Using TransformerLens (Nanda et al. 2022) rather than raw PyTorch gives
Glassbox several properties that are important for the compliance use case:

- **Reproducibility.** TransformerLens has stable, versioned implementations
  of transformer components. Raw PyTorch implementations vary between users.
- **Hook-based intervention.** The `run_with_cache` and `run_with_hooks`
  APIs make it trivial to implement the ablation-based interventions that
  faithfulness metrics require.
- **Model coverage.** TransformerLens supports 50+ model architectures,
  giving Glassbox broad coverage without maintaining custom code per model.

---

## 5. Academic References and Relationship to Prior Art

Glassbox builds on the following academic work. All references are cited
to establish what is prior art and what is original to Glassbox.

| Reference | What Glassbox uses | What Glassbox adds |
|-----------|-------------------|-------------------|
| Wang et al. 2022 — IOI task | Benchmark prompt, circuit structure | Compliance mapping, thresholds |
| Conmy et al. 2023 — ACDC | Patching objective formulation | Taylor approx (O(3) passes), MFC algorithm |
| Syed et al. 2024 — EAP | Edge Attribution Patching formula | Regulatory threshold mapping |
| Zou et al. 2023 — RepEng | Residual-stream mean-diff vectors | Article 9(2)(b) formalisation, provenance metadata, suppression testing |
| Li et al. 2023 — ITI | Inference-time hook injection | `test_suppression()` before/after faithfulness comparison |
| Elhage et al. 2022 — SAE | Sparse autoencoder features | `_RISK_TO_ARTICLES` legal risk mapping |
| Nanda et al. 2022 — TransformerLens | Model execution framework | — |
| Regulation (EU) 2024/1689 | Legal requirements | All technical implementations above |

The Annex IV section mapping, the threshold values, the grade system, the
contamination score formula, the responsibility score formula, the legal risk
category taxonomy, and the VaultEntry structure are original contributions
of Glassbox AI and are not derived from any academic paper.

---

## 6. Claims Summary (for Patent Prosecution Reference)

The following describe the novel concepts in Glassbox AI that may be
protectable as software patents or trade secrets. This list is provided for
reference only and does not constitute legal advice.

1. **Method for generating regulatory compliance documentation from
   mechanistic interpretability analysis** — mapping circuit faithfulness
   metrics to specific provisions of the EU AI Act in a structured,
   machine-readable format.

2. **Multi-agent liability scoring system** — contamination score and
   responsibility score formulas for attributing bias propagation across
   agent handoff chains.

3. **Steering vector compliance formalisation** — methodology for converting
   a representation-engineering vector into a documented, auditable
   Article 9(2)(b) risk mitigation measure.

4. **Evidence vault architecture** — mapping interpretability findings to
   Annex IV sections and EU AI Act articles in a structured evidence package
   suitable for regulatory submission.

5. **Faithfulness F1 compliance gate** — use of the harmonic mean of
   sufficiency and comprehensiveness as a single-number compliance threshold
   with regulatory thresholds derived from Article 15(1).

---

## 7. Change History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-21 | Initial documentation for v3.4.0 |

---

*This document is proprietary to Ajay Pravin Mahale. Unauthorised reproduction
or distribution is prohibited. For licensing enquiries: mahale.ajay01@gmail.com*
