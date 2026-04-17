# Patent Notice

## Glassbox AI — Patent-Pending Methods

**Copyright (C) 2026 Ajay Pravin Mahale**

---

### Patent-Pending Disclosure

The following methods and systems implemented in Glassbox AI are the subject
of a provisional patent application and/or are candidates for patent
protection. This notice is provided pursuant to 35 U.S.C. § 287 and
equivalent provisions under applicable international law.

---

### Method 1: Attribution-Patching-to-Regulatory-Document Pipeline

**Title:** System and Method for Generating AI Regulatory Compliance
Documentation via Causal Attribution Analysis of Neural Network Circuits

**Status:** Provisional patent application — patent pending

**Description:**

A computer-implemented method for automatically generating structured
regulatory compliance documentation (specifically EU AI Act Annex IV
technical documentation) from causal attribution analysis of transformer
neural network models, comprising:

1. Computing per-attention-head causal attribution scores using first-order
   Taylor approximation of activation patching (O(3) forward passes);
2. Identifying the minimum faithful circuit of attention heads responsible
   for a model prediction using threshold-based greedy selection;
3. Measuring exact causal sufficiency and comprehensiveness of identified
   circuits via positive and negative activation ablation;
4. Generating bootstrap confidence intervals over a distribution of prompts;
5. Mapping circuit attribution metadata to specific EU AI Act article
   requirements (Articles 9, 11, 13–14, 15, 72, Annex IV §1–§9);
6. Producing structured regulatory documentation with article-level citations
   and export to standardized formats (Markdown, PDF, HuggingFace model card).

---

### Method 2: CircuitDiff — Mechanistic Model Version Comparison

**Title:** System and Method for Mechanistic Differencing of Neural Network
Model Versions for Post-Market Monitoring Compliance

**Status:** Provisional patent application — patent pending

**Description:**

A computer-implemented method for comparing two versions or checkpoints of
a transformer neural network model at the mechanistic circuit level,
comprising:

1. Computing attribution circuits for both model versions on a shared
   prompt distribution;
2. Computing Jaccard circuit stability score between version circuits;
3. Computing normalized attribution drift (L1-norm of per-head attribution
   delta, normalized by baseline magnitude);
4. Generating a structured change summary categorizing: circuit membership
   changes (head additions/removals), attribution magnitude shifts, and
   behavioral correlation with held-out benchmarks;
5. Mapping mechanistic changes to EU AI Act Article 72 post-market
   monitoring documentation requirements;
6. Providing threshold-based alerts when circuit stability falls below
   configurable thresholds indicating significant behavioral change.

---

### Additional Potentially Patentable Methods

The following methods implemented in Glassbox AI are under evaluation for
patent protection:

- **Bootstrap-calibrated faithfulness metric certification**: Method for
  establishing statistical confidence bounds on mechanistic circuit
  faithfulness metrics and mapping these to regulatory evidence standards.

- **Tamper-evident AI audit chain**: Hash-chained audit log structure
  (SHA-256 linking) for AI governance records meeting Article 12 logging
  requirements.

- **Multi-prompt circuit stability suite**: Method for measuring circuit
  membership variance across paraphrases of equivalent prompts to establish
  robustness of mechanistic explanations.

---

### Important Notice

This PATENTS.md file is informational only. It does not constitute legal
advice. Patent applications are subject to examination and may or may not
ultimately issue as granted patents. The presence of any method in this
file does not preclude others from implementing functionally equivalent
methods through alternative technical approaches.

For licensing inquiries related to patented or patent-pending methods,
contact: **mahale.ajay01@gmail.com**

---

*This file was last updated: 2026-03-20*
*Glassbox AI version: 4.2.6*
