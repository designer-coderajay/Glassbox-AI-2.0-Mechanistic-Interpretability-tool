# Competitive & Literature Scan

*Compiled 2026-06-13. This addresses ROADMAP_V5_FOUNDATIONS.md Part 9.3 ("a current
literature & competitor scan ... a prerequisite for Phase C commitments"). Figures
are cited to public sources and labeled where reported rather than verified. Where a
claim could not be confirmed from a primary source it is marked. This scan was
assembled with live web search to correct for a model knowledge cutoff of May 2025.*

---

## 1. The one-line positioning

Glassbox sits in the gap between two markets that do not currently overlap:

- **Interpretability research/platforms** (Anthropic, Goodfire, EleutherAI, academia)
  push the *capability* frontier — tracing how models compute. None ship an EU AI Act
  compliance product.
- **EU AI Act compliance / GRC tooling** manages the *documentation workflow* — risk
  registers, model inventories, "AI bills of materials." None measure whether a model's
  explanation is causally true.

Glassbox's wedge is the intersection neither side occupies: **open-source, measured
faithfulness, emitted as Annex IV technical documentation.** The defensible claim is not
"best interpretability" (Goodfire outspends the field) — it is "the only tool that
refuses to put an explanation in a regulatory file without measuring whether it is true."

---

## 2. Interpretability research & platforms

| Player | What they ship | Scale (reported) | Overlap with Glassbox | Threat |
|---|---|---|---|---|
| **Goodfire** | Ember — hosted mechanistic-interpretability API (Llama 3.3 70B); CLT-based attribution graphs; "Silico" platform | ~$209M raised; $1.25B valuation (Series B, Feb 2026; Anthropic backed the Series A) | Same science (circuits, attribution); different buyer (AI labs, alignment), closed, no compliance output | **High (latent)** — could move down-market into compliance |
| **Anthropic** | Attribution graphs (Claude 3.5 Haiku, Apr 2025); open-sourced **Circuit Tracer** library | Frontier lab | Method overlap; not a product for third-party model audits | Medium — sets the methodology bar; unlikely to ship a compliance SKU |
| **EleutherAI** | **Attribute** — open attribution-graph library supporting cross-layer transcoders | Non-profit research | Open-source method overlap (feature-level) | Low — research, not product; potential upstream dependency |
| **Neuronpedia** | Hosts the circuits-research landscape / shared tooling | Community infra | Visualization/hosting overlap | Low |
| **Academia (MIB, 2025)** | Mechanistic Interpretability Benchmark — circuit-localization & causal-variable tracks; metrics CPR, CMD, interchange-intervention accuracy | Standard-setting | Defines how circuit faithfulness is scored | Low — adopt their metrics, cite them |

**Read:** Goodfire is the only well-capitalized commercial player in the same science,
and it is an order of magnitude better funded. Glassbox does **not** win head-to-head on
raw interpretability capability. It wins on (a) being open-source, (b) being
compliance-native (Annex IV out of the box), and (c) honesty-by-construction (evidence
tiers, refusal to fabricate). Treat a Goodfire move into EU compliance as the primary
strategic risk.

---

## 3. EU AI Act compliance / GRC tooling

The compliance-software market is real and growing into the **2 Aug 2026** high-risk
enforcement date (Art. 11 technical documentation, Art. 9 risk management, Art. 12
logging). Tools in this category — GRC platforms and "AI bill of materials" vendors —
generate and track the *technical file* as a workflow artifact.

**Critical gap they all share:** they document *that* a system has explainability
measures; none *measure faithfulness*. An Annex IV §4 entry that says "SHAP values were
computed" satisfies a checkbox, not the question of whether the explanation reflects the
model's actual computation. Glassbox's r ≈ 0.009 finding (confidence is orthogonal to
faithfulness) is the wedge against this entire category: a faithfulness number is a
property they structurally cannot produce without circuit access.

**Risk from this side:** a GRC vendor bolts on shallow black-box "explainability"
(SHAP/LIME) and markets "Annex IV explainability coverage." Rebuttal is the faithfulness
argument, but it requires the market to understand the difference — a sales/education
burden, not a technical one.

---

## 4. Methodology landscape (what to adopt, what to cite)

- **ERASER** (DeYoung et al. 2020) — origin of sufficiency/comprehensiveness, adapted by
  Glassbox to circuit level. Known limitation: no random baseline, so it cannot detect
  *anti-faithfulness*. Glassbox's decision benchmark adds the attribution-concentration
  comparator to address this.
- **Normalized AOPC** (ACL 2025) — fixes faithfulness metrics that conflate deletion with
  out-of-distribution degradation. Worth tracking for the next metric revision.
- **Cross-layer transcoders (CLT) / attribution graphs** (Anthropic; Goodfire; EleutherAI)
  — the field's move from head-level to **feature-level** units. This is the most
  important methodological shift for Glassbox: head-level circuits may be superseded
  where head faithfulness is weak. `glassbox/sae_attribution.py` is the seed; feature-level
  attribution at frontier scale remains open research (Roadmap Phase C / 9.2).
- **MIB metrics** (CPR, CMD, interchange-intervention accuracy) — align terminology and,
  where feasible, report against them for external comparability.

---

## 5. Honest gaps in this scan

- Funding/valuation figures are as reported by trade press, not audited.
- The compliance-vendor list was characterized by category, not feature-audited
  per-vendor — a per-competitor teardown is needed before any "we beat X" claim.
- Feature-level interpretability (SAEs/CLTs) is moving fast; this scan is a snapshot, not
  a moat. Re-run before any Phase C go/no-go.
- No primary-source confirmation that any GRC vendor has shipped *measured* faithfulness —
  treated as "none observed," not "none exists."

---

## 6. Sources

- [Goodfire — $150M Series B, $1.25B valuation (SiliconANGLE, Feb 2026)](https://siliconangle.com/2026/02/05/goodfire-raises-150m-funding-enhance-ai-interpretability-platform/)
- [Goodfire — $50M Series A (PRNewswire, Apr 2025)](https://www.prnewswire.com/news-releases/goodfire-raises-50m-series-a-to-advance-ai-interpretability-research-302431030.html)
- [Goodfire Ember announcement](https://www.goodfire.ai/blog/announcing-goodfire-ember)
- [Anthropic attribution graphs (MarkTechPost, Apr 2025)](https://www.marktechpost.com/2025/04/06/this-ai-paper-from-anthropic-introduces-attribution-graphs-a-new-interpretability-method-to-trace-internal-reasoning-in-claude-3-5-haiku/)
- [Circuits research landscape — Neuronpedia (Aug 2025)](https://www.neuronpedia.org/graph/info)
- [Normalized AOPC: fixing misleading faithfulness metrics (ACL 2025)](https://aclanthology.org/2025.acl-long.86.pdf)
- [EU AI Act Annex IV (Article 11(1))](https://artificialintelligenceact.eu/annex/4/)
- [Best EU AI Act compliance tools 2026 (Prediction Guard)](https://predictionguard.com/blog/best-eu-ai-act-compliance-tools-for-enterprise-ai-programs-in-2026)
