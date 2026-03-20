"""
glassbox.circuit_diff
=====================
Circuit-level diff between two model versions or checkpoints.

Answers: "When the model was fine-tuned / RLHF'd / updated, which attention
heads changed their causal role in this task?"

This is the mechanistic interpretability equivalent of a git diff — not a
diff of weights (that's just subtraction), but a diff of which heads are
*causally responsible* for a prediction, and by how much.

Primary use case
-----------------
  A compliance team has deployed GPT-X v1 and wants to know if the v2
  update changed the model's reasoning circuit for a regulated decision
  (e.g., loan approval). Regulatory obligation: Article 72 (post-market
  monitoring) and Annex IV Section 6 (lifecycle changes).

  With CircuitDiff they can:
    1. Identify which heads entered or left the circuit.
    2. Quantify how much each shared head's attribution weight changed.
    3. Detect if a new head type (e.g. negative mover) appeared.
    4. Get a numeric stability score (0–1) for the circuit across versions.

Algorithm
----------
  1. Run ``GlassboxV2.analyze()`` on model A and model B with identical
     prompt/correct/incorrect triple.
  2. Compute set operations:
       added   = circuit_B − circuit_A
       removed = circuit_A − circuit_B
       shared  = circuit_A ∩ circuit_B
  3. For shared heads, compute attribution delta:
       delta(h) = attr_B(h) / |LD_B| − attr_A(h) / |LD_A|   (normalised)
  4. Stability score = Jaccard similarity:
       J = |shared| / |circuit_A ∪ circuit_B|   ∈ [0, 1]
     J = 1.0 → circuits are identical; J = 0.0 → no overlap at all.
  5. Attribution drift = mean |delta(h)| for shared heads.

LEGAL NOTICE — DOCUMENTATION AID ONLY
---------------------------------------
This module is a research tool for post-market monitoring documentation.
Its outputs do not constitute a conformity assessment, legal advice, or
regulatory certification under Regulation (EU) 2024/1689. Consult qualified
legal and technical counsel before relying on circuit diff results for
regulatory decisions.

Regulatory References (informational only)
------------------------------------------
Regulation (EU) 2024/1689 — EU AI Act:
  Article 72   — Post-market monitoring obligations
  Annex IV, Section 6 — Lifecycle and version change documentation
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CircuitDiff", "CircuitDiffResult"]


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CircuitDiffResult:
    """
    Result of a circuit-level diff between two model versions.

    Attributes
    ----------
    prompt          : The prompt used for both analyses.
    correct         : The correct (target) token.
    incorrect       : The distractor token.
    model_a_label   : Descriptive label for model A (e.g. "gpt2-base").
    model_b_label   : Descriptive label for model B (e.g. "gpt2-finetuned").

    circuit_a       : List of (layer, head) tuples in model A's circuit.
    circuit_b       : List of (layer, head) tuples in model B's circuit.
    added_heads     : Heads in circuit_b but not circuit_a  (new after update).
    removed_heads   : Heads in circuit_a but not circuit_b  (dropped after update).
    shared_heads    : Heads present in both circuits.
    attribution_delta : For shared heads: normalised attr_B − attr_A.
                        Keys are "(layer, head)" strings. Positive → more important
                        in B; negative → less important in B.

    stability_score : Jaccard similarity of the two circuits ∈ [0, 1].
                      1.0 = identical; 0.0 = no overlap.
    attribution_drift : Mean |delta| across shared heads (0.0 if no shared heads).
    faithfulness_a  : Faithfulness dict from model A's analyze() result.
    faithfulness_b  : Faithfulness dict from model B's analyze() result.

    change_summary  : Human-readable summary string.
    """
    prompt:            str
    correct:           str
    incorrect:         str
    model_a_label:     str
    model_b_label:     str

    circuit_a:         List[Tuple[int, int]]
    circuit_b:         List[Tuple[int, int]]
    added_heads:       List[Tuple[int, int]]
    removed_heads:     List[Tuple[int, int]]
    shared_heads:      List[Tuple[int, int]]
    attribution_delta: Dict[str, float]

    stability_score:   float    # Jaccard
    attribution_drift: float    # mean |delta| on shared heads

    faithfulness_a:    Dict
    faithfulness_b:    Dict

    change_summary:    str = field(default="", init=False)

    def __post_init__(self) -> None:
        n_added   = len(self.added_heads)
        n_removed = len(self.removed_heads)
        n_shared  = len(self.shared_heads)

        if self.stability_score >= 0.9:
            verdict = "STABLE — circuits are nearly identical"
        elif self.stability_score >= 0.6:
            verdict = "MODERATE DRIFT — some heads changed"
        elif self.stability_score >= 0.3:
            verdict = "SIGNIFICANT DRIFT — majority of circuit changed"
        else:
            verdict = "MAJOR CHANGE — circuits have minimal overlap"

        self.change_summary = (
            f"{verdict}. "
            f"Jaccard={self.stability_score:.2f}. "
            f"{n_shared} shared heads, {n_added} added, {n_removed} removed. "
            f"Attr drift={self.attribution_drift:.3f}. "
            f"F1: {self.faithfulness_a.get('f1', 0):.2f} → "
            f"{self.faithfulness_b.get('f1', 0):.2f}."
        )

    def to_dict(self) -> Dict:
        """Serialisable dict for JSON export / audit log."""
        return {
            "prompt":            self.prompt,
            "correct":           self.correct,
            "incorrect":         self.incorrect,
            "model_a_label":     self.model_a_label,
            "model_b_label":     self.model_b_label,
            "circuit_a":         [list(h) for h in self.circuit_a],
            "circuit_b":         [list(h) for h in self.circuit_b],
            "added_heads":       [list(h) for h in self.added_heads],
            "removed_heads":     [list(h) for h in self.removed_heads],
            "shared_heads":      [list(h) for h in self.shared_heads],
            "attribution_delta": self.attribution_delta,
            "stability_score":   self.stability_score,
            "attribution_drift": self.attribution_drift,
            "faithfulness_a":    self.faithfulness_a,
            "faithfulness_b":    self.faithfulness_b,
            "change_summary":    self.change_summary,
        }

    def to_markdown(self) -> str:
        """Markdown-formatted diff for PR comments and audit reports."""
        added_str   = ", ".join(f"L{l}H{h}" for l, h in sorted(self.added_heads))   or "none"
        removed_str = ", ".join(f"L{l}H{h}" for l, h in sorted(self.removed_heads)) or "none"
        shared_str  = ", ".join(f"L{l}H{h}" for l, h in sorted(self.shared_heads))  or "none"

        drift_rows = "\n".join(
            f"| L{k.split(',')[0].strip('( ')}H{k.split(',')[1].strip(') ')} "
            f"| {'+' if v >= 0 else ''}{v:+.3f} |"
            for k, v in sorted(
                self.attribution_delta.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:10]
        )

        fa = self.faithfulness_a
        fb = self.faithfulness_b

        return f"""## Circuit Diff: `{self.model_a_label}` → `{self.model_b_label}`

**{self.change_summary}**

| Metric | Model A ({self.model_a_label}) | Model B ({self.model_b_label}) |
|--------|-------------------------------|-------------------------------|
| Circuit size | {len(self.circuit_a)} heads | {len(self.circuit_b)} heads |
| Sufficiency | {fa.get('sufficiency', 0):.2f} | {fb.get('sufficiency', 0):.2f} |
| Comprehensiveness | {fa.get('comprehensiveness', 0):.2f} | {fb.get('comprehensiveness', 0):.2f} |
| F1 | {fa.get('f1', 0):.2f} | {fb.get('f1', 0):.2f} |

**Added heads** (new in {self.model_b_label}): `{added_str}`
**Removed heads** (dropped from {self.model_b_label}): `{removed_str}`
**Shared heads**: `{shared_str}`

### Attribution delta (shared heads, top 10 by |Δ|)

| Head | Δ attr (normalised) |
|------|---------------------|
{drift_rows}

*Stability score (Jaccard): {self.stability_score:.3f} — Article 72 / Annex IV §6 post-market monitoring*
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class CircuitDiff:
    """
    Compute mechanistic circuit diffs between two model versions.

    Parameters
    ----------
    model_a : GlassboxV2
        First model (e.g. base checkpoint).
    model_b : GlassboxV2
        Second model (e.g. fine-tuned or updated checkpoint).
    label_a : str, optional
        Human-readable label for model A. Default "model_a".
    label_b : str, optional
        Human-readable label for model B. Default "model_b".

    Examples
    --------
    >>> from transformer_lens import HookedTransformer
    >>> from glassbox import GlassboxV2
    >>> from glassbox.circuit_diff import CircuitDiff
    >>>
    >>> model_base     = HookedTransformer.from_pretrained("gpt2")
    >>> model_finetuned = HookedTransformer.from_pretrained("my-org/gpt2-finetuned")
    >>>
    >>> gb_base = GlassboxV2(model_base)
    >>> gb_ft   = GlassboxV2(model_finetuned)
    >>>
    >>> differ = CircuitDiff(gb_base, gb_ft, label_a="gpt2-base", label_b="gpt2-ft")
    >>>
    >>> diff = differ.diff(
    ...     prompt    = "When Mary and John went to the store, John gave a drink to",
    ...     correct   = " Mary",
    ...     incorrect = " John",
    ... )
    >>> print(diff.change_summary)
    >>> print(diff.to_markdown())
    """

    def __init__(
        self,
        model_a,
        model_b,
        label_a: str = "model_a",
        label_b: str = "model_b",
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.label_a = label_a
        self.label_b = label_b

    # -------------------------------------------------------------------------

    def diff(
        self,
        prompt:    str,
        correct:   str,
        incorrect: str,
        method:    str = "taylor",
        n_steps:   int = 10,
    ) -> CircuitDiffResult:
        """
        Run circuit discovery on both models and compute the diff.

        Parameters
        ----------
        prompt    : Input text.
        correct   : Correct (target) token.
        incorrect : Distractor token.
        method    : Attribution method ("taylor" or "integrated_gradients").
        n_steps   : Steps for integrated gradients (ignored for taylor).

        Returns
        -------
        CircuitDiffResult with full diff metadata.
        """
        logger.info("CircuitDiff: analysing model_a (%s)…", self.label_a)
        result_a = self.model_a.analyze(
            prompt, correct, incorrect, method=method, n_steps=n_steps
        )

        logger.info("CircuitDiff: analysing model_b (%s)…", self.label_b)
        result_b = self.model_b.analyze(
            prompt, correct, incorrect, method=method, n_steps=n_steps
        )

        return self._compute_diff(prompt, correct, incorrect, result_a, result_b)

    # -------------------------------------------------------------------------

    def batch_diff(
        self,
        prompts:     List[Tuple[str, str, str]],
        method:      str  = "taylor",
        skip_errors: bool = True,
    ) -> List[CircuitDiffResult]:
        """
        Run diff() on a batch of (prompt, correct, incorrect) triples.

        Returns a list of CircuitDiffResult (one per prompt).
        Failed prompts are skipped if skip_errors=True, raised otherwise.
        """
        results = []
        for idx, (prompt, correct, incorrect) in enumerate(prompts):
            logger.info("BatchDiff %d/%d", idx + 1, len(prompts))
            try:
                results.append(self.diff(prompt, correct, incorrect, method=method))
            except Exception as exc:
                if not skip_errors:
                    raise
                logger.warning("BatchDiff: skipping prompt %d — %s", idx + 1, exc)
        return results

    # -------------------------------------------------------------------------

    def summary_stats(
        self,
        diffs: List[CircuitDiffResult],
    ) -> Dict:
        """
        Aggregate statistics across a batch of diffs.

        Returns mean/std stability score, mean attribution drift,
        most commonly added/removed heads, and F1 delta distribution.
        """
        if not diffs:
            return {"error": "No diffs provided."}

        stabilities = [d.stability_score  for d in diffs]
        drifts      = [d.attribution_drift for d in diffs]
        f1_deltas   = [
            d.faithfulness_b.get("f1", 0) - d.faithfulness_a.get("f1", 0)
            for d in diffs
        ]

        # Head frequency counts
        added_counts:   Dict[Tuple, int] = {}
        removed_counts: Dict[Tuple, int] = {}
        for d in diffs:
            for h in d.added_heads:
                added_counts[h]   = added_counts.get(h, 0) + 1
            for h in d.removed_heads:
                removed_counts[h] = removed_counts.get(h, 0) + 1

        top_added   = sorted(added_counts.items(),   key=lambda x: -x[1])[:5]
        top_removed = sorted(removed_counts.items(), key=lambda x: -x[1])[:5]

        return {
            "n_prompts":         len(diffs),
            "stability": {
                "mean":  float(np.mean(stabilities)),
                "std":   float(np.std(stabilities)),
                "min":   float(np.min(stabilities)),
                "max":   float(np.max(stabilities)),
            },
            "attribution_drift": {
                "mean":  float(np.mean(drifts)),
                "std":   float(np.std(drifts)),
            },
            "f1_delta": {
                "mean":  float(np.mean(f1_deltas)),
                "std":   float(np.std(f1_deltas)),
            },
            "most_commonly_added":   [(list(h), c) for h, c in top_added],
            "most_commonly_removed": [(list(h), c) for h, c in top_removed],
            "model_a_label": self.label_a,
            "model_b_label": self.label_b,
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _compute_diff(
        self,
        prompt:    str,
        correct:   str,
        incorrect: str,
        result_a:  Dict,
        result_b:  Dict,
    ) -> CircuitDiffResult:
        circuit_a = [tuple(h) for h in result_a["circuit"]]
        circuit_b = [tuple(h) for h in result_b["circuit"]]

        set_a = set(circuit_a)
        set_b = set(circuit_b)

        added   = sorted(set_b - set_a)
        removed = sorted(set_a - set_b)
        shared  = sorted(set_a & set_b)
        union   = set_a | set_b

        # Jaccard stability
        stability = len(shared) / len(union) if union else 1.0

        # Normalised attribution delta for shared heads
        ld_a = abs(result_a.get("clean_ld", 1.0)) or 1.0
        ld_b = abs(result_b.get("clean_ld", 1.0)) or 1.0

        attrs_a = result_a.get("attributions", {})
        attrs_b = result_b.get("attributions", {})

        delta: Dict[str, float] = {}
        for (l, h) in shared:
            key = str((l, h))
            norm_a = attrs_a.get(key, 0.0) / ld_a
            norm_b = attrs_b.get(key, 0.0) / ld_b
            delta[key] = float(norm_b - norm_a)

        drift = float(np.mean([abs(v) for v in delta.values()])) if delta else 0.0

        return CircuitDiffResult(
            prompt            = prompt,
            correct           = correct,
            incorrect         = incorrect,
            model_a_label     = self.label_a,
            model_b_label     = self.label_b,
            circuit_a         = circuit_a,
            circuit_b         = circuit_b,
            added_heads       = added,
            removed_heads     = removed,
            shared_heads      = shared,
            attribution_delta = delta,
            stability_score   = float(stability),
            attribution_drift = drift,
            faithfulness_a    = result_a.get("faithfulness", {}),
            faithfulness_b    = result_b.get("faithfulness", {}),
        )
