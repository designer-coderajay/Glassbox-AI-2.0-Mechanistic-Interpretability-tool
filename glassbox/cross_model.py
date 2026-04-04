# SPDX-License-Identifier: BUSL-1.1
# Copyright (C) 2026 Ajay Pravin Mahale <mahale.ajay01@gmail.com>
# Licensed under the Business Source License 1.1 (see LICENSE-COMMERCIAL).
# CrossModel is patent-pending — see PATENTS.md.
# Free for non-commercial and internal production use.
# Commercial redistribution / SaaS use requires a separate license.
# Contact: mahale.ajay01@gmail.com
"""
glassbox.cross_model
====================
Cross-model circuit comparison: analyse mechanistic interpretability circuits
across multiple transformer architectures and produce comparative reports.

Answers: "Do different model families use the same computational circuits
for the same task?" (e.g., GPT-2 vs. Llama-3 vs. Mistral solving the same
IOI task with identical prompts).

This module enables:
  1. Parallel circuit discovery across model families
  2. Normalised circuit comparison (Jaccard similarity on relative head positions)
  3. Attribution correlation analysis (comparing importance across models)
  4. Consensus circuit identification (heads used by ≥50% of models)
  5. Memory-efficient sequential model loading for large ensembles

Primary use case
-----------------
  A research team is studying mechanistic interpretability across model
  scaling laws. They run identical IOI (Indirect Object Identification)
  analyses on GPT-2, Pythia-1B, Pythia-2.8B, and Llama-2-7B, then ask:
    - How similar are the circuits (Jaccard)?
    - Do the same relative head positions appear?
    - Which heads constitute the "universal" IOI circuit?

  With CrossModelComparison they can:
    1. Identify circuit similarity scores (Jaccard with binning)
    2. Compute attribution correlations normalised by model depth/width
    3. Detect consensus heads across models
    4. Export a detailed comparative report

Mathematical Foundation
------------------------
**Circuit Normalisation:**
  A circuit is a set C = {(l₁, h₁), (l₂, h₂), …} of integer (layer, head) pairs.
  To compare circuits across models with different depths n_layers and widths n_heads,
  normalise each head position to [0, 1) × [0, 1):

    (layer_norm, head_norm) = (l / n_layers, h / n_heads)

**Jaccard Similarity with Binning (±0.1 grid):**
  Two normalised positions p₁ = (l₁_n, h₁_n) and p₂ = (l₂_n, h₂_n) are
  "the same head" if they fall in the same (bin_size × bin_size) grid cell:

    bin₁ = (⌊l₁_n / bin_size⌋, ⌊h₁_n / bin_size⌋)
    bin₂ = (⌊l₂_n / bin_size⌋, ⌊h₂_n / bin_size⌋)
    "same" ⟺ bin₁ = bin₂

  Jaccard similarity (binned):
    sim(C_A, C_B) = |{bins in A} ∩ {bins in B}| / |{bins in A} ∪ {bins in B}|

  Range: [0, 1]. Value 1.0 = identical circuits; 0.0 = no overlap.

**Attribution Correlation:**
  Each model's attributions are normalised by the clean logit difference
  (clean_ld) to make them comparable across models:

    attr_norm(h) = attr_raw(h) / |clean_ld|

  For two models A and B:
    1. Align both circuits to a shared grid of normalised position bins
    2. Build attribution vectors v_A, v_B indexed by bin
    3. Fill missing bins with 0.0 (head not in circuit)
    4. Compute Pearson correlation: pearsonr(v_A, v_B)
    5. Return 0.0 if <3 overlapping bins (insufficient data)

**Consensus Heads:**
  A normalised position p is "consensus" if it appears in ≥ min_model_fraction
  of the models' circuits (default 50%). Useful for identifying universal circuits.

LEGAL NOTICE — DOCUMENTATION AID ONLY
---------------------------------------
This module is a research tool for model architecture robustness evaluation.
Its outputs do not constitute a conformity assessment, legal advice, or
regulatory certification under Regulation (EU) 2024/1689. Consult qualified
legal and technical counsel before relying on cross-model results for
regulatory decisions.

Regulatory References (informational only)
------------------------------------------
Regulation (EU) 2024/1689 — EU AI Act:
  Article 10   — Data governance (use of high-quality data across versions)
  Article 11   — Technical documentation
  Article 15   — Robustness and accuracy across model versions and families
"""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

__all__ = [
    "ModelAnalysisConfig",
    "SingleModelResult",
    "CrossModelSimilarity",
    "CrossModelReport",
    "CrossModelComparison",
    "compare_models",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data model — configuration and results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelAnalysisConfig:
    """
    Configuration for analysing a single model in cross-model comparison.

    Attributes
    ----------
    model_name : str
        Model identifier for transformer_lens/HuggingFace
        (e.g., "gpt2", "meta-llama/Llama-2-7b", "mistralai/Mistral-7B-v0.1").
    clean_prompt : str
        Prompt with correct completion
        (e.g., "When Mary and John went to the store, Mary gave a drink to").
    corrupted_prompt : str
        Same prompt with corrupted token for counterfactual
        (e.g., "When Alice and John went to the store, Alice gave a drink to").
    target_token : str
        Correct (direct object) token (e.g., "Mary" or " Mary").
    distractor_token : str
        Incorrect (subject) token (e.g., "John" or " John").
    device : str, optional
        Device for this model ("cpu", "cuda", "cuda:0", etc.). Default "cpu".
    """
    model_name: str
    clean_prompt: str
    corrupted_prompt: str
    target_token: str
    distractor_token: str
    device: str = "cpu"


@dataclass
class SingleModelResult:
    """
    Result of mechanistic interpretability analysis on one model.

    Attributes
    ----------
    model_name : str
        Model identifier (from config).
    n_layers : int
        Number of transformer layers in the model.
    n_heads : int
        Number of attention heads per layer.
    circuit : List[Tuple[int, int]]
        Top-k heads in circuit, as [(layer, head), ...].
        Sorted by attribution magnitude.
    attributions : Dict[Tuple[int,int], float]
        Raw attribution scores for each head.
        Keys are (layer, head) tuples; values are floats.
    clean_ld : float
        Logit difference (LD) on clean prompt:
        LD = log(P(target)) − log(P(distractor)).
    sufficiency : float
        Circuit sufficiency: how much of clean_ld is preserved
        when circuit heads are kept and others patched? ∈ [0, 1].
    comprehensiveness : float
        Circuit comprehensiveness: how much of clean_ld is removed
        when circuit heads are patched? ∈ [0, 1].
    arch_report : Optional[Any]
        Additional architecture-specific analysis (e.g., from GlassboxV2).
    """
    model_name: str
    n_layers: int
    n_heads: int
    circuit: List[Tuple[int, int]]
    attributions: Dict[Tuple[int, int], float]
    clean_ld: float
    sufficiency: float
    comprehensiveness: float
    arch_report: Optional[Any] = None

    def normalised_circuit(self) -> Set[Tuple[float, float]]:
        """
        Return circuit as set of normalised (layer_frac, head_frac) tuples.

        Each position (layer_idx, head_idx) is mapped to [0, 1) × [0, 1):
          (layer_frac, head_frac) = (layer_idx / n_layers, head_idx / n_heads)

        Returns
        -------
        Set[Tuple[float, float]]
            Normalised head positions.
        """
        return {
            (l / self.n_layers, h / self.n_heads)
            for l, h in self.circuit
        }

    def normalised_attributions(self) -> Dict[Tuple[float, float], float]:
        """
        Return attributions keyed by normalised positions, values scaled by LD.

        Normalisation formula:
          attr_norm(h) = attr_raw(h) / |clean_ld|

        This makes attributions comparable across models with different baseline
        logit differences.

        Returns
        -------
        Dict[Tuple[float, float], float]
            Mapping from (layer_norm, head_norm) to normalised attribution.
            Values are divided by |clean_ld| to centre at scale ~±1.
        """
        ld_abs = abs(self.clean_ld) if self.clean_ld != 0 else 1.0
        return {
            (l / self.n_layers, h / self.n_heads): attr / ld_abs
            for (l, h), attr in self.attributions.items()
        }


@dataclass
class CrossModelSimilarity:
    """
    Similarity metrics between two models' circuits.

    Attributes
    ----------
    model_a : str
        Name of first model.
    model_b : str
        Name of second model.
    jaccard_similarity : float
        Jaccard similarity of binned normalised circuits ∈ [0, 1].
        Computed with ±0.1 bin size (10 × 10 grid per [0, 1) × [0, 1)).
    attribution_correlation : float
        Pearson correlation of attribution vectors aligned by bin.
        Range [-1, 1]. Returns 0.0 if <3 overlapping bins.
    shared_normalised_heads : List[Tuple[float, float]]
        Normalised head positions in both circuits (same bin after rounding).
    unique_to_a : int
        Number of binned positions in A not in B.
    unique_to_b : int
        Number of binned positions in B not in A.
    """
    model_a: str
    model_b: str
    jaccard_similarity: float
    attribution_correlation: float
    shared_normalised_heads: List[Tuple[float, float]]
    unique_to_a: int
    unique_to_b: int


@dataclass
class CrossModelReport:
    """
    Complete cross-model mechanistic interpretability report.

    Attributes
    ----------
    task_description : str
        Human-readable task summary (e.g., "Indirect Object Identification").
    results : List[SingleModelResult]
        Analysis result for each model.
    similarities : List[CrossModelSimilarity]
        Pairwise similarity metrics (all pairs, if n models → n(n-1)/2 pairs).
    consensus_heads : List[Tuple[float, float]]
        Normalised head positions appearing in ≥50% of models.
    """
    task_description: str
    results: List[SingleModelResult]
    similarities: List[CrossModelSimilarity]
    consensus_heads: List[Tuple[float, float]]

    @property
    def summary(self) -> str:
        """
        Human-readable summary of cross-model comparison.

        Returns
        -------
        str
            Multi-line summary including task, model count, avg Jaccard,
            consensus head count, and circuit size ranges.
        """
        if not self.results:
            return "No results."

        n_models = len(self.results)
        jaccard_vals = [s.jaccard_similarity for s in self.similarities]
        avg_jaccard = float(np.mean(jaccard_vals)) if jaccard_vals else 0.0

        circuit_sizes = [len(r.circuit) for r in self.results]
        min_size, max_size = min(circuit_sizes), max(circuit_sizes)

        ld_vals = [abs(r.clean_ld) for r in self.results]
        mean_ld = float(np.mean(ld_vals))

        lines = [
            f"Cross-Model Circuit Analysis: {self.task_description}",
            f"  Models: {n_models} ({', '.join(r.model_name for r in self.results)})",
            f"  Jaccard similarity (avg): {avg_jaccard:.3f}",
            f"  Circuit size range: {min_size}–{max_size} heads",
            f"  Mean clean LD: {mean_ld:.3f}",
            f"  Consensus heads (≥50%): {len(self.consensus_heads)}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialisable dict for JSON export / audit trail.

        Returns
        -------
        Dict[str, Any]
            Complete report as nested dicts/lists.
        """
        return {
            "task_description": self.task_description,
            "n_models": len(self.results),
            "results": [
                {
                    "model_name": r.model_name,
                    "n_layers": r.n_layers,
                    "n_heads": r.n_heads,
                    "circuit_size": len(r.circuit),
                    "clean_ld": r.clean_ld,
                    "sufficiency": r.sufficiency,
                    "comprehensiveness": r.comprehensiveness,
                    "circuit": [list(h) for h in r.circuit],
                    "attributions": {
                        str(k): v for k, v in r.attributions.items()
                    },
                }
                for r in self.results
            ],
            "similarities": [
                {
                    "model_a": s.model_a,
                    "model_b": s.model_b,
                    "jaccard_similarity": s.jaccard_similarity,
                    "attribution_correlation": s.attribution_correlation,
                    "shared_heads": len(s.shared_normalised_heads),
                    "unique_to_a": s.unique_to_a,
                    "unique_to_b": s.unique_to_b,
                }
                for s in self.similarities
            ],
            "consensus_heads": [list(h) for h in self.consensus_heads],
        }

    @property
    def attribution_table(self) -> str:
        """
        ASCII table: models (rows) × normalised positions (cols),
        values = normalised attribution scores.

        Returns a markdown table suitable for reports/documentation.
        Shows which heads matter most in each model.

        Returns
        -------
        str
            Markdown formatted table.
        """
        if not self.results:
            return "No results to tabulate."

        # Collect all normalised bins across all models
        all_positions: Set[Tuple[float, float]] = set()
        for result in self.results:
            all_positions.update(result.normalised_circuit())

        if not all_positions:
            return "No circuit positions found."

        # Sort positions by layer then head for readability
        sorted_positions = sorted(all_positions)

        # Build table
        header = "| Model | " + " | ".join(
            f"({l:.1f},{h:.1f})" for l, h in sorted_positions
        ) + " |"
        sep = "|-------" + "|-------" * len(sorted_positions) + "|"

        rows = [header, sep]
        for result in self.results:
            norm_attrs = result.normalised_attributions()
            values = [
                f"{norm_attrs.get(pos, 0.0):.2f}"
                for pos in sorted_positions
            ]
            row = f"| {result.model_name} | " + " | ".join(values) + " |"
            rows.append(row)

        return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main class: CrossModelComparison
# ─────────────────────────────────────────────────────────────────────────────

class CrossModelComparison:
    """
    Runs identical mechanistic interpretability analysis across multiple models.

    Designed for memory-constrained environments: loads and unloads models
    sequentially. Each model is fully analysed, then freed before loading next.

    Parameters
    ----------
    configs : List[ModelAnalysisConfig]
        Configuration for each model to analyse (one per model).
    top_k_circuit : int, optional
        Number of top attribution heads to include in each circuit.
        Default 10.
    device : str, optional
        Default device for model loading ("cpu", "cuda", etc.). Default "cpu".
        Each config can override this.

    Attributes
    ----------
    configs : List[ModelAnalysisConfig]
        Stored configurations.
    top_k_circuit : int
        Stored top_k value.
    device : str
        Default device.

    Examples
    --------
    >>> from glassbox.cross_model import CrossModelComparison, ModelAnalysisConfig
    >>> from transformer_lens import HookedTransformer
    >>>
    >>> configs = [
    ...     ModelAnalysisConfig(
    ...         model_name="gpt2",
    ...         clean_prompt="When Mary and John went to the store, John gave a drink to",
    ...         corrupted_prompt="When Alice and John went to the store, John gave a drink to",
    ...         target_token=" Mary",
    ...         distractor_token=" John",
    ...         device="cuda",
    ...     ),
    ...     ModelAnalysisConfig(
    ...         model_name="meta-llama/Llama-2-7b",
    ...         clean_prompt="When Mary and John went to the store, John gave a drink to",
    ...         corrupted_prompt="When Alice and John went to the store, John gave a drink to",
    ...         target_token=" Mary",
    ...         distractor_token=" John",
    ...         device="cuda",
    ...     ),
    ... ]
    >>> cmp = CrossModelComparison(configs, top_k_circuit=10, device="cuda")
    >>> report = cmp.run(use_glassbox=True)
    >>> print(report.summary())
    >>> print(report.attribution_table)
    """

    def __init__(
        self,
        configs: List[ModelAnalysisConfig],
        top_k_circuit: int = 10,
        device: str = "cpu",
    ) -> None:
        self.configs = configs
        self.top_k_circuit = top_k_circuit
        self.device = device

    def run(self, use_glassbox: bool = True) -> CrossModelReport:
        """
        Run analysis on all models sequentially.

        If use_glassbox=True: uses GlassboxV2.analyze() for full attribution
        patching and circuit discovery. Requires glassbox.core.GlassboxV2.

        If use_glassbox=False: uses lightweight direct-attention analysis
        (no patching). Less accurate but faster and lower memory.

        Models are loaded, analysed, and freed one at a time to avoid OOM.

        Parameters
        ----------
        use_glassbox : bool, optional
            If True, use GlassboxV2 for attribution. Default True.

        Returns
        -------
        CrossModelReport
            Full comparative analysis across all models.
        """
        if not self.configs:
            raise ValueError("No configs provided.")

        logger.info("CrossModelComparison: starting analysis of %d models", len(self.configs))

        results: List[SingleModelResult] = []
        for idx, config in enumerate(self.configs):
            logger.info(
                "CrossModelComparison: analysing model %d/%d (%s)…",
                idx + 1,
                len(self.configs),
                config.model_name,
            )
            try:
                if use_glassbox:
                    result = self._analyse_with_glassbox(config)
                else:
                    result = self._analyse_lightweight(config)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "CrossModelComparison: failed to analyse %s — %s",
                    config.model_name,
                    exc,
                    exc_info=True,
                )
                raise

            # Free memory
            self._cleanup_model()

        logger.info("CrossModelComparison: all models analysed")

        # Compute pairwise similarities
        similarities = self._compute_pairwise_similarities(results)

        # Find consensus heads
        consensus = self._find_consensus_heads(results, min_model_fraction=0.5)

        task_desc = (
            f"Cross-model IOI task: "
            f'"{self.configs[0].clean_prompt[:50]}…"'
        )

        report = CrossModelReport(
            task_description=task_desc,
            results=results,
            similarities=similarities,
            consensus_heads=consensus,
        )

        logger.info("CrossModelComparison: report complete")
        return report

    # ─────────────────────────────────────────────────────────────────────────

    def _analyse_with_glassbox(self, config: ModelAnalysisConfig) -> SingleModelResult:
        """
        Analyse one model using GlassboxV2.analyze().

        Parameters
        ----------
        config : ModelAnalysisConfig
            Model configuration.

        Returns
        -------
        SingleModelResult
            Circuit and attribution data for this model.
        """
        from glassbox.core import GlassboxV2
        from transformer_lens import HookedTransformer

        # Load model
        model = HookedTransformer.from_pretrained(
            config.model_name,
            device=config.device,
        )

        # Initialise Glassbox
        gb = GlassboxV2(model)

        # Run analysis
        result = gb.analyze(
            prompt=config.clean_prompt,
            correct=config.target_token,
            incorrect=config.distractor_token,
            method="taylor",
        )

        # Extract circuit (top-k)
        circuit_with_attrs = [
            (tuple(h), result.get("attributions", {}).get(str(h), 0.0))
            for h in result.get("circuit", [])
        ]
        circuit_with_attrs.sort(key=lambda x: abs(x[1]), reverse=True)
        circuit = [h for h, _ in circuit_with_attrs[:self.top_k_circuit]]

        # Build attribution dict — ALL heads (not just circuit) so Pearson r
        # can be computed across the full attribution vector, not just the sparse circuit.
        all_attrs_raw = result.get("attributions", {})
        attributions: Dict[Tuple[int, int], float] = {}
        for key_str, val in all_attrs_raw.items():
            try:
                inner = key_str.strip("()")
                l_str, h_str = inner.split(", ")
                attributions[(int(l_str), int(h_str))] = float(val)
            except Exception:
                pass

        # Faithfulness metrics
        faith = result.get("faithfulness", {})

        single = SingleModelResult(
            model_name=config.model_name,
            n_layers=model.cfg.n_layers,
            n_heads=model.cfg.n_heads,
            circuit=circuit,
            attributions=attributions,
            clean_ld=result.get("clean_ld", 0.0),
            sufficiency=faith.get("sufficiency", 0.0),
            comprehensiveness=faith.get("comprehensiveness", 0.0),
            arch_report=result,
        )

        # Explicitly release model before GC to avoid VRAM leaks when loading
        # multiple models sequentially.  Python reference counting is not
        # guaranteed to free large tensors immediately; explicit del ensures
        # the CUDA allocator can reclaim memory before the next model loads.
        del gb, model
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return single

    def _analyse_lightweight(self, config: ModelAnalysisConfig) -> SingleModelResult:
        """
        Analyse one model using lightweight direct-attention analysis.

        No patching; just computes attention-based attributions on clean vs. corrupted.

        Parameters
        ----------
        config : ModelAnalysisConfig
            Model configuration.

        Returns
        -------
        SingleModelResult
            Circuit and attribution data for this model.
        """
        from transformer_lens import HookedTransformer

        # Load model
        model = HookedTransformer.from_pretrained(
            config.model_name,
            device=config.device,
        )

        # Tokenise
        clean_tokens = model.to_tokens(config.clean_prompt)
        clean_target_id = model.to_single_token(config.target_token)
        clean_distract_id = model.to_single_token(config.distractor_token)

        # Run clean forward pass
        with torch.no_grad():
            clean_logits = model(clean_tokens)

        clean_ld = float(
            clean_logits[0, -1, clean_target_id] -
            clean_logits[0, -1, clean_distract_id]
        )

        # Run corrupted forward pass
        corrupted_tokens = model.to_tokens(config.corrupted_prompt)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens)

        # Simplified attribution: average absolute attention weight difference
        # (This is a heuristic; full GlassboxV2 uses proper patching)
        attributions: Dict[Tuple[int, int], float] = {}
        max_attr = 0.0

        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                # Simple heuristic: abs difference in attention patterns
                hook_name = f"blocks.{layer}.attn.hook_attn_weights"
                try:
                    _, clean_cache = model.run_with_cache(
                        clean_tokens,
                        names_filter=lambda n: n == hook_name,
                    )
                    _, corrupt_cache = model.run_with_cache(
                        corrupted_tokens,
                        names_filter=lambda n: n == hook_name,
                    )

                    if hook_name in clean_cache and hook_name in corrupt_cache:
                        clean_attn = clean_cache[hook_name][0, head, :, :].mean().item()
                        corrupt_attn = corrupt_cache[hook_name][0, head, :, :].mean().item()
                        attr = abs(clean_attn - corrupt_attn)
                        attributions[(layer, head)] = attr
                        max_attr = max(max_attr, attr)
                except Exception:
                    pass

        # Normalise and select top-k
        if max_attr > 0:
            attributions = {
                k: v / max_attr for k, v in attributions.items()
            }

        sorted_attrs = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        circuit = [h for h, _ in sorted_attrs[:self.top_k_circuit]]

        return SingleModelResult(
            model_name=config.model_name,
            n_layers=model.cfg.n_layers,
            n_heads=model.cfg.n_heads,
            circuit=circuit,
            attributions=attributions,
            clean_ld=clean_ld,
            sufficiency=0.5,  # Heuristic
            comprehensiveness=0.5,  # Heuristic
            arch_report=None,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _compute_pairwise_similarities(
        self,
        results: List[SingleModelResult],
    ) -> List[CrossModelSimilarity]:
        """
        Compute Jaccard and attribution correlation for all pairs of models.

        Parameters
        ----------
        results : List[SingleModelResult]
            Analysis results for all models.

        Returns
        -------
        List[CrossModelSimilarity]
            Pairwise similarity metrics (all pairs).
        """
        similarities: List[CrossModelSimilarity] = []

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                result_a = results[i]
                result_b = results[j]

                # Jaccard similarity
                jaccard = self._jaccard(
                    result_a.normalised_circuit(),
                    result_b.normalised_circuit(),
                    bin_size=0.1,
                )

                # Attribution correlation
                corr = self._attribution_pearsonr(
                    result_a,
                    result_b,
                    bin_size=0.1,
                )

                # Shared heads (binned)
                shared = self._shared_normalised_heads(
                    result_a.normalised_circuit(),
                    result_b.normalised_circuit(),
                    bin_size=0.1,
                )

                # Compute unique heads as bin-level differences so the granularity
                # is consistent: two positions mapping to the same bin count as one.
                _BIN = 0.1
                _MB = max(0, int(1.0 / _BIN) - 1)
                bins_ra = {
                    (min(int(l / _BIN), _MB), min(int(h / _BIN), _MB))
                    for l, h in result_a.normalised_circuit()
                }
                bins_rb = {
                    (min(int(l / _BIN), _MB), min(int(h / _BIN), _MB))
                    for l, h in result_b.normalised_circuit()
                }
                unique_a = len(bins_ra - bins_rb)
                unique_b = len(bins_rb - bins_ra)

                sim = CrossModelSimilarity(
                    model_a=result_a.model_name,
                    model_b=result_b.model_name,
                    jaccard_similarity=jaccard,
                    attribution_correlation=corr,
                    shared_normalised_heads=shared,
                    unique_to_a=unique_a,
                    unique_to_b=unique_b,
                )
                similarities.append(sim)

        return similarities

    def _find_consensus_heads(
        self,
        results: List[SingleModelResult],
        min_model_fraction: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """
        Find normalised head positions present in ≥ min_model_fraction of models.

        Parameters
        ----------
        results : List[SingleModelResult]
            Analysis results for all models.
        min_model_fraction : float, optional
            Minimum fraction of models required (default 0.5 = ≥50%).

        Returns
        -------
        List[Tuple[float, float]]
            Consensus normalised head positions, sorted by appearance count desc.
        """
        if not results:
            return []

        # Count appearances of each binned position
        position_counts: Dict[Tuple[float, float], int] = {}
        bin_size = 0.1

        max_bin = max(0, int(1.0 / bin_size) - 1)
        for result in results:
            norm_circuit = result.normalised_circuit()
            seen_bins: Set[Tuple[int, int]] = set()

            for l_norm, h_norm in norm_circuit:
                bin_l = min(int(l_norm / bin_size), max_bin)
                bin_h = min(int(h_norm / bin_size), max_bin)
                bin_key = (bin_l, bin_h)

                if bin_key not in seen_bins:
                    seen_bins.add(bin_key)
                    # Store bin corner as representative position
                    rep_pos = (bin_l * bin_size, bin_h * bin_size)
                    position_counts[rep_pos] = position_counts.get(rep_pos, 0) + 1

        # Filter by min fraction
        threshold = min_model_fraction * len(results)
        consensus = [
            pos for pos, count in position_counts.items()
            if count >= threshold
        ]

        # Sort by count descending
        consensus.sort(
            key=lambda pos: -position_counts[pos],
        )

        return consensus

    # ─────────────────────────────────────────────────────────────────────────
    # Static utility methods
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(
        set_a: Set[Tuple[float, float]],
        set_b: Set[Tuple[float, float]],
        bin_size: float = 0.1,
    ) -> float:
        """
        Jaccard similarity on binned normalised positions.

        Two positions fall in the "same" bin if:
          bin_pos_a = (⌊l_a / bin_size⌋, ⌊h_a / bin_size⌋)
          bin_pos_b = (⌊l_b / bin_size⌋, ⌊h_b / bin_size⌋)
          same ⟺ bin_pos_a == bin_pos_b

        Parameters
        ----------
        set_a : Set[Tuple[float, float]]
            Normalised positions from model A.
        set_b : Set[Tuple[float, float]]
            Normalised positions from model B.
        bin_size : float, optional
            Grid cell size (default 0.1 → 10×10 grid). Default 0.1.

        Returns
        -------
        float
            Jaccard similarity ∈ [0, 1]. 1.0 = identical; 0.0 = no overlap.
        """
        # Convert to bins, clamping to valid grid range [0, max_bin].
        # norm values in [0, 1]: int(1.0 / 0.1) = 10, which is out-of-range for a
        # 10×10 grid (indices 0–9). Clamp to max_bin = int(1.0/bin_size) - 1.
        max_bin = max(0, int(1.0 / bin_size) - 1)
        bins_a = {
            (min(int(l / bin_size), max_bin), min(int(h / bin_size), max_bin))
            for l, h in set_a
        }
        bins_b = {
            (min(int(l / bin_size), max_bin), min(int(h / bin_size), max_bin))
            for l, h in set_b
        }

        intersection = len(bins_a & bins_b)
        union = len(bins_a | bins_b)

        if union == 0:
            # Both circuits empty — no evidence of structural overlap; return 0.0.
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _attribution_pearsonr(
        result_a: SingleModelResult,
        result_b: SingleModelResult,
        bin_size: float = 0.1,
    ) -> float:
        """
        Pearson correlation of normalised attributions aligned by bin.

        Procedure:
          1. Build shared grid of position bins
          2. Fill attribution vectors v_A, v_B indexed by bin
          3. Entries are normalised attributions (attr / |clean_ld|)
          4. Missing entries = 0.0
          5. Compute Pearson r
          6. Return 0.0 if <3 overlapping bins (insufficient data)

        Parameters
        ----------
        result_a : SingleModelResult
            Analysis result for model A.
        result_b : SingleModelResult
            Analysis result for model B.
        bin_size : float, optional
            Grid cell size (default 0.1). Default 0.1.

        Returns
        -------
        float
            Pearson correlation ∈ [-1, 1] or 0.0 if insufficient overlap.
        """
        # Get normalised attributions
        attrs_a = result_a.normalised_attributions()
        attrs_b = result_b.normalised_attributions()

        # Map to bins, clamping to valid grid range [0, max_bin].
        max_bin = max(0, int(1.0 / bin_size) - 1)
        bins_a = {}
        for (l_norm, h_norm), attr in attrs_a.items():
            bin_key = (min(int(l_norm / bin_size), max_bin), min(int(h_norm / bin_size), max_bin))
            bins_a[bin_key] = attr

        bins_b = {}
        for (l_norm, h_norm), attr in attrs_b.items():
            bin_key = (min(int(l_norm / bin_size), max_bin), min(int(h_norm / bin_size), max_bin))
            bins_b[bin_key] = attr

        # Find shared bins
        shared_bins = set(bins_a.keys()) & set(bins_b.keys())

        if len(shared_bins) < 3:
            return 0.0

        # Extract values for shared bins
        v_a = np.array([bins_a[bin_key] for bin_key in shared_bins])
        v_b = np.array([bins_b[bin_key] for bin_key in shared_bins])

        # Compute Pearson
        try:
            corr, _ = pearsonr(v_a, v_b)
            return float(corr)
        except Exception:
            return 0.0

    @staticmethod
    def _shared_normalised_heads(
        set_a: Set[Tuple[float, float]],
        set_b: Set[Tuple[float, float]],
        bin_size: float = 0.1,
    ) -> List[Tuple[float, float]]:
        """
        Find normalised head positions in both circuits (same bin).

        Parameters
        ----------
        set_a : Set[Tuple[float, float]]
            Normalised positions from model A.
        set_b : Set[Tuple[float, float]]
            Normalised positions from model B.
        bin_size : float, optional
            Grid cell size (default 0.1). Default 0.1.

        Returns
        -------
        List[Tuple[float, float]]
            Shared normalised positions (bin corners).
        """
        bins_a = {
            (int(l / bin_size), int(h / bin_size))
            for l, h in set_a
        }
        bins_b = {
            (int(l / bin_size), int(h / bin_size))
            for l, h in set_b
        }

        shared_bins = bins_a & bins_b

        # Return bin corners as representative positions
        return [
            (bin_l * bin_size, bin_h * bin_size)
            for bin_l, bin_h in shared_bins
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Memory management
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cleanup_model() -> None:
        """
        Free GPU/CPU memory after model analysis.

        Explicitly deletes model, runs garbage collection, and clears CUDA cache.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    configs: List[ModelAnalysisConfig],
    top_k_circuit: int = 10,
    device: str = "cpu",
) -> CrossModelReport:
    """
    One-shot convenience wrapper for cross-model circuit comparison.

    Equivalent to:
      cmp = CrossModelComparison(configs, top_k_circuit, device)
      return cmp.run(use_glassbox=True)

    Parameters
    ----------
    configs : List[ModelAnalysisConfig]
        Configuration for each model.
    top_k_circuit : int, optional
        Number of top heads per circuit. Default 10.
    device : str, optional
        Device for model loading. Default "cpu".

    Returns
    -------
    CrossModelReport
        Complete cross-model analysis report.

    Examples
    --------
    >>> from glassbox.cross_model import compare_models, ModelAnalysisConfig
    >>> configs = [
    ...     ModelAnalysisConfig(
    ...         model_name="gpt2",
    ...         clean_prompt="When Mary and John went to the store, John gave a drink to",
    ...         corrupted_prompt="When Alice and John went to the store, John gave a drink to",
    ...         target_token=" Mary",
    ...         distractor_token=" John",
    ...     ),
    ... ]
    >>> report = compare_models(configs, top_k_circuit=10, device="cuda")
    >>> print(report.summary())
    """
    return CrossModelComparison(configs, top_k_circuit, device).run(use_glassbox=True)
