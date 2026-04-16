"""
glassbox/acdc.py
=================
Automated Circuit Discovery (ACDC) — v4.2.3
===========================================

Implements Automated Circuit Discovery for mechanistic interpretability
(Conmy et al., NeurIPS 2023, arXiv:2304.14997).

Background
----------
ACDC discovers minimal faithful circuits by testing each directed edge in
the transformer computation graph. For each edge (sender → receiver):

1. Patch sender's contribution with the corrupted activation
2. Measure KL divergence from clean output logits
3. If KL < threshold τ: exclude edge (not causally necessary)
4. If KL ≥ τ: retain edge (causally necessary)

Processing order: topological (layer 0 → L, within layer: attn → mlp)

Algorithm
---------
Given clean and corrupted inputs:

1. Collect clean and corrupted caches via forward passes
2. Build all possible edges: (sender_layer, sender_type, sender_head) →
   (receiver_layer, receiver_type, receiver_head) where receiver_layer > sender_layer
3. Test edges in topological order:
   - For each edge e: patch all edges NOT in current circuit ∪ {e}
   - Measure KL(clean || patched)
   - If KL ≥ τ: add e to circuit (edge is necessary)
4. Return circuit + per-edge KL scores

Circuit Faithfulness Metric
---------------------------
Once circuit is discovered, measure overall faithfulness:

    kl_circuit = KL(clean_output || circuit_output)

where circuit_output is obtained by patching all edges NOT in the circuit.
Interpretation:
    - kl_circuit < 0.80  : STRONG — circuit explains most behaviour
    - kl_circuit ∈ [0.80, 1.5) : PARTIAL — circuit missing some components
    - kl_circuit ≥ 1.5  : WEAK — circuit insufficient

References
----------
Conmy et al. 2023 — "Towards Automated Circuit Discovery for Mechanistic
    Interpretability" (DeepMind, NeurIPS 2023)
    https://arxiv.org/abs/2304.14997

Syed et al. 2024 — "Attribution Patching Outperforms Automated Circuit
    Discovery" (notes on comparison with EAP)
    https://arxiv.org/abs/2310.10348
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ACDC KL threshold (Conmy et al. 2023, IOI experiments)
ACDC_KL_THRESHOLD: float = 0.10

# Circuit faithfulness threshold: kl_circuit must be below this for "STRONG"
ACDC_FAITHFULNESS_THRESHOLD: float = 0.80


# ──────────────────────────────────────────────────────────────────────────────
# Type aliases and data classes
# ──────────────────────────────────────────────────────────────────────────────

# NodeKey: (layer, 'attn'|'mlp', head_idx)
# For MLP nodes, head_idx is always 0 by convention
NodeKey = Tuple[int, str, int]


@dataclass(frozen=True)
class ACDCEdge:
    """
    A directed edge in the transformer computation graph.

    Attributes
    ----------
    sender   : (layer, 'attn'|'mlp', head_idx) — output source
    receiver : (layer, 'attn'|'mlp', head_idx) — input target
    """
    sender: NodeKey
    receiver: NodeKey

    def __repr__(self) -> str:
        """Return compact representation like "L0AH3→L2AH7" or "L0MLP→L1AH2"."""
        s_layer, s_type, s_head = self.sender
        r_layer, r_type, r_head = self.receiver
        s_str = f"L{s_layer}{'AH' if s_type == 'attn' else 'MLP'}{s_head if s_type == 'attn' else ''}"
        r_str = f"L{r_layer}{'AH' if r_type == 'attn' else 'MLP'}{r_head if r_type == 'attn' else ''}"
        return f"{s_str}→{r_str}"


@dataclass
class ACDCCircuit:
    """
    Discovered circuit: a minimal set of edges required for task performance.

    Attributes
    ----------
    edges    : Set of ACDCEdge objects retained in the circuit
    n_layers : Total transformer layers (for max-edge calculation)
    n_heads  : Heads per layer (for density calculation)
    """
    edges: Set[ACDCEdge]
    n_layers: int
    n_heads: int

    def n_edges(self) -> int:
        """Number of edges in circuit."""
        return len(self.edges)

    def head_nodes(self) -> Set[Tuple[int, int]]:
        """
        Union of all (layer, head) tuples for attention heads in circuit.

        Returns set of (layer, head_idx) for all attn edges in circuit.
        """
        nodes = set()
        for edge in self.edges:
            s_layer, s_type, s_head = edge.sender
            r_layer, r_type, r_head = edge.receiver
            if s_type == "attn":
                nodes.add((s_layer, s_head))
            if r_type == "attn":
                nodes.add((r_layer, r_head))
        return nodes

    def density(self) -> float:
        """
        Fraction of maximum possible edges retained.

        Max possible edges: all senders (layers 0..L-1) → all receivers
        (layers 1..L), where layers form strictly increasing pairs.
        """
        # For each receiver layer l_r in 1..L-1:
        # - senders from layers 0..l_r-1
        # - attn heads + mlp (1 per layer)
        # So max edges = sum over l_r of (l_r senders) * (n_heads + 1 receivers)
        max_edges = 0
        for l_r in range(1, self.n_layers):
            n_senders = l_r * (self.n_heads + 1)  # attn + mlp per layer
            n_receivers = self.n_heads + 1  # attn heads + mlp
            max_edges += n_senders * n_receivers

        if max_edges == 0:
            return 0.0
        return float(len(self.edges)) / float(max_edges)

    def to_head_list(self) -> List[Tuple[int, int]]:
        """Return sorted list of (layer, head) for all attn heads in circuit."""
        heads = self.head_nodes()
        return sorted(heads)


@dataclass
class ACDCResult:
    """
    Result of automated circuit discovery.

    Attributes
    ----------
    circuit            : ACDCCircuit object with edges
    kl_circuit         : KL divergence of circuit output vs clean (faithfulness)
    n_edges_total      : Total directed edges tested
    n_edges_retained   : Edges in final circuit
    n_edges_pruned     : Edges not in circuit
    threshold          : τ value used for pruning
    faithful           : True if kl_circuit < ACDC_FAITHFULNESS_THRESHOLD
    pruning_kl_scores  : Dict[edge -> kl] for each edge tested
    """
    circuit: ACDCCircuit
    kl_circuit: float
    n_edges_total: int
    n_edges_retained: int
    n_edges_pruned: int
    threshold: float
    faithful: bool
    pruning_kl_scores: Dict[ACDCEdge, float]

    def summary(self) -> str:
        """Return compact one-line summary of discovery results."""
        grade = self.faithfulness_grade()
        return (
            f"ACDC | circuit={self.circuit.n_edges()} edges "
            f"(density={self.circuit.density():.3f}) | "
            f"KL_circuit={self.kl_circuit:.4f} | "
            f"faithfulness={grade}"
        )

    def faithfulness_grade(self) -> str:
        """
        Grade circuit faithfulness based on KL divergence.

        Returns "STRONG", "PARTIAL", or "WEAK".
        """
        if self.kl_circuit < 0.80:
            return "STRONG"
        elif self.kl_circuit < 1.5:
            return "PARTIAL"
        else:
            return "WEAK"

    def to_dict(self) -> Dict:
        """Serialize result to dictionary for logging."""
        return {
            "n_edges_circuit": self.circuit.n_edges(),
            "n_edges_total": self.n_edges_total,
            "n_edges_pruned": self.n_edges_pruned,
            "circuit_density": round(self.circuit.density(), 4),
            "kl_circuit": round(self.kl_circuit, 4),
            "threshold": self.threshold,
            "faithful": self.faithful,
            "faithfulness_grade": self.faithfulness_grade(),
            "head_list": self.circuit.to_head_list(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# AutomatedCircuitDiscovery main class
# ──────────────────────────────────────────────────────────────────────────────

class AutomatedCircuitDiscovery:
    """
    Automated Circuit Discovery (Conmy et al. 2023).

    Discovers minimal faithful circuits by testing each edge in the
    transformer computation graph for causal necessity.

    Parameters
    ----------
    model    : HookedTransformer instance
    threshold: KL threshold τ for edge inclusion (default 0.10)
    verbose  : Enable debug logging (default False)

    Usage
    -----
    >>> acdc = AutomatedCircuitDiscovery(model, threshold=0.10)
    >>> result = acdc.discover(
    ...     clean_tokens=clean_toks,        # [1, seq_len]
    ...     corrupted_tokens=corrupted_toks, # [1, seq_len]
    ... )
    >>> print(result.summary())
    ACDC | circuit=42 edges (density=0.018) | KL_circuit=0.6234 | faithfulness=STRONG
    """

    def __init__(
        self,
        model: object,
        threshold: float = ACDC_KL_THRESHOLD,
        verbose: bool = False,
    ) -> None:
        """
        Initialize ACDC discovery engine.

        Parameters
        ----------
        model     : HookedTransformer instance
        threshold : KL threshold τ (default 0.10 from Conmy et al.)
        verbose   : Enable debug output
        """
        self.model = model
        self.threshold = threshold
        self.verbose = verbose
        self._n_layers = model.cfg.n_layers
        self._n_heads = model.cfg.n_heads
        self._d_model = model.cfg.d_model

    def discover(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
    ) -> ACDCResult:
        """
        Discover the minimal faithful circuit.

        Parameters
        ----------
        clean_tokens     : Clean input tokens [1, seq_len]
        corrupted_tokens : Corrupted input tokens [1, seq_len]

        Returns
        -------
        ACDCResult with discovered circuit and faithfulness metrics
        """
        # Step 1: Collect caches.
        # names_filter restricts caching to hook_z, hook_resid_pre, and
        # hook_mlp_out only — the three hook types ACDC actually uses.
        # Without this filter, run_with_cache stores ALL TransformerLens hook
        # outputs (>50 per layer), causing OOM on large models.
        # NOTE: hook_result is intentionally excluded because its per-head
        # d_model slices are 32× larger than hook_z.  We reconstruct the
        # per-head residual-stream contribution via hook_z @ W_O instead.
        _ACDC_HOOKS = lambda name: (  # noqa: E731
            "hook_z" in name or "hook_resid_pre" in name or "hook_mlp_out" in name
        )

        # Warn on edge-count explosion before any computation.
        n_edges_estimate = self._n_layers * (self._n_layers - 1) // 2 * (self._n_heads + 1) ** 2
        if n_edges_estimate > 100_000:
            logger.warning(
                "ACDC: this model has ~%d candidate edges (%d layers × %d heads). "
                "Full ACDC may take many hours. Consider using a smaller model or "
                "restricting layers via the GlassboxV2.analyze() MFC algorithm instead.",
                n_edges_estimate, self._n_layers, self._n_heads,
            )

        if self.verbose:
            logger.info("ACDC: collecting clean cache")
        with torch.no_grad():
            clean_logits, clean_cache = self.model.run_with_cache(
                clean_tokens, names_filter=_ACDC_HOOKS
            )

        if self.verbose:
            logger.info("ACDC: collecting corrupted cache")
        with torch.no_grad():
            _, corr_cache = self.model.run_with_cache(
                corrupted_tokens, names_filter=_ACDC_HOOKS
            )

        # Step 2: Get clean logits and logprobs
        clean_logprobs = self._get_clean_logprobs(clean_tokens)

        # Step 3: Build all possible edges in topological order
        all_edges = self._build_all_edges()
        if self.verbose:
            logger.info(f"ACDC: built {len(all_edges)} candidate edges")

        # Step 4: Test edges in topological order
        pruning_kl_scores: Dict[ACDCEdge, float] = {}
        circuit_edges: Set[ACDCEdge] = set()

        for i, edge in enumerate(all_edges):
            if self.verbose and i % 50 == 0:
                logger.info(f"ACDC: testing edge {i}/{len(all_edges)}")

            # Test this edge: if KL ≥ threshold, it's necessary.
            # Correct ACDC semantics: patch only previously PRUNED edges + current edge.
            # Retained edges must remain active so we measure marginal importance correctly.
            pruned_before_i = [e for e in all_edges[:i] if e not in circuit_edges]
            kl_score = self._test_edge_kl(
                edge,
                pruned_so_far=pruned_before_i,
                clean_tokens=clean_tokens,
                clean_cache=clean_cache,
                corr_cache=corr_cache,
                clean_logprobs=clean_logprobs,
            )
            pruning_kl_scores[edge] = kl_score

            if kl_score >= self.threshold:
                circuit_edges.add(edge)
                if self.verbose:
                    logger.debug(f"  → RETAINED: {edge} (kl={kl_score:.4f})")
            else:
                if self.verbose:
                    logger.debug(f"  → PRUNED: {edge} (kl={kl_score:.4f})")

        # Step 5: Evaluate circuit faithfulness
        circuit = ACDCCircuit(circuit_edges, self._n_layers, self._n_heads)
        kl_circuit = self._circuit_kl(
            circuit, clean_tokens, clean_cache, corr_cache, clean_logprobs
        )

        faithful = kl_circuit < ACDC_FAITHFULNESS_THRESHOLD

        if self.verbose:
            logger.info(f"ACDC: discovered circuit with {len(circuit_edges)} edges")
            logger.info(f"ACDC: circuit KL = {kl_circuit:.4f} (faithful={faithful})")

        return ACDCResult(
            circuit=circuit,
            kl_circuit=float(kl_circuit),
            n_edges_total=len(all_edges),
            n_edges_retained=len(circuit_edges),
            n_edges_pruned=len(all_edges) - len(circuit_edges),
            threshold=self.threshold,
            faithful=faithful,
            pruning_kl_scores=pruning_kl_scores,
        )

    def _get_clean_logprobs(self, clean_tokens: torch.Tensor) -> np.ndarray:
        """
        Get log-softmax of clean output at last position.

        Returns
        -------
        np.ndarray of shape [vocab_size] with log-probabilities
        """
        with torch.no_grad():
            logits = self.model(clean_tokens)
        last_logits = logits[0, -1, :]  # [vocab_size]
        logprobs = torch.log_softmax(last_logits, dim=-1)
        return logprobs.cpu().float().numpy()

    def _build_all_edges(self) -> List[ACDCEdge]:
        """
        Build all possible edges in topological order.

        Senders: attn heads (layers 0..L-1) + MLP (layers 0..L-1)
        Receivers: attn heads (layers 1..L-1) + MLP (layers 0..L-1)
        Constraint: receiver_layer > sender_layer

        Returns edges sorted by:
        1. receiver_layer (ascending)
        2. receiver_type ('attn' before 'mlp')
        3. receiver_head (ascending)
        Then by sender_layer, sender_type, sender_head
        """
        edges: List[ACDCEdge] = []

        # Iterate over receivers in topological order
        for r_layer in range(self._n_layers):
            # Attn heads as receivers
            for r_head in range(self._n_heads):
                receiver = (r_layer, "attn", r_head)
                # All senders from earlier layers
                for s_layer in range(r_layer):
                    # Attn senders
                    for s_head in range(self._n_heads):
                        sender = (s_layer, "attn", s_head)
                        edges.append(ACDCEdge(sender, receiver))
                    # MLP sender
                    sender = (s_layer, "mlp", 0)
                    edges.append(ACDCEdge(sender, receiver))

            # MLP as receiver
            receiver = (r_layer, "mlp", 0)
            for s_layer in range(r_layer):
                # Attn senders
                for s_head in range(self._n_heads):
                    sender = (s_layer, "attn", s_head)
                    edges.append(ACDCEdge(sender, receiver))
                # MLP sender
                sender = (s_layer, "mlp", 0)
                edges.append(ACDCEdge(sender, receiver))

        return edges

    def _test_edge_kl(
        self,
        edge: ACDCEdge,
        pruned_so_far: List[ACDCEdge],
        clean_tokens: torch.Tensor,
        clean_cache: object,
        corr_cache: object,
        clean_logprobs: np.ndarray,
    ) -> float:
        """
        Test if edge is causally necessary by measuring KL when patched.

        Patches the edge (and all previously pruned edges) with corrupted
        activations, runs forward pass, and measures KL divergence.

        Returns
        -------
        float: KL(clean_logprobs || patched_logprobs)
        """
        edges_to_patch = set(pruned_so_far) | {edge}
        hooks = self._build_fwd_hooks(edges_to_patch, clean_cache, corr_cache)

        with torch.no_grad():
            patched_logits = self.model.run_with_hooks(
                clean_tokens, fwd_hooks=hooks
            )

        patched_logprobs = torch.log_softmax(patched_logits[0, -1, :], dim=-1)
        patched_logprobs = patched_logprobs.cpu().float().numpy()

        # KL(clean || patched) = Σ P(x) · [log P(x) - log Q(x)]
        # Use float64 to avoid precision loss with large vocabularies (e.g. Llama-3: 128k tokens).
        # float32 accumulation over 128k terms can lose ~3 digits of precision.
        p = np.exp(clean_logprobs.astype(np.float64))
        log_ratio = clean_logprobs.astype(np.float64) - patched_logprobs.astype(np.float64)
        kl = float(np.sum(p * log_ratio))
        kl = max(0.0, kl)  # Guard against tiny negative values from floating-point rounding
        return kl

    def _circuit_kl(
        self,
        circuit: ACDCCircuit,
        clean_tokens: torch.Tensor,
        clean_cache: object,
        corr_cache: object,
        clean_logprobs: np.ndarray,
    ) -> float:
        """
        Measure faithfulness of circuit: KL when all non-circuit edges are patched.

        Returns
        -------
        float: KL(clean || circuit_output)
        """
        # Inverse: patch all edges NOT in circuit
        all_edges = self._build_all_edges()
        non_circuit_edges = [e for e in all_edges if e not in circuit.edges]

        hooks = self._build_fwd_hooks(set(non_circuit_edges), clean_cache, corr_cache)

        with torch.no_grad():
            circuit_logits = self.model.run_with_hooks(
                clean_tokens, fwd_hooks=hooks
            )

        circuit_logprobs = torch.log_softmax(circuit_logits[0, -1, :], dim=-1)
        circuit_logprobs = circuit_logprobs.cpu().float().numpy()

        # KL(clean || circuit) — use float64 for large-vocabulary precision
        p = np.exp(clean_logprobs.astype(np.float64))
        log_ratio = clean_logprobs.astype(np.float64) - circuit_logprobs.astype(np.float64)
        kl = float(np.sum(p * log_ratio))
        kl = max(0.0, kl)
        return kl

    def _build_fwd_hooks(
        self,
        edges_to_patch: Set[ACDCEdge],
        clean_cache: object,
        corr_cache: object,
    ) -> List[Tuple[str, Callable]]:
        """
        Build forward hooks to patch specified edges.

        For each receiver layer, creates one hook on hook_resid_pre that
        patches all senders to that layer in a single pass.

        Attn contribution is computed via hook_z @ W_O[layer][head] rather
        than hook_result, because hook_result is excluded from the cache to
        avoid OOM (it is 32× larger than hook_z for models with d_model=4096).

        Parameters
        ----------
        edges_to_patch : Set of ACDCEdge objects to patch
        clean_cache    : Cache from clean forward pass
        corr_cache     : Cache from corrupted forward pass

        Returns
        -------
        List[(hook_name, hook_fn)] for use with model.run_with_hooks()
        """
        # Group edges by receiver layer
        edges_by_receiver_layer: Dict[int, List[ACDCEdge]] = {}
        for edge in edges_to_patch:
            r_layer = edge.receiver[0]
            if r_layer not in edges_by_receiver_layer:
                edges_by_receiver_layer[r_layer] = []
            edges_by_receiver_layer[r_layer].append(edge)

        hooks: List[Tuple[str, Callable]] = []

        for r_layer, edges in edges_by_receiver_layer.items():
            # Create hook for this receiver layer.
            # Pass model explicitly to avoid closure-capture issues.
            def make_hook(r_layer_val, edges_val, clean_cache_val, corr_cache_val, model_ref):
                def hook(resid_pre, hook=None):  # noqa: ARG001
                    patched_resid = resid_pre.clone()
                    orig_dtype = patched_resid.dtype

                    for edge in edges_val:
                        s_layer, s_type, s_head = edge.sender

                        # Get per-head sender contribution to residual stream.
                        if s_type == "attn":
                            # hook_result is NOT cached (too large).
                            # Reconstruct per-head residual contribution as:
                            #   head_out[h] = hook_z[:, :, h, :] @ W_O[h]
                            # where W_O has shape [n_heads, d_head, d_model].
                            hook_z_key = f"blocks.{s_layer}.attn.hook_z"
                            # hook_z: [batch, seq, n_heads, d_head]
                            clean_z = clean_cache_val[hook_z_key][
                                :, :, s_head, :
                            ].float()  # [batch, seq, d_head]
                            corr_z = corr_cache_val[hook_z_key][
                                :, :, s_head, :
                            ].float()
                            W_O_h = model_ref.blocks[s_layer].attn.W_O[s_head].detach().float()  # [d_head, d_model]
                            # einsum "bsd,dm->bsm" for both clean and corrupted
                            clean_contrib = torch.einsum("bsd,dm->bsm", clean_z, W_O_h)
                            corr_contrib  = torch.einsum("bsd,dm->bsm", corr_z, W_O_h)
                        else:  # mlp
                            mlp_key = f"blocks.{s_layer}.hook_mlp_out"
                            clean_contrib = clean_cache_val[mlp_key].float()
                            corr_contrib  = corr_cache_val[mlp_key].float()

                        # Swap clean → corrupted contribution in residual stream.
                        # Handle potential sequence-length mismatch (name-swap fallback).
                        delta = corr_contrib - clean_contrib  # [batch, seq, d_model]
                        min_seq = min(patched_resid.shape[1], delta.shape[1])
                        patched_resid[:, :min_seq, :] = (
                            patched_resid[:, :min_seq, :].float() + delta[:, :min_seq, :]
                        ).to(orig_dtype)

                    return patched_resid

                return hook

            hook_name = f"blocks.{r_layer}.hook_resid_pre"
            hooks.append((hook_name, make_hook(r_layer, edges, clean_cache, corr_cache, self.model)))

        return hooks
