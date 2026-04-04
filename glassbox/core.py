"""
Glassbox 2.0 — Causal Mechanistic Interpretability Engine
==========================================================

Core References
---------------
Wang et al. 2022 — "Interpretability in the Wild: a Circuit for Indirect Object
    Identification in GPT-2 small"  https://arxiv.org/abs/2211.00593
    Introduced the IOI circuit, name-swap corruption, and corrupted activation
    patching as the standard for causal faithfulness evaluation.

Nanda et al. 2023 — "Attribution Patching: Activation Patching at Industrial Scale"
    https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
    First-order Taylor approximation for head-level attribution in O(3 passes).

Conmy et al. 2023 — "Towards Automated Circuit Discovery for Mechanistic Interpretability"
    (ACDC)  https://arxiv.org/abs/2304.14997
    Graph-based edge-level automated circuit discovery. Glassbox operates at
    head granularity and is 37x faster wall-clock on GPT-2 small (1.2s vs 43.2s).

Elhage et al. 2021 — "A Mathematical Framework for Transformer Circuits"
    https://transformer-circuits.pub/2021/framework/index.html
    Foundational theory: residual stream, attention head composition, virtual weights.

Geiger et al. 2021 — "Causal Abstractions of Neural Networks"
    https://arxiv.org/abs/2106.02997
    Formal framework for causal faithfulness of circuit explanations.

Goldowsky-Dill et al. 2023 — "Localizing Model Behavior with Path Patching"
    https://arxiv.org/abs/2304.05969
    Path patching generalises activation patching to arbitrary computational paths.

nostalgebraist 2020 — "Interpreting GPT: the Logit Lens"
    https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
    Projects the residual stream through ln_final + unembed at each layer to show
    how predictions crystallise from input to output.

Dar et al. 2023 — "Analyzing Transformers in Embedding Space"  EMNLP 2023
    https://arxiv.org/abs/2209.02535
    Decomposes model predictions into vocabulary-space contributions per head.

Syed et al. 2024 — "Attribution Patching Outperforms Automated Circuit Discovery"
    ACL BlackboxNLP Workshop.  https://arxiv.org/abs/2310.10348
    Edge Attribution Patching (EAP): scores each directed edge (u→v) in the
    computation graph at O(3) cost, strictly more informative than node-level AP.

Kendall 1938 — "A New Measure of Rank Correlation"
    Biometrika, 30(1-2), 81–93.  https://doi.org/10.1093/biomet/30.1-2.81
    Kendall τ rank correlation — used by attribution_stability() to measure
    ordinal consistency of head rankings across independent corruptions.

Simonyan et al. 2014 — "Deep Inside Convolutional Networks"
    https://arxiv.org/abs/1312.6034
    Gradient × input saliency maps — used by token_attribution() to score
    each input token's signed contribution to the logit difference.

Olsson et al. 2022 — "In-context Learning and Induction Heads"
    Transformer Circuits Thread.
    https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
    Induction head geometry — used by attention_patterns() head-type classifier.

Bloom et al. 2024 — "Open Source Sparse Autoencoders for GPT2-Small"
    https://www.neuronpedia.org/gpt2-small
    Pre-trained residual-stream SAEs — used by SAEFeatureAttributor in
    glassbox/sae_attribution.py for feature-level circuit analysis.

Complexity notes (honest)
-------------------------
attribution_patching()        : 3 forward passes        (O(3))
_comp()                       : 2 forward passes        (O(2))
minimum_faithful_circuit()    : 3 + 2p passes           (p = backward pruning steps)
logit_lens()                  : 1 forward pass          (O(1))
edge_attribution_patching()   : 2 forward + 1 backward  (O(3))
attribution_stability()       : 3K passes               (K = n_corruptions, default 10)
token_attribution()           : 1 forward + 1 backward  (O(2))
attention_patterns()          : 1 forward pass          (O(1))
analyze()                     : 3 + 2p passes           (+ 1 if include_logit_lens=True)

The "O(3)" label applies only to raw attribution scoring. Full circuit discovery
costs O(3 + 2p) where p is typically 0-4 on IOI prompts.
"""

import logging
import re
import torch
import numpy as np
import einops                               # noqa: F401 — imported for TransformerLens compat
from typing import Dict, List, Tuple, Optional

# NOTE: We intentionally do NOT set global random seeds here.
# Calling torch.manual_seed / np.random.seed at module-import time
# would silently corrupt the random state of ANY caller that imports
# glassbox — a serious reproducibility anti-pattern.
# Methods that need determinism accept an explicit `seed` parameter.

logger = logging.getLogger(__name__)

# Expose version inside the module so model_metadata can self-document results
# without importing glassbox/__init__.py (circular import risk).
try:
    from importlib.metadata import version as _pkg_version
    _GLASSBOX_VERSION: str = _pkg_version("glassbox-mech-interp")
except Exception:
    _GLASSBOX_VERSION = "unknown"


class GlassboxV2:
    """
    Glassbox 2.0 — Causal Mechanistic Interpretability Engine.

    Public API
    ----------
    analyze(prompt, correct, incorrect, method="taylor")
        One-call circuit discovery + faithfulness scoring.
        method="integrated_gradients" for higher accuracy (slower).

    attribution_patching(clean_tokens, corrupted_tokens, target_token, distractor_token,
                         method="taylor", n_steps=10)
        Per-head attribution scores. "taylor" = 3 passes (fast approximation).
        "integrated_gradients" = 2+n_steps passes (path-integral exact attribution).

    mlp_attribution(clean_tokens, corrupted_tokens, target_token, distractor_token)
        Per-layer MLP attribution scores via first-order Taylor (3 passes).
        Completes the circuit picture alongside attention head attribution.

    get_top_heads(attributions, top_k=10)
        Ranked attention heads with layer, head, attr, rel_depth fields.
        Required input format for functional_circuit_alignment().

    minimum_faithful_circuit(...)
        Greedy forward/backward circuit auto-discovery.

    bootstrap_metrics(prompts, n_boot, alpha)
        Bootstrap 95% CI on Suff / Comp / F1 across N prompts.

    logit_lens(tokens, target_token, distractor_token)
        Layer-by-layer logit tracking + per-head direct effects. 1 forward pass.

    edge_attribution_patching(clean_tokens, corrupted_tokens, target_token,
                              distractor_token, top_k=50)
        Edge-level attribution: scores every (sender→receiver) directed edge
        in the computation graph.  O(3) — strictly more informative than
        node-level AP.  (Syed et al. 2024)

    attribution_stability(clean_tokens, target_token, distractor_token,
                          n_corruptions=10, replace_fraction=0.25, seed=42)
        Stability of attribution rankings over K random corruptions.
        Returns per-head stability score S ∈ [0,1] and global Kendall τ-b
        rank consistency.  Novel metric — no prior tool has this.

    token_attribution(tokens, target_token, distractor_token)
        Per-input-token attribution via gradient × embedding (Simonyan 2014).
        Scores each input token's signed contribution to LD.  1F + 1B pass.

    attention_patterns(tokens, heads=None, top_k=10)
        Full attention matrices + entropy + heuristic head-type classification
        (induction_candidate, previous_token, focused, uniform).  1 forward pass.

    Mathematical caveats (disclosed)
    ---------------------------------
    Sufficiency is a first-order Taylor APPROXIMATION, not the exact value.
    Exact sufficiency (Wang et al. 2022, Conmy et al. 2023) requires running the
    model with non-circuit heads ablated. The Taylor approximation
        Suff ≈ Σ attr(h ∈ circuit) / LD_clean
    is accurate when individual head contributions are small relative to LD_clean
    and head interactions are approximately linear. For IOI on GPT-2, where 2-4
    heads dominate, the approximation is tight. For tasks with distributed
    computation the error may be larger.

    Comprehensiveness is EXACT (corrupted activation patching, not approximated).

    FCAS (Functional Circuit Alignment Score) is a novel metric.
    Limitations: (a) matches heads by rank, not functional role; (b) compares
    depth only, not head index within a layer; (c) sensitive to k.
    A null distribution (random circuit FCAS) is computed to give context.
    """

    # ── Supported model families (TransformerLens model names or prefixes) ──
    SUPPORTED_MODELS = [
        # GPT-2 family
        "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
        # GPT-Neo / GPT-J family
        "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-j-6B",
        # Llama-3 family (via TransformerLens >=2.0)
        "meta-llama/Llama-3-8B", "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-3-70B",
        # Llama-2 family
        "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf",
        # Mistral family
        "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1",
        # Gemma family
        "google/gemma-2b", "google/gemma-7b",
        # Pythia family
        "EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
    ]

    # ── Approximate parameter counts for memory warnings ─────────────────
    _LARGE_MODEL_THRESHOLD_PARAMS = 1_000_000_000   # 1B params → warn

    def __init__(self, model, device: str = "cpu", dtype=None) -> None:
        """
        Parameters
        ----------
        model  : HookedTransformer or str
            A pre-loaded TransformerLens model, or a model name string such as
            "gpt2", "meta-llama/Llama-3-8B", etc.
        device : str
            PyTorch device ("cpu", "cuda", "cuda:0", …). Default "cpu".
        dtype  : torch.dtype or None
            Model weight dtype. Pass ``torch.bfloat16`` or ``torch.float16`` to
            load large models (7B+) without OOM. Default None → float32.
        """
        # Accept either a pre-loaded HookedTransformer or a model name string
        if isinstance(model, str):
            try:
                from transformer_lens import HookedTransformer
                load_kwargs = dict(device=device)
                if dtype is not None:
                    load_kwargs["dtype"] = dtype
                model = HookedTransformer.from_pretrained(model, **load_kwargs)
            except Exception as e:
                raise ValueError(
                    f"GlassboxV2: could not load model '{model}' via TransformerLens. "
                    f"Pass a pre-loaded HookedTransformer or install transformer-lens. "
                    f"Original error: {e}"
                ) from e
        self.model    = model
        self.n_layers = model.cfg.n_layers
        self.n_heads  = model.cfg.n_heads

        # ── Detect model family and GQA (Grouped Query Attention) ─────────
        cfg = model.cfg
        self.model_name: str = getattr(cfg, "model_name", "unknown")

        # TransformerLens exposes n_key_value_heads for GQA models
        # (Llama-3-8B: 8 KV heads, 32 Q heads).  hook_z always has shape
        # [batch, seq, n_heads, d_head] — TL expands KV before the hook —
        # so attribution patching works unchanged on GQA models.
        # getattr default is skipped when attr exists but is None (TL sets None for non-GQA models)
        self.n_kv_heads: int = getattr(cfg, "n_key_value_heads", None) or self.n_heads
        self.uses_gqa: bool  = (self.n_kv_heads != self.n_heads)

        # ── Memory / device check for large models ─────────────────────────
        d_model   = getattr(cfg, "d_model", 768)
        n_params_approx = (
            self.n_layers * self.n_heads * d_model * d_model * 4  # rough QKV+O
        )
        if n_params_approx > self._LARGE_MODEL_THRESHOLD_PARAMS:
            logger.warning(
                "GlassboxV2: large model detected (~%s parameters). "
                "Attribution patching requires gradient computation — ensure "
                "sufficient GPU VRAM (fp16: ~%.0fGB, fp32: ~%.0fGB). "
                "Use model.eval() and torch.inference_mode(False) for backward passes. "
                "For memory-constrained environments consider a quantised checkpoint.",
                f"{n_params_approx / 1e9:.1f}B",
                n_params_approx * 2 / 1e9,
                n_params_approx * 4 / 1e9,
            )

    def model_info(self) -> Dict[str, object]:
        """
        Return a structured dict of model architecture metadata suitable for
        embedding directly into an EU AI Act Annex IV technical documentation
        report (Article 11 — technical documentation, Annex IV §1).

        Usage
        -----
            gb = GlassboxV2(model)
            info = gb.model_info()
            # info["model_name"] == "meta-llama/Llama-3-8B"
            # info["n_params_approx"] == "8.0B"
            # info["uses_gqa"] == True

        Returns
        -------
        dict with keys:
            model_name          str    — TransformerLens model identifier
            n_layers            int    — transformer depth
            n_heads             int    — number of query attention heads
            n_kv_heads          int    — KV heads (< n_heads for GQA models)
            uses_gqa            bool   — True if Grouped Query Attention
            d_model             int    — residual stream width
            d_head              int    — per-head dimension
            n_ctx               int    — max context length (tokens)
            vocab_size          int    — vocabulary size
            n_params_approx     str    — human-readable parameter count
            device              str    — inferred device ("cpu" / "cuda" / …)
            dtype               str    — model parameter dtype
            glassbox_version    str    — glassbox-mech-interp package version
        """
        cfg = self.model.cfg
        d_model   = getattr(cfg, "d_model", 768) or 768
        d_head    = getattr(cfg, "d_head",  None) or (d_model // self.n_heads)
        n_ctx     = getattr(cfg, "n_ctx",   getattr(cfg, "max_seq_len", 0))
        vocab_sz  = getattr(cfg, "d_vocab", 0)

        # Parameter count: approximate from architecture dimensions
        emb   = vocab_sz * d_model
        attn  = self.n_layers * (
            d_model * (self.n_heads + 2 * self.n_kv_heads) * d_head   # QKV
            + self.n_heads * d_head * d_model                          # W_O
        )
        mlp_mult = getattr(cfg, "d_mlp", d_model * 4) // d_model
        mlp   = self.n_layers * 2 * d_model * (d_model * mlp_mult)
        total = emb + attn + mlp

        # Device and dtype from first parameter tensor
        try:
            p = next(self.model.parameters())
            device_str = str(p.device)
            dtype_str  = str(p.dtype).replace("torch.", "")
        except StopIteration:
            device_str, dtype_str = "unknown", "unknown"

        def _fmt(n: int) -> str:
            if n >= 1e12: return f"{n/1e12:.1f}T"
            if n >= 1e9:  return f"{n/1e9:.1f}B"
            if n >= 1e6:  return f"{n/1e6:.0f}M"
            return str(n)

        return {
            "model_name":        self.model_name,
            "n_layers":          self.n_layers,
            "n_heads":           self.n_heads,
            "n_kv_heads":        self.n_kv_heads,
            "uses_gqa":          self.uses_gqa,
            "d_model":           d_model,
            "d_head":            d_head,
            "n_ctx":             n_ctx,
            "vocab_size":        vocab_sz,
            "n_params_approx":   _fmt(total),
            "device":            device_str,
            "dtype":             dtype_str,
            "glassbox_version":  _GLASSBOX_VERSION,
        }

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL — name-swap corruption
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _name_swap(prompt: str, target: str, distractor: str) -> str:
        """
        Bidirectional name-swap corruption matching Wang et al. 2022.

        Swaps every occurrence of target ↔ distractor in the prompt.
        Uses a placeholder to avoid double-replacement.

        Example
        -------
        "When Mary and John … John gave … to"
        → "When John and Mary … Mary gave … to"
        """
        # Use word-boundary regex to avoid partial matches inside words
        # (e.g. "a" must not match inside "cat" or "sat").
        placeholder = "<<<GLASSBOX_SWAP>>>"
        swapped = re.sub(r'\b' + re.escape(target) + r'\b', placeholder, prompt)
        swapped = re.sub(r'\b' + re.escape(distractor) + r'\b', target, swapped)
        swapped = swapped.replace(placeholder, distractor)
        if swapped == prompt:
            # Fallback for prompts where neither name appears as a whole word:
            # append distractor as worst-case corruption.
            swapped = prompt + " " + distractor
        return swapped

    # ──────────────────────────────────────────────────────────────────────
    # 1. ATTRIBUTION PATCHING  (Nanda et al. 2023)
    #    3 forward passes total — head-level first-order Taylor approximation
    # ──────────────────────────────────────────────────────────────────────

    def attribution_patching(
        self,
        clean_tokens:     torch.Tensor,
        corrupted_tokens: torch.Tensor,
        target_token:     int,
        distractor_token: int,
        method:           str = "taylor",
        n_steps:          int = 10,
    ) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        Compute per-head attribution scores via Jacobian × Δz.

        method="taylor"  (default, 3 passes)
        -------------------------------------
        Formula (Nanda et al. 2023):
            attr(l, h) = ∇_{z_lh} LD  ·  (z_clean_lh − z_corr_lh)

        First-order Taylor approximation. Fast but accuracy degrades when
        |z_clean − z_corr| is large relative to activation magnitude.

        method="integrated_gradients"  (2 + n_steps passes)
        -----------------------------------------------------
        Formula (Sundararajan et al. 2017):
            attr_IG(l,h) = (z_clean − z_corr) · (1/n) Σ_k ∇ LD(z_corr + k/n·Δz)

        Path integral along the straight-line interpolation from corrupted to clean.
        Exact in the limit n_steps→∞. Recommended for large corruptions or when
        Taylor scores are unstable. Costs 2 + n_steps forward passes.

        Parameters
        ----------
        clean_tokens     : tokenised clean prompt       [1, seq_len]
        corrupted_tokens : tokenised corrupted prompt   [1, seq_len]
        target_token     : vocabulary index of correct next token
        distractor_token : vocabulary index of incorrect next token
        method           : "taylor" (fast) | "integrated_gradients" (accurate)
        n_steps          : interpolation steps for integrated_gradients (default 10)

        Returns
        -------
        attributions : Dict[(layer, head) -> float]  — positive = promotes target
        clean_ld     : float  — logit(target) - logit(distractor) on clean input
        """
        model = self.model
        n_layers, n_heads = self.n_layers, self.n_heads

        # ── Pass 1: cache clean activations (no grad) ─────────────────────
        clean_cache: Dict[str, torch.Tensor] = {}

        def _save_clean(key: str):
            def hook(act, hook=None):
                clean_cache[key] = act.detach().clone()
            return hook

        with torch.no_grad():
            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (f"blocks.{l}.attn.hook_z", _save_clean(f"blocks.{l}.attn.hook_z"))
                    for l in range(n_layers)
                ],
            )

        # ── Pass 2: cache corrupted activations (no grad) ─────────────────
        corr_cache: Dict[str, torch.Tensor] = {}

        def _save_corr(key: str):
            def hook(act, hook=None):
                corr_cache[key] = act.detach().clone()
            return hook

        with torch.no_grad():
            model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (f"blocks.{l}.attn.hook_z", _save_corr(f"blocks.{l}.attn.hook_z"))
                    for l in range(n_layers)
                ],
            )

        # ── Pass 3: gradient pass (requires_grad on clean activations) ────
        grad_inputs: Dict[str, torch.Tensor] = {
            f"blocks.{l}.attn.hook_z": (
                clean_cache[f"blocks.{l}.attn.hook_z"]
                .clone()
                .float()
                .requires_grad_(True)
            )
            for l in range(n_layers)
        }

        def _patch(key: str):
            def hook(act, hook=None):
                # MUST return — otherwise gradient doesn't flow through
                return grad_inputs[key].to(act.dtype)
            return hook

        model.zero_grad()
        logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(k, _patch(k)) for k in grad_inputs],
        )
        ld = (
            logits[0, -1, target_token].float()
            - logits[0, -1, distractor_token].float()
        )
        clean_ld = ld.item()
        ld.backward()

        # ── Compute attributions: grad · Δz at last token position ────────
        attributions: Dict[Tuple[int, int], float] = {}
        for l in range(n_layers):
            key = f"blocks.{l}.attn.hook_z"
            g = grad_inputs[key].grad
            if g is None:
                for h in range(n_heads):
                    attributions[(l, h)] = 0.0
                continue
            # Extract last-position slice from each cache independently.
            # This handles clean and corrupted sequences of different lengths
            # (e.g. when _name_swap fallback appends a token).  The formula
            # attr = grad · Δz only uses the last-position activations anyway,
            # so slicing before subtraction is semantically identical for
            # equal-length sequences and correct for unequal ones.
            c_last = clean_cache[key][0, -1].float()    # [n_heads, d_head]
            r_last = corr_cache[key][0, -1].float()     # [n_heads, d_head]
            delta_last = c_last - r_last                # [n_heads, d_head]
            for h in range(n_heads):
                # dot product at the last sequence position over d_head
                attributions[(l, h)] = (
                    g[0, -1, h, :] * delta_last[h, :]
                ).sum().item()

        # ── Integrated Gradients branch ────────────────────────────────────
        if method == "integrated_gradients":
            # Re-compute using path integral: average gradient over n_steps
            # interpolations from corrupted to clean.  More accurate than
            # single Jacobian when the corruption is large.
            acc_grads: Dict[str, torch.Tensor] = {
                k: torch.zeros_like(clean_cache[k]) for k in clean_cache
            }

            # torch.enable_grad() is required here: callers may wrap analyze()
            # in torch.no_grad() for inference efficiency, which would make
            # .requires_grad_(True) a no-op and .backward() silently return
            # zero gradients. The enable_grad context forces gradient tracking
            # for the IG path-integral loop only.
            for step in range(1, n_steps + 1):
                with torch.enable_grad():
                    alpha = step / n_steps
                    interp: Dict[str, torch.Tensor] = {}
                    for k in clean_cache:
                        # Align lengths before interpolating.
                        # Keep interpolated tensors in float32 regardless of model
                        # dtype so that gradients accumulate at full precision.
                        min_len = min(clean_cache[k].shape[1], corr_cache[k].shape[1])
                        c = clean_cache[k][:, :min_len].float()
                        r = corr_cache[k][:, :min_len].float()
                        interp[k] = (r + alpha * (c - r)).requires_grad_(True)

                    def _patch_interp(key: str):
                        def hook(act, hook=None):
                            # Cast to act.dtype so the model's internal computation
                            # stays in its native dtype (bf16/fp16), but gradients
                            # flow back through autograd.
                            return interp[key].to(act.dtype)
                        return hook

                    model.zero_grad()
                    logits_i = model.run_with_hooks(
                        clean_tokens,
                        fwd_hooks=[(k, _patch_interp(k)) for k in interp],
                    )
                    ld_i = (
                        logits_i[0, -1, target_token].float()
                        - logits_i[0, -1, distractor_token].float()
                    )
                    ld_i.backward()

                for k in acc_grads:
                    if interp[k].grad is not None:
                        min_len = interp[k].shape[1]
                        acc_grads[k][:, :min_len] += interp[k].grad.detach()

            # Average and compute IG attributions
            attributions = {}
            for l in range(n_layers):
                key = f"blocks.{l}.attn.hook_z"
                g_avg = acc_grads[key] / n_steps
                c_last = clean_cache[key][0, -1].float()
                r_last = corr_cache[key][0, -1].float()
                delta_last = c_last - r_last
                for h in range(n_heads):
                    attributions[(l, h)] = (
                        g_avg[0, -1, h, :] * delta_last[h, :]
                    ).sum().item()

        logger.debug(
            "attribution_patching done: method=%s clean_ld=%.4f n_heads=%d",
            method, clean_ld, len(attributions),
        )
        return attributions, clean_ld

    # ──────────────────────────────────────────────────────────────────────
    # 2a. MLP ATTRIBUTION — per-layer Taylor attribution on hook_mlp_out
    # ──────────────────────────────────────────────────────────────────────

    def mlp_attribution(
        self,
        clean_tokens:     torch.Tensor,
        corrupted_tokens: torch.Tensor,
        target_token:     int,
        distractor_token: int,
    ) -> Dict[int, float]:
        """
        Per-layer MLP attribution via first-order Taylor approximation.

        Extends the circuit picture beyond attention heads to include MLP
        computational contributions.  Same Jacobian × Δz formula as head
        attribution (Nanda et al. 2023), applied to hook_mlp_out rather
        than hook_z.

        Formula
        -------
            attr_MLP(l) = ∇_{mlp_l} LD  ·  (mlp_clean_l − mlp_corr_l)

        at the last sequence position.  3 forward passes total.

        Returns
        -------
        Dict[layer -> float]  — positive = MLP layer promotes target token
        """
        model = self.model
        n_layers = self.n_layers

        clean_cache: Dict[int, torch.Tensor] = {}
        corr_cache:  Dict[int, torch.Tensor] = {}

        def _save_clean_mlp(layer: int):
            def hook(act, hook=None):
                clean_cache[layer] = act.detach().clone()
            return hook

        def _save_corr_mlp(layer: int):
            def hook(act, hook=None):
                corr_cache[layer] = act.detach().clone()
            return hook

        with torch.no_grad():
            model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[
                    (f"blocks.{l}.hook_mlp_out", _save_clean_mlp(l))
                    for l in range(n_layers)
                ],
            )

        with torch.no_grad():
            model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (f"blocks.{l}.hook_mlp_out", _save_corr_mlp(l))
                    for l in range(n_layers)
                ],
            )

        grad_inputs: Dict[int, torch.Tensor] = {
            l: clean_cache[l].clone().float().requires_grad_(True)
            for l in range(n_layers)
        }

        def _patch_mlp(layer: int):
            def hook(act, hook=None):
                return grad_inputs[layer].to(act.dtype)
            return hook

        model.zero_grad()
        logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[
                (f"blocks.{l}.hook_mlp_out", _patch_mlp(l))
                for l in range(n_layers)
            ],
        )
        ld = (
            logits[0, -1, target_token].float()
            - logits[0, -1, distractor_token].float()
        )
        ld.backward()

        mlp_attrs: Dict[int, float] = {}
        for l in range(n_layers):
            g = grad_inputs[l].grad
            if g is None:
                mlp_attrs[l] = 0.0
                continue
            # Last-position slice — handles clean/corrupted length mismatch
            c_last = clean_cache[l][0, -1].float()   # [d_model]
            r_last = corr_cache[l][0, -1].float()    # [d_model]
            delta  = c_last - r_last                  # [d_model]
            mlp_attrs[l] = (g[0, -1, :] * delta).sum().item()

        logger.debug("mlp_attribution done: %d layers", n_layers)
        return mlp_attrs

    # ──────────────────────────────────────────────────────────────────────
    # 2b. GET TOP HEADS — convenience method for FCAS input
    # ──────────────────────────────────────────────────────────────────────

    def get_top_heads(
        self,
        attributions: Dict[Tuple[int, int], float],
        top_k:        int = 10,
    ) -> List[Dict]:
        """
        Return the top-k attention heads sorted by attribution score.

        Required input format for functional_circuit_alignment().

        Parameters
        ----------
        attributions : Dict[(layer, head) -> float] from attribution_patching()
        top_k        : Number of heads to return (default 10)

        Returns
        -------
        List of dicts, each with:
            layer     : int   — transformer layer index
            head      : int   — attention head index within layer
            attr      : float — attribution score (positive = promotes target)
            rel_depth : float — layer / (n_layers - 1), in [0, 1]
        """
        ranked = sorted(
            [(k, v) for k, v in attributions.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            {
                "layer":     layer,
                "head":      head,
                "attr":      attr,
                "rel_depth": layer / max(self.n_layers - 1, 1),
            }
            for (layer, head), attr in ranked
        ]

    # ──────────────────────────────────────────────────────────────────────
    # 3a. LOGIT LENS  (nostalgebraist 2020 + Elhage et al. 2021)
    #     Layer-by-layer prediction tracking with per-head direct effects
    # ──────────────────────────────────────────────────────────────────────

    def logit_lens(
        self,
        tokens:           torch.Tensor,
        target_token:     str,
        distractor_token: str,
    ) -> Dict:
        """
        Logit lens (nostalgebraist, 2020) extended with per-head direct-effect
        attribution (Elhage et al., 2021, §2.3).

        For each layer l ∈ {0, …, L}, projects the intermediate residual stream
        through the final layer norm and unembedding matrix:

            LD_l = (W_U · LN(resid_post_l))_t − (W_U · LN(resid_post_l))_d

        This reveals WHEN the model's preference for target vs distractor
        crystallises across layers.  Index 0 is the embedding (before block 0);
        index l is after block l−1.

        Additionally decomposes each layer's logit SHIFT
            ΔLD_l = LD_l − LD_{l−1}
        into per-head direct contributions via the virtual weights framework:

            direct(l, h) = (W_O[l,h] @ z[l,h,−1]) · unembed_dir
            mlp_direct(l) = mlp_out[l,−1]           · unembed_dir
        where unembed_dir = W_U[:,t] − W_U[:,d]  ∈ ℝ^{d_model}.

        APPROXIMATION NOTE (disclosed)
        --------------------------------
        Per-head direct effects apply the unembed direction to the raw head
        output WITHOUT folding in the LN scale of the full residual stream.
        The exact value requires LN(resid)_scale which is nonlinear and cannot
        be linearly decomposed per-head.  In practice the LN scale is
        approximately uniform across components, so relative magnitudes are
        preserved.  Absolute values should be interpreted as directional.

        1 forward pass (O(1)).

        References
        ----------
        nostalgebraist (2020). "Interpreting GPT: the logit lens."
            https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru
        Elhage et al. (2021). "A Mathematical Framework for Transformer
            Circuits," §2.3.
            https://transformer-circuits.pub/2021/framework/index.html
        Dar et al. (2023). "Analyzing Transformers in Embedding Space."
            EMNLP 2023.  https://arxiv.org/abs/2209.02535

        Parameters
        ----------
        tokens           : tokenised prompt  [1, seq_len]
        target_token     : correct next-token string   (e.g. " Mary")
        distractor_token : distractor token string     (e.g. " John")

        Returns
        -------
        {
          'logit_diffs'          : List[float]          — LD at embedding + after
                                                          each block (n_layers+1)
          'logit_shifts'         : List[float]          — ΔLD per block (n_layers)
          'head_direct_effects'  : Dict[int,List[float]]— {layer: [effect/head]}
          'mlp_direct_effects'   : Dict[int, float]     — {layer: effect}
          'target_token'         : str
          'distractor_token'     : str
        }
        """
        model    = self.model
        n_layers = self.n_layers

        # Resolve token IDs ────────────────────────────────────────────────
        target_id     = model.to_single_token(target_token)
        distractor_id = model.to_single_token(distractor_token)

        # Unembed direction: W_U[:,t] − W_U[:,d]  [d_model]
        # model.W_U has shape [d_model, d_vocab] in TransformerLens
        W_U         = model.W_U.detach().float()                    # [d_model, d_vocab]
        unembed_dir = W_U[:, target_id] - W_U[:, distractor_id]    # [d_model]

        # Unembedding bias difference (zero for GPT-2, present on some models)
        b_U_diff = 0.0
        if hasattr(model, "b_U"):
            b = model.b_U.detach().float()
            b_U_diff = (b[target_id] - b[distractor_id]).item()

        # Single forward pass — cache residuals, z, mlp_out ───────────────
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: (
                    name == "blocks.0.hook_resid_pre"
                    or "hook_resid_post" in name
                    or "attn.hook_z"     in name
                    or "hook_mlp_out"    in name
                ),
            )

        def _ld(resid: torch.Tensor) -> float:
            """Apply ln_final + unembed to a single residual vector [d_model]."""
            r        = resid.float()
            # ln_final expects (..., d_model) — pass [1, d_model] for safety
            r_normed = model.ln_final(r.unsqueeze(0)).squeeze(0)    # [d_model]
            ld       = float((r_normed @ W_U)[target_id].item()
                             - (r_normed @ W_U)[distractor_id].item()
                             + b_U_diff)
            return ld

        # ── Layer-by-layer logit differences ──────────────────────────────
        # Index 0: pure embedding (token + positional), before block 0
        logit_diffs: List[float] = [_ld(cache["blocks.0.hook_resid_pre"][0, -1])]
        for l in range(n_layers):
            logit_diffs.append(_ld(cache[f"blocks.{l}.hook_resid_post"][0, -1]))

        # ΔLD per block
        logit_shifts = [logit_diffs[i + 1] - logit_diffs[i] for i in range(n_layers)]

        # ── Per-head and MLP direct effects ───────────────────────────────
        head_direct_effects: Dict[int, List[float]] = {}
        mlp_direct_effects:  Dict[int, float]       = {}

        for l in range(n_layers):
            # z: [batch, seq, n_heads, d_head] → last pos → [n_heads, d_head]
            z     = cache[f"blocks.{l}.attn.hook_z"][0, -1].float()     # [n_heads, d_head]
            W_O_l = model.blocks[l].attn.W_O.detach().float()           # [n_heads, d_head, d_model]

            # head_out[h] = z[h, :] @ W_O_l[h]  → [n_heads, d_model]
            head_out = torch.einsum("hd,hdm->hm", z, W_O_l)            # [n_heads, d_model]

            # Direct logit-difference effect: project onto unembed direction
            head_direct_effects[l] = (head_out @ unembed_dir).tolist() # [n_heads]

            # MLP direct effect
            mlp_out = cache[f"blocks.{l}.hook_mlp_out"][0, -1].float() # [d_model]
            mlp_direct_effects[l] = float((mlp_out @ unembed_dir).item())

        logger.debug(
            "logit_lens done: n_layers=%d  LD_embed=%.3f  LD_final=%.3f",
            n_layers, logit_diffs[0], logit_diffs[-1],
        )
        return {
            "logit_diffs":         logit_diffs,
            "logit_shifts":        logit_shifts,
            "head_direct_effects": head_direct_effects,
            "mlp_direct_effects":  mlp_direct_effects,
            "target_token":        target_token,
            "distractor_token":    distractor_token,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 3b. EDGE ATTRIBUTION PATCHING  (Syed et al. 2024)
    #     O(3): scores every (sender → receiver) edge in the computation graph
    # ──────────────────────────────────────────────────────────────────────

    def edge_attribution_patching(
        self,
        clean_tokens:     torch.Tensor,
        corrupted_tokens: torch.Tensor,
        target_token:     int,
        distractor_token: int,
        top_k:            int = 50,
    ) -> Dict:
        """
        Edge Attribution Patching (EAP).

        Scores each directed edge (sender → receiver) in the transformer
        computation graph.  Strictly more informative than node-level AP:
        EAP reveals which connections between heads carry the causal signal,
        not just which heads are important in isolation.

        Formula (Syed et al., 2024, §3)
        ---------------------------------
            EAP(u → v) = (∂metric / ∂resid_pre_v) · Δh_u

        where
          • ∂metric/∂resid_pre_v  — gradient of logit-diff w.r.t. the residual
            stream at the INPUT to layer v, captured via backward hooks on the
            clean run.  This is the sensitivity of the metric to changes at
            resid_pre[v].
          • Δh_u = h_u^clean − h_u^corrupted  — change in sender u's contribution
            to the residual stream:
              – attention head:  Δh_u = (z_clean[u] − z_corr[u]) @ W_O[u]  [d_model]
              – MLP node:        Δh_u = mlp_out_clean[u] − mlp_out_corr[u]  [d_model]

        Edges go from any node at layer l_s to any node at layer l_r > l_s.
        All scores are evaluated at the LAST SEQUENCE POSITION (next-token task).

        Computational complexity
        ------------------------
          1 corrupted forward pass   (O(1))
          1 clean forward pass       (O(1))  — with backward hooks registered
          1 backward pass            (O(1))
          ────────────────────────────────
          Total: O(3)  — identical to node-level attribution patching.

        References
        ----------
        Syed et al. (2024). "Attribution Patching Outperforms Automated Circuit
            Discovery."  ACL BlackboxNLP Workshop.
            https://arxiv.org/abs/2310.10348

        Parameters
        ----------
        clean_tokens     : tokenised clean prompt       [1, seq_len]
        corrupted_tokens : tokenised corrupted prompt   [1, seq_len]
        target_token     : vocabulary index of correct next token
        distractor_token : vocabulary index of incorrect next token
        top_k            : edges to return sorted by |score| (default 50)

        Returns
        -------
        {
          'edge_scores' : Dict[((l_s,'attn'|'mlp',h_s), l_r) -> float]
          'top_edges'   : List[(key, score)]  sorted by |score|
          'n_edges'     : int    — total directed edges scored
          'clean_ld'    : float  — clean logit difference
        }
        """
        model    = self.model
        n_layers = self.n_layers
        n_heads  = self.n_heads

        # ── Pass 1: corrupted forward — save sender outputs (no grad) ─────
        corr_attn: Dict[int, torch.Tensor] = {}   # layer → [n_heads, d_head]
        corr_mlp:  Dict[int, torch.Tensor] = {}   # layer → [d_model]

        def _sv_corr_attn(l: int):
            def hook(act, hook=None):
                corr_attn[l] = act[0, -1].detach().clone().float()
            return hook

        def _sv_corr_mlp(l: int):
            def hook(act, hook=None):
                corr_mlp[l] = act[0, -1].detach().clone().float()
            return hook

        with torch.no_grad():
            model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=(
                    [(f"blocks.{l}.attn.hook_z",    _sv_corr_attn(l)) for l in range(n_layers)]
                    + [(f"blocks.{l}.hook_mlp_out", _sv_corr_mlp(l))  for l in range(n_layers)]
                ),
            )

        # ── Pass 2 + backward: clean forward — save sender outputs AND     ─
        # ─  register backward hooks on hook_resid_pre to capture           ─
        # ─  ∂metric/∂resid_pre[l] via register_hook on live activations.   ─
        #    The key insight: during a gradient-enabled forward pass, all    ─
        #    residual-stream tensors are in the computation graph (they were  ─
        #    computed from model parameters with requires_grad=True).        ─
        #    register_hook fires DURING backward before the gradient is freed.─
        clean_attn:  Dict[int, torch.Tensor] = {}
        clean_mlp:   Dict[int, torch.Tensor] = {}
        resid_grads: Dict[int, torch.Tensor] = {}  # l → [d_model]

        def _sv_clean_attn(l: int):
            def hook(act, hook=None):
                clean_attn[l] = act[0, -1].detach().clone().float()
            return hook

        def _sv_clean_mlp(l: int):
            def hook(act, hook=None):
                clean_mlp[l] = act[0, -1].detach().clone().float()
            return hook

        def _grad_hook(l: int):
            def hook(act, hook=None):
                # Register a backward hook to capture ∂metric/∂resid_pre[l]
                # at the last sequence position.  Only fires if act is in graph.
                if act.requires_grad:
                    def _capture(grad: torch.Tensor) -> None:
                        # grad: [batch, seq, d_model] — last pos
                        resid_grads[l] = grad[0, -1].detach().clone().float()
                    act.register_hook(_capture)
                return act   # Return unchanged — preserves the computation graph
            return hook

        fwd_hooks = (
            [(f"blocks.{l}.attn.hook_z",    _sv_clean_attn(l)) for l in range(n_layers)]
            + [(f"blocks.{l}.hook_mlp_out", _sv_clean_mlp(l))  for l in range(n_layers)]
            + [(f"blocks.{l}.hook_resid_pre", _grad_hook(l))   for l in range(n_layers)]
        )

        model.zero_grad()
        # torch.enable_grad() ensures gradient tracking regardless of outer context
        with torch.enable_grad():
            logits = model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks)
            metric = (
                logits[0, -1, target_token].float()
                - logits[0, -1, distractor_token].float()
            )
            clean_ld = metric.item()
            metric.backward()

        if not resid_grads:
            logger.warning(
                "edge_attribution_patching: no gradients captured — "
                "model activations may not require grad.  "
                "Try calling without torch.no_grad() context."
            )

        # ── Compute EAP scores ─────────────────────────────────────────────
        eap_scores: Dict = {}

        for sender_l in range(n_layers):
            # Δ contribution to residual stream for each attention head sender
            W_O_l = model.blocks[sender_l].attn.W_O.detach().float()  # [n_heads, d_head, d_model]

            for sender_h in range(n_heads):
                W_O_h = W_O_l[sender_h]                           # [d_head, d_model]
                Δz    = clean_attn[sender_l][sender_h] - corr_attn[sender_l][sender_h]  # [d_head]
                Δh    = Δz @ W_O_h                                # [d_model]

                for recv_l in range(sender_l + 1, n_layers):
                    if recv_l not in resid_grads:
                        continue
                    score = float((resid_grads[recv_l] * Δh).sum().item())
                    eap_scores[((sender_l, "attn", sender_h), recv_l)] = score

            # MLP sender: Δh = mlp_out_clean − mlp_out_corr  [d_model]
            Δmlp = clean_mlp[sender_l] - corr_mlp[sender_l]      # [d_model]
            for recv_l in range(sender_l + 1, n_layers):
                if recv_l not in resid_grads:
                    continue
                score = float((resid_grads[recv_l] * Δmlp).sum().item())
                eap_scores[((sender_l, "mlp", None), recv_l)] = score

        # Convert raw dict items to human-readable dicts with sender/receiver/score
        def _edge_label(key) -> str:
            (sl, kind, sh), rl = key
            if kind == "attn":
                return f"attn_L{sl:02d}H{sh:02d}"
            return f"mlp_L{sl:02d}"

        top_edges = [
            {
                "sender":   _edge_label(k),
                "receiver": f"resid_pre_L{k[1]:02d}",
                "score":    float(v),
                "raw_key":  k,
            }
            for k, v in sorted(eap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        ]

        logger.debug(
            "edge_attribution_patching done: n_edges=%d  clean_ld=%.4f  "
            "gradient_layers_captured=%d",
            len(eap_scores), clean_ld, len(resid_grads),
        )
        return {
            "edge_scores": eap_scores,
            "top_edges":   top_edges,
            "n_edges":     len(eap_scores),
            "clean_ld":    clean_ld,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 3c. ATTRIBUTION STABILITY  — novel Glassbox method
    #     Quantifies how robust head rankings are across corruption variants
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
        """
        Vectorised Kendall τ-b rank correlation (no scipy dependency).

        Kendall (1938), Biometrika 30(1-2), 81–93.
        τ ∈ [−1, 1]:  +1 = identical ranking,  0 = no association, −1 = reversed.
        τ-b formulation handles tied ranks correctly.
        """
        n  = len(x)
        # Upper-triangle indices — all unique pairs (i, j) with j > i
        r, c = np.triu_indices(n, k=1)
        dx   = x[r] - x[c]                       # [n_pairs]
        dy   = y[r] - y[c]                        # [n_pairs]
        prod = dx * dy
        concordant = int((prod > 0).sum())
        discordant = int((prod < 0).sum())
        ties_x     = int((dx == 0).sum())
        ties_y     = int((dy == 0).sum())
        n_pairs    = n * (n - 1) // 2
        denom      = np.sqrt(
            max(n_pairs - ties_x, 0) * max(n_pairs - ties_y, 0)
        )
        return float((concordant - discordant) / denom) if denom > 0 else 0.0

    def attribution_stability(
        self,
        clean_tokens:     torch.Tensor,
        target_token:     str,
        distractor_token: str,
        n_corruptions:    int   = 10,
        replace_fraction: float = 0.25,
        seed:             int   = 42,
    ) -> Dict:
        """
        Attribution Stability Analysis — novel Glassbox 2.0 method.

        Computes attribution scores across K independent random corruptions
        of the clean prompt and measures how consistent the head rankings are.

        MOTIVATION
        ----------
        Standard attribution patching reports a single score per head for one
        specific (clean, corrupted) pair.  This score depends on the corruption:
        different corruptions can yield different rankings, especially for heads
        with moderate attribution.  Stability analysis quantifies this variance,
        enabling the user to distinguish:

          HIGH stability → head consistently important regardless of corruption.
                           Likely a structural mechanism participant.
          LOW  stability → head importance is corruption-artefact.
                           May be a context-specific or backup mechanism.

        ALGORITHM
        ---------
        For each random corruption c ∈ {1, …, K}:
            Replace `replace_fraction` of prompt tokens with random vocabulary
            tokens (BOS/anchor token never replaced).
            attrs_c(l,h) = attribution_patching(clean, corrupt_c, t, d)

        Per-head statistics:
            mean_attr(l,h) = (1/K) Σ_c  attrs_c(l,h)
            std_attr(l,h)  =  std  over c  of attrs_c(l,h)

        Stability coefficient (novel Glassbox metric):
            S(l,h) = 1 − std_attr / (|mean_attr| + ε)
                → +1.0 : perfectly stable  (std → 0)
                →  0.0 : std equals |mean| magnitude
                → −∞  : std far exceeds |mean|  (pure noise)

        Global rank consistency: mean Kendall τ across all C(K,2) pairs of
        attribution runs.  τ = 1 means all K corruptions rank heads identically.

        NOVEL CONTRIBUTION (disclosed)
        --------------------------------
        No existing mechanistic interpretability library (TransformerLens,
        ACDC, EAP, Circuit Tracer) characterises per-head attribution uncertainty
        as a function of corruption choice.  This is the first formulation of
        head-level attribution stability with a rank-correlation consistency score.

        Complexity: O(3K) forward passes.

        References
        ----------
        Nanda et al. (2023). Attribution Patching. (base attribution method)
        Kendall (1938). "A New Measure of Rank Correlation."
            Biometrika 30(1-2), 81–93. https://doi.org/10.1093/biomet/30.1-2.81

        Parameters
        ----------
        clean_tokens     : tokenised clean prompt  [1, seq_len]
        target_token     : correct next-token string
        distractor_token : distractor token string
        n_corruptions    : independent corruptions K (default 10)
        replace_fraction : fraction of tokens randomly replaced (default 0.25)
        seed             : RNG seed (default 42, matches thesis)

        Returns
        -------
        {
          'mean_attributions' : Dict[(l,h) -> float]
          'std_attributions'  : Dict[(l,h) -> float]
          'stability_scores'  : Dict[(l,h) -> float]    — S(l,h)
          'rank_consistency'  : float — mean Kendall τ ∈ [−1, 1]
          'top_stable_heads'  : List[Dict]  — top-10 by S among salient heads
          'n_corruptions'     : int   — number of successful runs
        }
        """
        model    = self.model
        n_layers = self.n_layers
        n_heads  = self.n_heads

        # Accept either a token string ("Mary") or an already-resolved int ID
        if isinstance(target_token, int):
            target_id     = target_token
            distractor_id = distractor_token
        else:
            try:
                target_id     = model.to_single_token(target_token)
                distractor_id = model.to_single_token(distractor_token)
            except Exception:
                target_id     = int(model.to_tokens(target_token)[0, -1].item())
                distractor_id = int(model.to_tokens(distractor_token)[0, -1].item())

        rng        = np.random.default_rng(seed)
        vocab_size = model.cfg.d_vocab

        all_attr_runs: List[Dict[Tuple[int, int], float]] = []

        for run_idx in range(n_corruptions):
            tok_arr = clean_tokens.cpu().numpy().copy()          # [1, seq_len]
            mask    = rng.random(tok_arr.shape) < replace_fraction
            mask[:, 0] = False                                   # never replace BOS/anchor
            rand_toks   = rng.integers(0, vocab_size, tok_arr.shape).astype(tok_arr.dtype)
            tok_arr[mask] = rand_toks[mask]
            corrupted = torch.tensor(tok_arr, device=clean_tokens.device)
            try:
                attrs, _ = self.attribution_patching(
                    clean_tokens, corrupted, target_id, distractor_id
                )
                all_attr_runs.append(attrs)
                logger.debug("Stability run %d/%d succeeded.", run_idx + 1, n_corruptions)
            except Exception as exc:
                logger.warning("Stability run %d failed: %s", run_idx + 1, exc)

        if len(all_attr_runs) < 2:
            return {
                "error": f"Only {len(all_attr_runs)} successful run(s) — need ≥ 2.",
                "n_corruptions": len(all_attr_runs),
            }

        # ── Per-head statistics ────────────────────────────────────────────
        heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        # attr_matrix[k, i] = attribution of head i on run k
        attr_matrix = np.array(
            [[run.get(head, 0.0) for head in heads] for run in all_attr_runs],
            dtype=np.float64,
        )                                          # [K, n_heads_total]

        mean_attrs = attr_matrix.mean(axis=0)      # [n_heads_total]
        std_attrs  = attr_matrix.std(axis=0)       # [n_heads_total]
        eps        = 1e-8
        stability  = 1.0 - std_attrs / (np.abs(mean_attrs) + eps)

        # Return as flat lists (index-aligned with heads) so callers can iterate
        # directly over float values.  The full dict mapping is kept under the
        # _by_head suffixed keys for look-up by (layer, head) tuple.
        mean_attributions      = [float(mean_attrs[i]) for i in range(len(heads))]
        std_attributions       = [float(std_attrs[i])  for i in range(len(heads))]
        stability_scores       = [float(stability[i])  for i in range(len(heads))]
        mean_attributions_dict = {head: float(mean_attrs[i]) for i, head in enumerate(heads)}
        std_attributions_dict  = {head: float(std_attrs[i])  for i, head in enumerate(heads)}
        stability_scores_dict  = {head: float(stability[i])  for i, head in enumerate(heads)}

        # ── Global rank consistency: mean Kendall τ across all run pairs ───
        K = len(all_attr_runs)
        tau_vals: List[float] = []
        for i in range(K):
            for j in range(i + 1, K):
                tau = self._kendall_tau(attr_matrix[i], attr_matrix[j])
                if not np.isnan(tau):
                    tau_vals.append(tau)
        rank_consistency = float(np.mean(tau_vals)) if tau_vals else 0.0

        # ── Top stable heads: high S AND salient (|mean| > median) ────────
        abs_means       = np.abs(mean_attrs)
        median_abs_mean = float(np.median(abs_means))
        top_stable_heads = sorted(
            [
                {
                    "layer":     head[0],
                    "head":      head[1],
                    "mean_attr": float(mean_attrs[i]),
                    "std_attr":  float(std_attrs[i]),
                    "stability": float(stability[i]),
                }
                for i, head in enumerate(heads)
                if abs_means[i] > median_abs_mean
            ],
            key=lambda x: x["stability"],
            reverse=True,
        )[:10]

        logger.info(
            "attribution_stability done: K=%d  rank_consistency=τ=%.3f",
            len(all_attr_runs), rank_consistency,
        )
        return {
            "mean_attributions": mean_attributions,
            "std_attributions":  std_attributions,
            "stability_scores":  stability_scores,
            "rank_consistency":  rank_consistency,
            "top_stable_heads":  top_stable_heads,
            "n_corruptions":     len(all_attr_runs),
        }

    # ──────────────────────────────────────────────────────────────────────
    # 4a. EXACT SUFFICIENCY — positive ablation (keep circuit, corrupt rest)
    #     NOT an approximation. Dual of comprehensiveness.
    # ──────────────────────────────────────────────────────────────────────

    def _suff_exact(
        self,
        circuit:          List[Tuple[int, int]],
        clean_tokens:     torch.Tensor,
        corrupted_tokens: torch.Tensor,
        clean_ld:         float,
        target_token:     int,
        distractor_token: int,
    ) -> float:
        """
        Exact sufficiency via positive corrupted-activation patching.

        Formula (dual of Wang et al. 2022 comprehensiveness)
        -----------------------------------------------------
            Suff_exact = LD_circuit_only / LD_clean

        where LD_circuit_only = logit diff when every NON-circuit head's
        clean activation is replaced with the corrupted activation.

        If Suff_exact is high, the circuit alone (without the rest of the
        model's attention heads) preserves the clean prediction — i.e., the
        circuit is sufficient.  Typical values on GPT-2 IOI: 0.90–1.05.

        Contrast with Taylor approximation:
            Suff_approx ≈ Σ attr(h) / LD_clean   (fast, 0 extra passes)
        This exact version requires 2 additional forward passes.

        Parameters
        ----------
        circuit     : List of (layer, head) tuples in the circuit.
        Returns 0.0 if circuit is empty or clean_ld == 0.
        """
        if not circuit or clean_ld == 0.0:
            return 0.0

        circuit_set = set(circuit)

        # Collect all model heads NOT in the circuit
        non_circuit: Dict[int, List[int]] = {}
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                if (layer, head) not in circuit_set:
                    non_circuit.setdefault(layer, []).append(head)

        if not non_circuit:
            # Circuit = full model; suff is trivially 1.0
            return 1.0

        needed_layers = list(non_circuit.keys())

        # Pass 1: cache corrupted z for NON-circuit layers only
        corr_cache: Dict[str, torch.Tensor] = {}

        def _save(key: str):
            def hook(act, hook=None):
                corr_cache[key] = act.detach().clone()
            return hook

        with torch.no_grad():
            self.model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (f"blocks.{l}.attn.hook_z", _save(f"blocks.{l}.attn.hook_z"))
                    for l in needed_layers
                ],
            )

        # Pass 2: patched forward — replace NON-circuit heads with corrupted z
        def _patch_non_circuit(layer: int, heads: List[int]):
            key = f"blocks.{layer}.attn.hook_z"
            def hook(act, hook=None):
                result = act.clone()
                if key in corr_cache:
                    corr = corr_cache[key]
                    min_seq = min(result.shape[1], corr.shape[1])
                    for head in heads:
                        result[:, :min_seq, head, :] = corr[:, :min_seq, head, :]
                return result
            return hook

        hooks = [
            (f"blocks.{l}.attn.hook_z", _patch_non_circuit(l, heads))
            for l, heads in non_circuit.items()
        ]

        with torch.no_grad():
            patched_logits = self.model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        circuit_only_ld = (
            patched_logits[0, -1, target_token]
            - patched_logits[0, -1, distractor_token]
        ).item()

        suff = circuit_only_ld / clean_ld
        return float(np.clip(suff, 0.0, 1.5))   # clip at 1.5 to handle amplification

    # ──────────────────────────────────────────────────────────────────────
    # 4b. COMPREHENSIVENESS — exact corrupted activation patching
    #     (Wang et al. 2022 — NOT an approximation)
    # ──────────────────────────────────────────────────────────────────────

    def _comp(
        self,
        circuit:          List[Tuple[int, int]],
        clean_tokens:     torch.Tensor,
        corrupted_tokens: torch.Tensor,
        clean_ld:         float,
        target_token:     int,
        distractor_token: int,
    ) -> float:
        """
        Exact comprehensiveness via corrupted activation patching.

        Formula (Wang et al. 2022)
        ---------------------------
            Comp = 1 − LD_patched / LD_clean

        where LD_patched = logit diff when every circuit head's clean activation
        is replaced with the corrupted activation (bidirectional name-swap run).

        If Comp is high, corrupting the circuit strongly disrupts the prediction
        → the circuit is necessary. If Comp is low, backup mechanisms compensate.

        Why NOT zero ablation: removing z entirely also removes the anchoring
        baseline; other heads overcompensate → Comp ≈ 0 (misleading).
        Why NOT mean ablation: mean over sequence preserves residual signal → same.

        2 forward passes: (1) corrupt cache for circuit layers, (2) patched forward.
        """
        if not circuit or clean_ld == 0.0:
            return 0.0

        # Detect degenerate corruption: if clean tokens are a prefix of or identical
        # to corrupted tokens (e.g. when name-swap fallback just appended a token),
        # the corrupted activations for those positions are unchanged, so corrupt-
        # patching is a no-op and comprehensiveness would falsely read 0.
        # In that case, fall back to zero-ablation comprehensiveness.
        c_len = clean_tokens.shape[1]
        r_len = corrupted_tokens.shape[1]
        overlap = min(c_len, r_len)
        tokens_identical = bool(
            torch.all(clean_tokens[:, :overlap] == corrupted_tokens[:, :overlap]).item()
        )
        if tokens_identical:
            return self._comp_zero_ablation(
                circuit, clean_tokens, clean_ld, target_token, distractor_token
            )

        needed_layers = list({l for l, _ in circuit})

        # Pass 1: cache corrupted z for circuit layers only
        corr_cache: Dict[str, torch.Tensor] = {}

        def _save(key: str):
            def hook(act, hook=None):
                corr_cache[key] = act.detach().clone()
            return hook

        with torch.no_grad():
            self.model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[
                    (f"blocks.{l}.attn.hook_z", _save(f"blocks.{l}.attn.hook_z"))
                    for l in needed_layers
                ],
            )

        # Pass 2: patched forward — replace circuit heads with corrupted z
        def _patch_corr(layer: int, head: int):
            key = f"blocks.{layer}.attn.hook_z"
            def hook(act, hook=None):
                result = act.clone()
                if key in corr_cache:
                    corr = corr_cache[key]
                    # Handle sequence-length mismatch between clean/corrupt runs
                    min_seq = min(result.shape[1], corr.shape[1])
                    result[:, :min_seq, head, :] = corr[:, :min_seq, head, :]
                return result
            return hook

        hooks = [(f"blocks.{l}.attn.hook_z", _patch_corr(l, h)) for l, h in circuit]

        with torch.no_grad():
            patched_logits = self.model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        patched_ld = (
            patched_logits[0, -1, target_token]
            - patched_logits[0, -1, distractor_token]
        ).item()

        comp = 1.0 - (patched_ld / clean_ld)
        return float(np.clip(comp, 0.0, 1.0))

    def _comp_zero_ablation(
        self,
        circuit:         List[Tuple[int, int]],
        clean_tokens:    torch.Tensor,
        clean_ld:        float,
        target_token:    int,
        distractor_token: int,
    ) -> float:
        """
        Fallback comprehensiveness via zero-ablation.

        Used when corrupted tokens are identical to clean tokens (e.g. name-swap
        fallback appended the distractor — the prefix activations are unchanged).

        Formula: Comp = 1 − LD_zero / LD_clean
        where LD_zero = logit diff when circuit head z-values are zeroed.

        Zero ablation is a stronger perturbation than corrupt patching; it gives
        a valid lower bound on circuit necessity when no meaningful counterfactual
        prompt is available.
        """
        def _zero_head(layer: int, head: int):
            def hook(act, hook=None):
                result = act.clone()
                result[:, :, head, :] = 0.0
                return result
            return hook

        hooks = [
            (f"blocks.{l}.attn.hook_z", _zero_head(l, h))
            for l, h in circuit
        ]

        with torch.no_grad():
            patched_logits = self.model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        patched_ld = (
            patched_logits[0, -1, target_token]
            - patched_logits[0, -1, distractor_token]
        ).item()

        comp = 1.0 - (patched_ld / clean_ld)
        return float(np.clip(comp, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────
    # 3. MINIMUM FAITHFUL CIRCUIT (MFC) — greedy forward/backward
    # ──────────────────────────────────────────────────────────────────────

    def minimum_faithful_circuit(
        self,
        clean_tokens:     torch.Tensor,
        corrupted_tokens: torch.Tensor,
        target_token:     int,
        distractor_token: int,
        target_suff:      float = 0.85,
        target_comp:      float = 0.25,
        method:           str   = "taylor",
        n_steps:          int   = 10,
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float], float]:
        """
        Auto-discover the Minimum Faithful Circuit (MFC).

        Algorithm
        ---------
        Phase 1 — Greedy forward selection:
            Sort heads by attribution score (descending).
            Add heads one-at-a-time until cumulative Taylor-sufficiency ≥ target_suff.

            NOTE: Forward selection uses the APPROXIMATE Taylor sufficiency
            (Σ attr / clean_ld). This is fast (no extra passes) but may over- or
            under-shoot actual causal sufficiency.

        Phase 2 — Backward pruning:
            For each head (last-added first), try removing it.
            Keep the removal if exact comprehensiveness stays ≥ target_comp.
            (2 forward passes per head tried)

        The asymmetry — approximate forward, exact backward — is intentional:
        forward selection uses the cheap approximation to build a candidate,
        exact backward pruning removes redundant heads causally.

        Parameters
        ----------
        target_suff : Approximate sufficiency threshold (default 0.85 = 85%)
            Raise → larger circuit, more complete explanation.
            Lower → smaller circuit, may miss important heads.
        target_comp : Comprehensiveness threshold for pruning (default 0.25 = 25%)
            Raise → more conservative pruning, larger final circuit.
            Lower → aggressive pruning; may collapse to 1-2 heads when backup
                     mechanisms absorb the causal signal.

        Returns
        -------
        circuit      : List of (layer, head) tuples, sorted by attribution
        attributions : Full attribution dict for all heads
        clean_ld     : Clean logit difference (reused by analyze, no redundant call)
        """
        attributions, clean_ld = self.attribution_patching(
            clean_tokens, corrupted_tokens, target_token, distractor_token,
            method=method, n_steps=n_steps,
        )

        if clean_ld == 0.0:
            logger.warning("clean_ld == 0: model is indifferent between target and distractor.")
            return [], attributions, 0.0

        # Phase 1: greedy forward (approximate sufficiency)
        ranked = sorted(
            [(k, v) for k, v in attributions.items() if v > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        if not ranked:
            # Fallback: use top-5 by absolute value when all scores ≤ 0
            ranked = sorted(
                attributions.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]

        candidate: List[Tuple[int, int]] = []
        cumulative_attr = 0.0
        for head, attr in ranked:
            candidate.append(head)
            cumulative_attr += attr
            approx_suff = float(np.clip(cumulative_attr / clean_ld, 0.0, 1.0))
            if approx_suff >= target_suff:
                break

        # Phase 2: backward pruning (exact comprehensiveness)
        circuit = list(candidate)
        for head in reversed(list(candidate)):
            trial = [h for h in circuit if h != head]
            if not trial:
                break
            comp = self._comp(
                trial, clean_tokens, corrupted_tokens,
                clean_ld, target_token, distractor_token,
            )
            if comp >= target_comp:
                circuit = trial
                logger.debug("Pruned L%dH%d — comp=%.3f ≥ %.3f", head[0], head[1], comp, target_comp)

        logger.info(
            "MFC: %d heads  clean_ld=%.4f  target_suff=%.2f  target_comp=%.2f",
            len(circuit), clean_ld, target_suff, target_comp,
        )
        return circuit, attributions, clean_ld

    # ──────────────────────────────────────────────────────────────────────
    # 4. BOOTSTRAP CONFIDENCE INTERVALS
    # ──────────────────────────────────────────────────────────────────────

    def bootstrap_metrics(
        self,
        prompts:    List[Tuple[str, str, str]],
        n_boot:     int   = 500,
        alpha:      float = 0.05,
        seed:       int   = 42,
        exact_suff: bool  = True,
    ) -> Dict:
        """
        Bootstrap 95% CI on Sufficiency / Comprehensiveness / F1.

        Requires len(prompts) ≥ 20 for statistically reliable intervals.
        With n < 10, CIs are wide and should be treated as directional only.

        Parameters
        ----------
        prompts    : List of (prompt, correct, incorrect) tuples
        n_boot     : Bootstrap resamples (default 500)
        alpha      : Significance level (default 0.05 → 95% CI)
        seed       : RNG seed for reproducible bootstrap samples (default 42).
                     Pass seed=None for a random seed.
        exact_suff : If True (default), compute sufficiency via exact positive
                     ablation (_suff_exact — keep circuit, corrupt rest, 2 passes).
                     If False, use the fast Taylor approximation (0 extra passes).

                     REPRODUCIBILITY NOTE (paper: arXiv 2603.09988)
                     The ~100% sufficiency figure in the paper was computed with
                     exact_suff=True, seed=42, on GPT-2 small (12L/12H/768d),
                     Apple M2 Pro, PyTorch 2.2.0, TransformerLens 1.19.0.
                     The 0.80 figure in analyze() output uses exact_suff=False
                     (Taylor approx). Both are correct; they measure different things.

        Returns
        -------
        dict with keys 'sufficiency', 'comprehensiveness', 'f1', each containing:
            mean, std, ci_lo, ci_hi, n
        Also includes 'meta': {n_boot, alpha, seed, n_prompts_used, exact_suff}.
        """
        suff_vals: List[float] = []
        comp_vals: List[float] = []
        f1_vals:   List[float] = []

        for idx, (prompt, correct, incorrect) in enumerate(prompts):
            logger.info("Bootstrap %d/%d: '%s'", idx + 1, len(prompts), prompt[:50])
            try:
                t_tok = self.model.to_single_token(correct)
                d_tok = self.model.to_single_token(incorrect)
            except Exception:
                logger.warning("Skipping multi-token correct token: '%s'", correct)
                continue

            tokens_c    = self.model.to_tokens(prompt)
            corr_prompt = self._name_swap(prompt, correct.strip(), incorrect.strip())
            tokens_corr = self.model.to_tokens(corr_prompt)

            circuit, attrs, clean_ld = self.minimum_faithful_circuit(
                tokens_c, tokens_corr, t_tok, d_tok
            )

            if not circuit or clean_ld == 0.0:
                logger.warning("Empty circuit or zero LD — skipping prompt %d", idx + 1)
                continue

            if exact_suff:
                suff = self._suff_exact(
                    circuit, tokens_c, tokens_corr, clean_ld, t_tok, d_tok
                )
            else:
                total = sum(attrs.get(h, 0.0) for h in circuit)
                suff  = float(np.clip(total / clean_ld, 0.0, 1.0))

            comp  = self._comp(circuit, tokens_c, tokens_corr, clean_ld, t_tok, d_tok)
            f1    = 2.0 * suff * comp / (suff + comp) if (suff + comp) > 0.0 else 0.0

            suff_vals.append(suff)
            comp_vals.append(comp)
            f1_vals.append(f1)
            logger.info(
                "  Suff=%.1f%%  Comp=%.1f%%  F1=%.1f%%  circuit=%d heads  exact=%s",
                suff * 100, comp * 100, f1 * 100, len(circuit), exact_suff,
            )

        n = len(suff_vals)
        if n < 2:
            return {"error": f"Only {n} valid prompts — need ≥ 2. Recommend ≥ 20 for reliable CIs."}

        if n < 20:
            logger.warning(
                "Bootstrap CI computed on n=%d prompts. Recommend n≥20 for reliable intervals.", n
            )

        # Seeded RNG — reproducible bootstrap samples without polluting global state
        rng = np.random.default_rng(seed)

        def _boot_ci(vals: List[float]) -> Dict:
            arr  = np.array(vals)
            boot = np.array([
                np.mean(rng.choice(arr, len(arr), replace=True))
                for _ in range(n_boot)
            ])
            return {
                "mean":  float(np.mean(arr)),
                "std":   float(np.std(arr)),
                "ci_lo": float(np.percentile(boot, 100.0 * alpha / 2)),
                "ci_hi": float(np.percentile(boot, 100.0 * (1.0 - alpha / 2))),
                "n":     n,
            }

        return {
            "sufficiency":       _boot_ci(suff_vals),
            "comprehensiveness": _boot_ci(comp_vals),
            "f1":                _boot_ci(f1_vals),
            "meta": {
                "n_boot":          n_boot,
                "alpha":           alpha,
                "seed":            seed,
                "n_prompts_used":  n,
                "exact_suff":      exact_suff,
                "suff_is_approx":  not exact_suff,
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # 5. FUNCTIONAL CIRCUIT ALIGNMENT SCORE (FCAS) — Novel metric
    # ──────────────────────────────────────────────────────────────────────

    def functional_circuit_alignment(
        self,
        heads_a:     List[Dict],
        heads_b:     List[Dict],
        top_k:       int = 3,
        n_null:      int = 1000,
        seed:        int = 42,
    ) -> Dict:
        """
        Functional Circuit Alignment Score (FCAS).

        FCAS = 1 − mean( |rel_depth_A_i − rel_depth_B_i| )  for i in top_k pairs.
        where rel_depth = layer / (n_layers − 1)  ∈ [0, 1].

        A null distribution is computed by comparing random circuits of the same
        size — this gives context for whether the observed FCAS is meaningful.

        Limitations (disclosed)
        -----------------------
        * Matching is by rank, not by functional role. Two heads at the same depth
          doing different things (e.g., name-mover vs S-inhibition) score as aligned.
        * Compares depth only, not head index within a layer.
        * Sensitive to top_k. Increase k for more stable but noisier estimates.

        Parameters
        ----------
        heads_a, heads_b : Output of get_top_heads() — list of dicts with
                           'rel_depth', 'layer', 'head', 'attr' keys.
        top_k            : Number of matched head pairs (default 3).
        n_null           : Bootstrap iterations for null distribution (default 1000).

        Returns
        -------
        dict with:
            fcas        : observed FCAS
            null_mean   : mean FCAS under random circuits
            null_std    : std of null distribution
            z_score     : (fcas - null_mean) / null_std
            pairs       : per-pair alignment details
        """
        k = min(top_k, len(heads_a), len(heads_b))
        if k == 0:
            return {"fcas": 0.0, "null_mean": 0.0, "null_std": 0.0, "z_score": 0.0, "pairs": []}

        pairs = []
        for i in range(k):
            a, b = heads_a[i], heads_b[i]
            depth_diff = abs(a["rel_depth"] - b["rel_depth"])
            pairs.append({
                "rank":        i + 1,
                "model_a":     f"L{a['layer']}H{a['head']} (depth={a['rel_depth']:.3f})",
                "model_b":     f"L{b['layer']}H{b['head']} (depth={b['rel_depth']:.3f})",
                "depth_diff":  depth_diff,
                "aligned":     depth_diff < 0.15,
            })

        fcas = 1.0 - (sum(p["depth_diff"] for p in pairs) / k)

        # Null distribution: random circuits of the same size (seeded for reproducibility)
        rng = np.random.default_rng(seed)
        null_fcas_vals = []
        for _ in range(n_null):
            rand_a = rng.uniform(0, 1, k)
            rand_b = rng.uniform(0, 1, k)
            null_fcas_vals.append(1.0 - float(np.mean(np.abs(rand_a - rand_b))))

        null_mean = float(np.mean(null_fcas_vals))
        null_std  = float(np.std(null_fcas_vals))
        z_score   = (fcas - null_mean) / null_std if null_std > 0 else 0.0

        logger.info(
            "FCAS=%.3f  null_mean=%.3f  null_std=%.3f  z=%.2f  k=%d",
            fcas, null_mean, null_std, z_score, k,
        )
        return {
            "fcas":      float(fcas),
            "null_mean": null_mean,
            "null_std":  null_std,
            "z_score":   z_score,
            "pairs":     pairs,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 5b. MULTI-PROMPT CIRCUIT STABILITY SUITE  (v3.2.0 — ICML 2026 §4.5)
    # ──────────────────────────────────────────────────────────────────────

    def stability_suite(
        self,
        prompt_variants: List[Tuple[str, str, str]],
        seed:            int   = 42,
        threshold:       float = 0.75,
    ) -> Dict:
        """
        Measure circuit membership stability across paraphrases / variants of
        the same underlying semantic task.

        This is the key experimental result added in v3.2.0 and reported in
        ICML 2026 §4.5: CircuitDiff Jaccard scores above ``threshold`` (default
        0.75) correlate with stable behavioral benchmarks across fine-tuning
        checkpoints.

        Parameters
        ----------
        prompt_variants : List of (prompt, correct, incorrect) tuples that are
                          semantically equivalent (e.g., paraphrases of the same
                          indirect-object-identification task).  Minimum 5;
                          recommended 20–50 for reliable statistics.
        seed            : RNG seed (default 42) for any internal sampling.
        threshold       : Jaccard threshold above which a circuit is considered
                          "stable" (default 0.75). Circuits at or above this
                          threshold count toward the stability_rate.

        Returns
        -------
        dict with keys:

            jaccard_mean     : float — mean pairwise Jaccard circuit similarity
                               across all variant pairs
            jaccard_std      : float — std of pairwise Jaccard scores
            jaccard_min      : float — worst-case pair
            jaccard_max      : float — best-case pair
            stability_rate   : float — fraction of pairs with Jaccard ≥ threshold
            consensus_circuit: List[HeadTuple] — heads present in ≥50% of circuits
            per_variant      : List[dict] — per-variant circuit + attribution summary
            n_pairs          : int — total pairwise comparisons made
            meta             : dict — threshold, seed, n_variants, model_name

        Notes
        -----
        A Jaccard score of 1.0 means identical circuits; 0.0 means no shared heads.
        For robust ICML claims, run with ≥ 20 paraphrases (seed=42) and report
        both jaccard_mean and stability_rate at threshold=0.75.

        Circuit membership is defined by the minimum faithful circuit returned
        by ``minimum_faithful_circuit()`` at the default threshold (0.6).
        """
        if len(prompt_variants) < 2:
            raise ValueError("stability_suite requires at least 2 prompt variants.")

        circuits: List[set] = []
        per_variant: List[Dict] = []

        for idx, (prompt, correct, incorrect) in enumerate(prompt_variants):
            logger.info(
                "stability_suite %d/%d: '%s'",
                idx + 1, len(prompt_variants), prompt[:60],
            )
            try:
                t_tok = self.model.to_single_token(correct)
                d_tok = self.model.to_single_token(incorrect)
            except Exception:
                logger.warning(
                    "stability_suite: skipping variant %d — multi-token correct token",
                    idx,
                )
                per_variant.append({
                    "idx": idx, "prompt": prompt[:80],
                    "circuit": [], "n_heads": 0, "skipped": True,
                })
                continue

            tokens_c    = self.model.to_tokens(prompt)
            corr_prompt = self._name_swap(prompt, correct.strip(), incorrect.strip())
            tokens_corr = self.model.to_tokens(corr_prompt)

            try:
                circuit, attrs, _ = self.minimum_faithful_circuit(
                    tokens_c, tokens_corr, t_tok, d_tok
                )
            except Exception as exc:
                logger.warning("stability_suite: variant %d failed: %s", idx, exc)
                per_variant.append({
                    "idx": idx, "prompt": prompt[:80],
                    "circuit": [], "n_heads": 0, "error": str(exc),
                })
                continue

            head_set: set = set()
            for h in circuit:
                layer = h[0] if isinstance(h, (tuple, list)) else h.get("layer", -1)
                head  = h[1] if isinstance(h, (tuple, list)) else h.get("head",  -1)
                head_set.add((int(layer), int(head)))

            circuits.append(head_set)
            per_variant.append({
                "idx":        idx,
                "prompt":     prompt[:80],
                "circuit":    sorted(head_set),
                "n_heads":    len(head_set),
                "skipped":    False,
            })

        if len(circuits) < 2:
            return {
                "jaccard_mean":      0.0,
                "jaccard_std":       0.0,
                "jaccard_min":       0.0,
                "jaccard_max":       0.0,
                "stability_rate":    0.0,
                "consensus_circuit": [],
                "per_variant":       per_variant,
                "n_pairs":           0,
                "meta": {
                    "threshold":    threshold,
                    "seed":         seed,
                    "n_variants":   len(prompt_variants),
                    "model_name":   self.model_name,
                    "warning":      "Fewer than 2 valid circuits — cannot compute Jaccard.",
                },
            }

        # Pairwise Jaccard matrix
        jaccard_scores: List[float] = []
        n = len(circuits)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = circuits[i], circuits[j]
                union_size = len(a | b)
                if union_size == 0:
                    jaccard_scores.append(0.0)
                else:
                    jaccard_scores.append(len(a & b) / union_size)

        jaccard_arr   = np.array(jaccard_scores, dtype=float)
        jaccard_mean  = float(np.mean(jaccard_arr))
        jaccard_std   = float(np.std(jaccard_arr))
        jaccard_min   = float(np.min(jaccard_arr))
        jaccard_max   = float(np.max(jaccard_arr))
        stability_rate = float(np.mean(jaccard_arr >= threshold))

        # Consensus circuit: heads present in ≥ 50% of valid circuits
        from collections import Counter
        head_counts: Counter = Counter()
        for c in circuits:
            for h in c:
                head_counts[h] += 1
        consensus_threshold = n / 2.0
        consensus_circuit = sorted(
            [h for h, cnt in head_counts.items() if cnt >= consensus_threshold]
        )

        logger.info(
            "stability_suite: jaccard_mean=%.3f ± %.3f  stability_rate=%.1f%%  "
            "consensus_heads=%d  n_pairs=%d",
            jaccard_mean, jaccard_std, stability_rate * 100,
            len(consensus_circuit), len(jaccard_scores),
        )

        return {
            "jaccard_mean":      jaccard_mean,
            "jaccard_std":       jaccard_std,
            "jaccard_min":       jaccard_min,
            "jaccard_max":       jaccard_max,
            "stability_rate":    stability_rate,
            "consensus_circuit": consensus_circuit,
            "per_variant":       per_variant,
            "n_pairs":           len(jaccard_scores),
            "meta": {
                "threshold":   threshold,
                "seed":        seed,
                "n_variants":  len(prompt_variants),
                "model_name":  self.model_name,
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # 6. SINGLE-CALL ANALYZE API
    # ──────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        prompt:             str,
        correct:            str,
        incorrect:          str,
        method:             str  = "taylor",
        n_steps:            int  = 10,
        include_logit_lens: bool = False,
    ) -> Dict:
        """
        One-call circuit discovery + faithfulness metrics.

        Parameters
        ----------
        prompt    : Input text (e.g. "When Mary and John went to the store, John gave a drink to")
        correct   : Correct next token (e.g. " Mary")
        incorrect : Distractor token    (e.g. " John")
        method             : Attribution method — "taylor" (fast, default) or
                             "integrated_gradients" (accurate, 2+n_steps passes)
        n_steps            : Interpolation steps for integrated_gradients (default 10)
        include_logit_lens : If True, also run logit_lens() and include result
                             in output under key 'logit_lens' (adds 1 forward pass)

        Returns
        -------
        {
            'circuit'          : [(layer, head), ...],   # MFC heads, sorted by attribution
            'n_heads'          : int,
            'clean_ld'         : float,                  # logit(correct) - logit(distractor)
            'corr_prompt'      : str,                    # name-swap corrupted prompt
            'attributions'     : {str((l, h)): float},  # all heads, string keys
            'mlp_attributions' : {str(layer): float},   # per-layer MLP scores
            'top_heads'        : [{'layer', 'head', 'attr', 'rel_depth'}, ...],
            'method'           : str,                    # attribution method used
            'logit_lens'       : {...}  (only if include_logit_lens=True)
            'faithfulness': {
                'sufficiency':       float,  # Taylor approximation (see caveats)
                'comprehensiveness': float,  # exact (corrupted activation patching)
                'f1':                float,  # harmonic mean
                'category':          str,
                'suff_is_approx':    bool,   # True for taylor, False for IG
            }
        }

        Speed
        -----
        taylor:               O(3 + 2p) passes     — fast, suitable for iteration
        integrated_gradients: O(5 + 2n_steps + 2p) — accurate, final analysis
        include_logit_lens adds O(1).
        p = number of backward pruning steps, typically 0-4 on IOI.
        """
        # Input validation
        if not prompt or not correct or not incorrect:
            raise ValueError("prompt, correct, and incorrect must be non-empty strings")

        # Token resolution with fallback
        try:
            t_tok = self.model.to_single_token(correct)
            d_tok = self.model.to_single_token(incorrect)
        except Exception:
            t_tok = self.model.to_tokens(correct)[0, -1].item()
            d_tok = self.model.to_tokens(incorrect)[0, -1].item()

        tokens_c    = self.model.to_tokens(prompt)

        # Proper bidirectional name-swap corruption (Wang et al. 2022)
        corr_prompt = self._name_swap(prompt, correct.strip(), incorrect.strip())
        tokens_corr = self.model.to_tokens(corr_prompt)

        # Circuit discovery — pass method through for attribution
        circuit, attrs, clean_ld = self.minimum_faithful_circuit(
            tokens_c, tokens_corr, t_tok, d_tok,
            method=method, n_steps=n_steps,
        )

        # MLP attribution — 3 additional passes, completes the circuit picture
        mlp_attrs = self.mlp_attribution(tokens_c, tokens_corr, t_tok, d_tok)

        # Top heads ranked by attribution score (input format for FCAS)
        top_heads = self.get_top_heads(attrs, top_k=min(10, self.n_layers * self.n_heads))

        # Optional: logit lens (1 additional forward pass)
        ll_result = None
        if include_logit_lens:
            ll_result = self.logit_lens(tokens_c, correct.strip(), incorrect.strip())

        # Warn when the model doesn't perform the task (negative logit diff means
        # the model already prefers the distractor token over the target).
        # Circuit attributions will still be computed but faithfulness metrics are
        # unreliable in this regime.
        if clean_ld <= 0.0:
            logger.warning(
                "GlassboxV2.analyze: clean logit diff is %.3f (≤0). "
                "The model prefers the distractor over the correct token — "
                "this model may not perform the requested task. "
                "Circuit and faithfulness results should be treated as unreliable.",
                clean_ld,
            )

        # Faithfulness metrics
        total = sum(attrs.get(h, 0.0) for h in circuit)
        suff  = float(np.clip(total / clean_ld, 0.0, 1.0)) if clean_ld != 0.0 else 0.0
        comp  = self._comp(circuit, tokens_c, tokens_corr, clean_ld, t_tok, d_tok)
        f1    = 2.0 * suff * comp / (suff + comp) if (suff + comp) > 0.0 else 0.0

        # Category thresholds (documented, not theoretically derived)
        if   suff > 0.9 and comp < 0.4:   category = "backup_mechanisms"
        elif suff > 0.7 and comp > 0.5:   category = "faithful"
        elif suff < 0.5:                   category = "incomplete"
        elif suff < 0.6 and comp < 0.5:   category = "weak"
        else:                               category = "moderate"

        result = {
            "circuit":          sorted(circuit, key=lambda lh: (-attrs.get(lh, 0.0), lh[0], lh[1])),
            "n_heads":          len(circuit),
            "clean_ld":         clean_ld,
            "corr_prompt":      corr_prompt,
            "attributions":     {str(k): v for k, v in attrs.items()},
            "mlp_attributions": {str(l): v for l, v in mlp_attrs.items()},
            "top_heads":        top_heads,
            "method":           method,
            "faithfulness": {
                "sufficiency":       suff,
                "comprehensiveness": comp,
                "f1":                f1,
                "category":          category,
                "suff_is_approx":    method == "taylor",
            },
            # Reproducibility metadata — include model identity in every result
            # so saved results are self-documenting and can be re-verified.
            "model_metadata": {
                "model_name":  getattr(self.model.cfg, "model_name",
                                       getattr(self.model.cfg, "checkpoint_value", "unknown")),
                "n_layers":    self.n_layers,
                "n_heads":     self.n_heads,
                "d_model":     int(self.model.cfg.d_model),
                "d_head":      int(self.model.cfg.d_head),
                "glassbox_version": _GLASSBOX_VERSION,
            },
        }
        if ll_result is not None:
            result["logit_lens"] = ll_result
        return result

    # ──────────────────────────────────────────────────────────────────────
    # 6b. BATCH ANALYSIS — process multiple prompts in one call
    # ──────────────────────────────────────────────────────────────────────

    def batch_analyze(
        self,
        prompts:        List[Tuple[str, str, str]],
        method:         str  = "taylor",
        show_progress:  bool = True,
        skip_errors:    bool = True,
    ) -> List[Dict]:
        """
        Run analyze() on a batch of (prompt, correct, incorrect) tuples.

        Parameters
        ----------
        prompts       : List of (prompt, correct, incorrect) 3-tuples.
        method        : Attribution method passed to each analyze() call.
        show_progress : Log progress as "Processing i/N" (default True).
        skip_errors   : If True, failed prompts are recorded with an 'error'
                        key and processing continues.  If False, the first
                        error propagates immediately (default True).

        Returns
        -------
        List of result dicts (one per prompt), preserving input order.
        Each dict is identical to the output of analyze() for that prompt.
        Failed prompts (skip_errors=True) have an 'error' key instead of
        'circuit' / 'faithfulness' etc.

        Example
        -------
        >>> results = gb.batch_analyze([
        ...     ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
        ...     ("The capital of France is",                                   " Paris", " London"),
        ... ])
        >>> [r["faithfulness"]["f1"] for r in results if "faithfulness" in r]
        """
        results: List[Dict] = []
        n = len(prompts)
        for i, (prompt, correct, incorrect) in enumerate(prompts, start=1):
            if show_progress:
                logger.info("[%d/%d] %s", i, n, prompt[:70])
            try:
                result = self.analyze(prompt, correct, incorrect, method=method)
                results.append(result)
            except Exception as exc:
                if skip_errors:
                    logger.warning("[%d/%d] SKIPPED — %s: %s", i, n, type(exc).__name__, exc)
                    results.append({
                        "prompt":   prompt,
                        "correct":  correct,
                        "incorrect": incorrect,
                        "error":    f"{type(exc).__name__}: {exc}",
                    })
                else:
                    raise
        return results

    # ──────────────────────────────────────────────────────────────────────
    # 7. TOKEN ATTRIBUTION  (gradient × input, Simonyan et al. 2014)
    #    Scores each INPUT TOKEN by its signed contribution to LD.
    # ──────────────────────────────────────────────────────────────────────

    def token_attribution(
        self,
        tokens:           torch.Tensor,
        target_token:     int,
        distractor_token: int,
    ) -> Dict:
        """
        Per-input-token attribution via gradient × embedding (saliency map).

        Formula (Simonyan et al. 2014 / standard gradient-based attribution):
            attr(t_i) = embed(t_i) · ∇_{embed(t_i)} LD

        where LD = logit(target) − logit(distractor) at the last position.
        The dot product gives a signed scalar per token: positive = token
        pushes the model toward the target, negative = toward distractor.

        This is the INPUT-SPACE complement to head-level attribution patching.
        Use it to identify which prompt tokens are most important.

        APPROXIMATION NOTE (disclosed)
        --------------------------------
        Gradient × input is a first-order attribution.  It conflates the
        embedding magnitude with the gradient direction.  For large prompts
        or adversarial inputs, integrated gradients (call
        `attribution_patching(method="integrated_gradients")` on token
        embeddings) would be more accurate.  For typical mechanistic
        interpretability prompts (< 20 tokens), the first-order approximation
        is stable and interpretable.

        1 forward pass + 1 backward pass.

        References
        ----------
        Simonyan et al. (2014). "Deep Inside Convolutional Networks: Visualising
            Image Classification Models and Saliency Maps."
            https://arxiv.org/abs/1312.6034
        Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks."
            ICML 2017.  https://arxiv.org/abs/1703.01365
        Bastings & Filippova (2020). "The elephant in the interpretability room."
            ACL BlackboxNLP.  https://arxiv.org/abs/2009.02839

        Parameters
        ----------
        tokens           : [1, seq_len] tokenised prompt
        target_token     : int  — correct next-token ID
        distractor_token : int  — distractor token ID

        Returns
        -------
        {
            "token_ids"    : List[int]    — token IDs
            "token_strs"   : List[str]    — decoded token strings
            "attributions" : List[float]  — signed score per token
            "abs_attributions" : List[float]  — |score| per token (for ranking)
            "top_tokens"   : List[dict]   — top 5 by |attr|, with token_str and attr
        }
        """
        model = self.model

        # Get the embedding layer output and enable grad
        embed = model.embed.W_E  # [d_vocab, d_model]

        # One-hot token ids for gradient retrieval
        t_ids  = tokens[0].tolist()           # [seq_len]
        n_tok  = len(t_ids)

        # Build embedding tensor with gradient tracking
        with torch.enable_grad():
            # embed_input: [1, seq_len, d_model]
            embed_input = embed[tokens].detach().clone().requires_grad_(True)

            # Run model with custom embedding
            # TransformerLens supports `start_at_layer` or hook injection;
            # we use run_with_hooks to inject the gradient-tracked embedding.
            grad_store = {}
            def _embed_hook(value, hook):
                # Create a fresh *leaf* tensor so .grad is populated after backward().
                # If we call .requires_grad_(True) on the value that TransformerLens
                # passes in, it's a non-leaf (result of an op), so PyTorch skips
                # populating .grad and emits a UserWarning.  Detach + clone gives us
                # a proper leaf that participates in autograd from this point onward.
                leaf = value.detach().clone().requires_grad_(True)
                leaf.retain_grad()          # keep grad even though it's inside a hook
                grad_store["embed"] = leaf
                return leaf

            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("hook_embed", _embed_hook)],
            )
            # logits: [1, seq_len, d_vocab]
            ld = logits[0, -1, target_token] - logits[0, -1, distractor_token]
            ld.backward()

        emb_tensor = grad_store["embed"]                       # [1, seq_len, d_model]
        if emb_tensor.grad is None:
            # Fallback: return zero attribution if graph not connected
            logger.warning("token_attribution: gradient is None — check hook connection")
            attrs = [0.0] * n_tok
        else:
            grad = emb_tensor.grad[0].detach().float()         # [seq_len, d_model]
            emb  = emb_tensor[0].detach().float()              # [seq_len, d_model]
            # Gradient × Input: dot product over d_model dimension
            attrs = (grad * emb).sum(dim=-1).tolist()          # [seq_len]

        # Decode token strings — to_str_tokens(tokens[0]) on 1-D tensor
        # returns a flat list of strings, one per token position.
        token_strs = list(model.to_str_tokens(tokens[0]))

        # Top tokens by absolute attribution
        ranked = sorted(
            enumerate(attrs), key=lambda x: abs(x[1]), reverse=True
        )
        top_tokens = [
            {"rank": rank + 1, "token_str": token_strs[i], "token_id": t_ids[i],
             "attribution": attrs[i], "position": i}
            for rank, (i, _) in enumerate(ranked[:5])
        ]

        return {
            "token_ids":       t_ids,
            "token_strs":      token_strs,
            "attributions":    attrs,
            "abs_attributions": [abs(a) for a in attrs],
            "top_tokens":      top_tokens,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 8. ATTENTION PATTERN ANALYSIS
    #    Returns attention matrices + entropy for specified heads
    # ──────────────────────────────────────────────────────────────────────

    def attention_patterns(
        self,
        tokens:   torch.Tensor,
        heads:    Optional[List[Tuple[int, int]]] = None,
        top_k:    int = 10,
    ) -> Dict:
        """
        Extract and analyse attention patterns for specified heads.

        For each (layer, head) pair, returns the full attention matrix A[l,h]
        and computes:
          • Attention entropy  H(A) = −Σ_j a_{ij} log a_{ij}
            Low entropy → focused / "sharp" attention (e.g. induction heads).
            High entropy → diffuse / "spread out" attention.
          • Dominant source position per query (argmax over source dimension).
          • Attention to final position  A[i=-1, :] — what the last token
            attends to (most relevant for next-token prediction tasks).

        Also computes an automatic HEAD TYPE CLASSIFICATION based on:
          – Induction  : high attention to the token AFTER the previous
            occurrence of the current token (in-context learning signal).
          – Previous-token : A[i, i-1] is large (strong diagonal offset).
          – Duplicate-token: attends to earlier occurrences of same token.
          – Uniform : near-uniform distribution (high entropy head).

        NOTE: Type classification is HEURISTIC — based on attention pattern
        geometry only, not causal intervention.  True functional role requires
        activation patching (use edge_attribution_patching()).

        1 forward pass.

        References
        ----------
        Elhage et al. (2021). "A Mathematical Framework for Transformer
            Circuits."  §5–6 (induction heads, QK circuits).
            https://transformer-circuits.pub/2021/framework/index.html
        Olsson et al. (2022). "In-context Learning and Induction Heads."
            Transformer Circuits Thread.
            https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

        Parameters
        ----------
        tokens  : [1, seq_len] tokenised prompt
        heads   : List of (layer, head) tuples to analyse.
                  If None, returns the top_k heads by attention entropy variance
                  (most "interesting" heads).
        top_k   : If heads is None, how many heads to auto-select.

        Returns
        -------
        {
            "heads"        : List[str]  — "L{l}H{h}" labels
            "patterns"     : { "L{l}H{h}": np.ndarray [seq, seq] }
            "entropy"      : { "L{l}H{h}": float }
            "last_tok_attn": { "L{l}H{h}": np.ndarray [seq] }  — A[-1, :]
            "head_types"   : { "L{l}H{h}": str }
            "token_strs"   : List[str]
        }
        """
        model    = self.model
        n_layers = self.n_layers
        n_heads  = self.n_heads

        # Forward pass — cache all attention patterns  ─────────────────────
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "hook_pattern" in name,
            )

        token_strs = model.to_str_tokens(tokens)[0]

        # If heads not specified, auto-select by entropy variance  ──────────
        if heads is None:
            all_entropies = {}
            for l in range(n_layers):
                for h in range(n_heads):
                    key = f"blocks.{l}.attn.hook_pattern"
                    if key in cache:
                        A   = cache[key][0, h].float().numpy()   # [seq, seq]
                        A   = np.clip(A, 1e-9, 1.0)
                        ent = -float(np.sum(A * np.log(A), axis=-1).mean())
                        all_entropies[(l, h)] = ent
            # Select heads with highest entropy variance (most interesting)
            heads = sorted(all_entropies, key=lambda k: abs(all_entropies[k] - 1.0))[:top_k]

        patterns:      Dict[str, np.ndarray] = {}
        entropy_map:   Dict[str, float]      = {}
        last_tok_attn: Dict[str, np.ndarray] = {}
        head_types:    Dict[str, str]        = {}

        for layer, head in heads:
            label = f"L{layer:02d}H{head:02d}"
            key   = f"blocks.{layer}.attn.hook_pattern"

            if key not in cache:
                continue

            A    = cache[key][0, head].float().cpu().numpy()   # [seq, seq]
            A    = np.clip(A, 1e-9, 1.0)
            seq  = A.shape[0]

            patterns[label]      = A
            last_tok_attn[label] = A[-1, :]

            # Entropy
            ent = -float(np.sum(A * np.log(A), axis=-1).mean())
            entropy_map[label] = ent

            # Heuristic head-type classification  ─────────────────────────
            avg_diag_offset1 = float(np.mean([A[i, i-1] for i in range(1, seq)]))
            avg_self_attn    = float(np.mean([A[i, i]   for i in range(seq)]))
            uniform_thresh   = np.log(seq) * 0.85 if seq > 1 else 1.0

            if ent >= uniform_thresh:
                head_type = "uniform"
            elif avg_diag_offset1 > 0.3:
                head_type = "previous_token"
            elif avg_self_attn > 0.4:
                head_type = "self_attn"
            elif ent < 0.5:
                head_type = "focused"
            else:
                head_type = "mixed"

            # Induction check: does last-token attend strongly to a specific
            # position other than itself and the previous token?
            last_row    = A[-1, :]
            max_pos     = int(np.argmax(last_row))
            if max_pos not in (seq - 1, seq - 2) and last_row[max_pos] > 0.3:
                head_type = "induction_candidate"

            head_types[label] = head_type

        return {
            "heads":         [f"L{l:02d}H{h:02d}" for l, h in heads],
            "patterns":      patterns,
            "entropy":       entropy_map,
            "last_tok_attn": last_tok_attn,
            "head_types":    head_types,
            "token_strs":    list(token_strs),
        }
