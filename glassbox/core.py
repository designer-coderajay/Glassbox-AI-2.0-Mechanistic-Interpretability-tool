import torch
import numpy as np
import einops

# Reproducibility (seed=42 matches thesis)
torch.manual_seed(42)
np.random.seed(42)
from transformer_lens import HookedTransformer

# ═══════════════════════════════════════════════════════════════════════════
# GLASSBOX 2.0 — CELL 2: GlassboxV2 Core Engine
#
# Comprehensiveness: CORRUPTED ACTIVATION PATCHING (Wang et al. 2022)
#   NOT zero/mean ablation — both give unreliable results because:
#   - Zero ablation removes head's "anchoring" signal → other heads overcompensate
#   - Mean ablation preserves residual signal → logit diff barely changes
#   Corrupted patching: replaces clean z_h with corrupted z_h (different name)
#   → removes exactly the task-relevant signal without model overcompensation
# ═══════════════════════════════════════════════════════════════════════════

class GlassboxV2:
    """
    Glassbox 2.0 — Causal Mechanistic Interpretability Engine

    attribution_patching  : Jacobian × Δz, O(3 passes) — Nanda et al. 2023
    minimum_faithful_circuit : Greedy forward/backward auto-discovery
    bootstrap_metrics     : 95% CI on Suff/Comp/F1 — novel, no other tool reports this
    analyze               : Single-call API
    """

    def __init__(self, model):
        self.model    = model
        self.n_layers = model.cfg.n_layers
        self.n_heads  = model.cfg.n_heads

    # ──────────────────────────────────────────────────────────────────────
    # 1. ATTRIBUTION PATCHING
    # ──────────────────────────────────────────────────────────────────────

    def attribution_patching(self, clean_tokens, corrupted_tokens,
                              target_token, distractor_token):
        model = self.model

        # Pass 1: cache clean
        clean_cache = {}
        def make_save_clean(key):
            def hook(act, hook):
                clean_cache[key] = act.detach().clone()
            return hook
        with torch.no_grad():
            model.run_with_hooks(clean_tokens,
                fwd_hooks=[(f'blocks.{l}.attn.hook_z',
                            make_save_clean(f'blocks.{l}.attn.hook_z'))
                           for l in range(self.n_layers)])

        # Pass 2: cache corrupted
        corr_cache = {}
        def make_save_corr(key):
            def hook(act, hook):
                corr_cache[key] = act.detach().clone()
            return hook
        with torch.no_grad():
            model.run_with_hooks(corrupted_tokens,
                fwd_hooks=[(f'blocks.{l}.attn.hook_z',
                            make_save_corr(f'blocks.{l}.attn.hook_z'))
                           for l in range(self.n_layers)])

        # Pass 3: gradient pass
        grad_inputs = {
            f'blocks.{l}.attn.hook_z':
                clean_cache[f'blocks.{l}.attn.hook_z'].clone().float().requires_grad_(True)
            for l in range(self.n_layers)
        }
        def make_patch(key):
            def hook(act, hook):
                return grad_inputs[key].to(act.dtype)  # MUST return for grad flow
            return hook

        model.zero_grad()
        logits = model.run_with_hooks(clean_tokens,
            fwd_hooks=[(key, make_patch(key)) for key in grad_inputs])
        ld = (logits[0, -1, target_token].float()
              - logits[0, -1, distractor_token].float())
        clean_ld = ld.item()
        ld.backward()

        attributions = {}
        for l in range(self.n_layers):
            key = f'blocks.{l}.attn.hook_z'
            g   = grad_inputs[key].grad
            if g is None:
                for h in range(self.n_heads): attributions[(l, h)] = 0.0
                continue
            delta = (clean_cache[key] - corr_cache[key]).float()
            for h in range(self.n_heads):
                attributions[(l, h)] = (g[0, -1, h, :] * delta[0, -1, h, :]).sum().item()

        return attributions, clean_ld

    # ──────────────────────────────────────────────────────────────────────
    # 2. COMPREHENSIVENESS — CORRUPTED ACTIVATION PATCHING
    #
    #    Formula: Comp = 1 - LD(clean_run | circuit=corrupted) / LD_clean
    #
    #    For each (layer, head) in circuit, replace z_h^clean with z_h^corr.
    #    If the circuit is causally important, LD drops → Comp is high.
    #    This is precisely what Wang et al. 2022 used for the IOI circuit.
    #
    #    Why NOT zero ablation:
    #      Zeroing removes both signal AND anchoring baseline. Model overcompensates
    #      → LD stays flat or increases → misleading Comp ≈ 0%.
    #    Why NOT mean ablation:
    #      Mean over sequence positions preserves residual signal → same problem.
    # ──────────────────────────────────────────────────────────────────────

    def _comp(self, circuit, clean_tokens, corrupted_tokens,
               clean_ld, target_token, distractor_token):
        """
        Corrupted activation patching comprehensiveness.
        Caches corrupted z for circuit layers, patches into clean forward pass.
        """
        if not circuit or clean_ld == 0:
            return 0.0

        needed_layers = list(set(l for l, h in circuit))

        # Cache corrupted activations for circuit layers only
        corr_cache = {}
        def make_save(key):
            def hook(act, hook):
                corr_cache[key] = act.detach().clone()
            return hook

        with torch.no_grad():
            self.model.run_with_hooks(corrupted_tokens,
                fwd_hooks=[(f'blocks.{l}.attn.hook_z',
                            make_save(f'blocks.{l}.attn.hook_z'))
                           for l in needed_layers])

        # Patch: replace clean z_h with corrupted z_h for circuit heads
        def make_patch_corr(layer, head):
            key = f'blocks.{layer}.attn.hook_z'
            def hook(act, hook):
                result = act.clone()
                if key in corr_cache:
                    result[:, :, head, :] = corr_cache[key][:, :, head, :]
                return result
            return hook

        hooks = [(f'blocks.{l}.attn.hook_z', make_patch_corr(l, hd))
                 for l, hd in circuit]

        with torch.no_grad():
            patched_logits = self.model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        patched_ld = (patched_logits[0, -1, target_token]
                      - patched_logits[0, -1, distractor_token]).item()
        comp = 1.0 - (patched_ld / clean_ld)
        return float(max(0.0, min(1.0, comp)))

    # ──────────────────────────────────────────────────────────────────────
    # 3. MINIMUM FAITHFUL CIRCUIT
    # ──────────────────────────────────────────────────────────────────────

    def minimum_faithful_circuit(self, clean_tokens, corrupted_tokens,
                                  target_token, distractor_token,
                                  target_suff=0.85, target_comp=0.15):
        attributions, clean_ld = self.attribution_patching(
            clean_tokens, corrupted_tokens, target_token, distractor_token)
        if clean_ld == 0:
            return [], attributions

        # Phase 1: greedy forward — add heads until suff ≥ target_suff
        ranked = sorted([(k, v) for k, v in attributions.items() if v > 0],
                        key=lambda x: x[1], reverse=True)
        if not ranked:
            ranked = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        candidate = []
        for head, attr in ranked:
            candidate.append(head)
            total_attr = sum(attributions.get(h, 0) for h in candidate)
            suff = min(1.0, max(0.0, total_attr / clean_ld))
            if suff >= target_suff:
                break

        # Phase 2: greedy backward — prune if comp still ≥ target_comp
        circuit = list(candidate)
        for head in reversed(list(candidate)):
            trial = [h for h in circuit if h != head]
            if not trial:
                break
            comp = self._comp(trial, clean_tokens, corrupted_tokens,
                               clean_ld, target_token, distractor_token)
            if comp >= target_comp:
                circuit = trial

        return circuit, attributions

    # ──────────────────────────────────────────────────────────────────────
    # 4. BOOTSTRAP CONFIDENCE INTERVALS
    # ──────────────────────────────────────────────────────────────────────

    def bootstrap_metrics(self, prompts, n_boot=500, alpha=0.05):
        suff_vals, comp_vals, f1_vals = [], [], []

        for idx, (prompt, correct, incorrect) in enumerate(prompts):
            print(f"  Bootstrap {idx+1}/{len(prompts)}: '{prompt[:50]}...'")
            try:
                t_tok = self.model.to_single_token(correct)
                d_tok = self.model.to_single_token(incorrect)
            except Exception:
                print(f"    ⚠️  Skipping multi-token: '{correct}'")
                continue

            tokens_c    = self.model.to_tokens(prompt)
            tokens_corr = self.model.to_tokens(
                prompt.replace(correct.strip(), incorrect.strip()))

            circuit, attrs = self.minimum_faithful_circuit(
                tokens_c, tokens_corr, t_tok, d_tok)
            _, clean_ld = self.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)

            if not circuit or clean_ld == 0:
                print(f"    ⚠️  Empty circuit / zero LD — skipping")
                continue

            total = sum(attrs.get(h, 0) for h in circuit)
            suff  = float(np.clip(total / clean_ld, 0, 1))
            comp  = self._comp(circuit, tokens_c, tokens_corr, clean_ld, t_tok, d_tok)
            f1    = 2 * suff * comp / (suff + comp) if (suff + comp) > 0 else 0.0

            suff_vals.append(suff); comp_vals.append(comp); f1_vals.append(f1)
            print(f"    Suff={suff:.1%}  Comp={comp:.1%}  F1={f1:.1%}  circuit={len(circuit)} heads")

        if len(suff_vals) < 2:
            return {'error': f'Only {len(suff_vals)} valid prompts — need ≥ 2'}

        def boot_ci(vals):
            arr  = np.array(vals)
            boot = np.array([np.mean(np.random.choice(arr, len(arr), replace=True))
                             for _ in range(n_boot)])
            return {'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
                    'ci_lo': float(np.percentile(boot, 100 * alpha / 2)),
                    'ci_hi': float(np.percentile(boot, 100 * (1 - alpha / 2))),
                    'n': len(arr)}

        return {'sufficiency':       boot_ci(suff_vals),
                'comprehensiveness': boot_ci(comp_vals),
                'f1':                boot_ci(f1_vals)}

    # ──────────────────────────────────────────────────────────────────────
    # 5. SINGLE-CALL ANALYZE API
    # ──────────────────────────────────────────────────────────────────────

    def analyze(self, prompt, correct, incorrect):
        try:
            t_tok = self.model.to_single_token(correct)
            d_tok = self.model.to_single_token(incorrect)
        except Exception:
            t_tok = self.model.to_tokens(correct)[0, -1].item()
            d_tok = self.model.to_tokens(incorrect)[0, -1].item()

        tokens_c    = self.model.to_tokens(prompt)
        tokens_corr = self.model.to_tokens(
            prompt.replace(correct.strip(), incorrect.strip()))

        circuit, attrs = self.minimum_faithful_circuit(
            tokens_c, tokens_corr, t_tok, d_tok)
        _, clean_ld = self.attribution_patching(tokens_c, tokens_corr, t_tok, d_tok)

        total = sum(attrs.get(h, 0) for h in circuit)
        suff  = float(np.clip(total / clean_ld, 0, 1)) if clean_ld != 0 else 0.0
        comp  = self._comp(circuit, tokens_c, tokens_corr, clean_ld, t_tok, d_tok)
        f1    = 2 * suff * comp / (suff + comp) if (suff + comp) > 0 else 0.0

        if   suff > 0.9 and comp < 0.4:  category = 'backup_mechanisms'
        elif suff > 0.7 and comp > 0.5:  category = 'faithful'
        elif suff < 0.6 and comp < 0.5:  category = 'weak'
        elif suff < 0.5:                  category = 'incomplete'
        else:                              category = 'moderate'

        return {
            'circuit':    circuit,
            'n_heads':    len(circuit),
            'clean_ld':   clean_ld,
            'attributions': {str(k): v for k, v in attrs.items()},
            'faithfulness': {
                'sufficiency': suff, 'comprehensiveness': comp,
                'f1': f1, 'category': category
            }
        }


print("✅ GlassboxV2 engine loaded.")
print("   Comprehensiveness: CORRUPTED ACTIVATION PATCHING (Wang et al. 2022)")
print("   More causally precise than zero/mean ablation for transformer circuits")