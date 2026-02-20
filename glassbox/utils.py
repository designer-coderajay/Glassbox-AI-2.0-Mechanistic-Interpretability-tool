"""
Hook utilities for TransformerLens activation caching and patching.
Used internally by GlassboxV2; also importable for custom analyses.
"""


def make_cache_hook(cache, key):
    """Returns a hook that stores activations in cache[key]."""
    def hook(act, hook):
        cache[key] = act.detach().clone()
        return act
    return hook


def make_patch_hook(source_cache, key, head_idx):
    """
    Returns a hook that replaces one attention head's activations
    with values from source_cache (used for comprehensiveness).
    """
    def hook(act, hook):
        result = act.clone()
        result[:, :, head_idx, :] = source_cache[key][:, :, head_idx, :]
        return result
    return hook


def logit_diff(logits, target_token, distractor_token):
    """
    Scalar logit difference at the last sequence position.
    LD = logit[target] - logit[distractor]
    """
    return float(
        logits[0, -1, target_token] - logits[0, -1, distractor_token]
    )
