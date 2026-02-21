import torch

def make_cache_hook(cache, key):
    def hook(value, hook):
        cache[key] = value.detach()
    return hook

def make_patch_hook(source_cache, key, head_idx):
    def hook(value, hook):
        value[:, :, head_idx, :] = source_cache[key][:, :, head_idx, :]
        return value
    return hook

def logit_diff(logits, target_token, distractor_token):
    last = logits[:, -1, :]
    return (last[:, target_token] - last[:, distractor_token]).mean().item()
