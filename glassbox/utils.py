"""
glassbox/utils.py — Shared utility functions used across Glassbox modules.

Functions here are internal helpers unless documented with a public docstring.
They are NOT part of the public API and may change between minor versions.
"""
from __future__ import annotations

import functools
import logging
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import torch

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Hook factory functions (used in core.py and tests)
# ---------------------------------------------------------------------------

def make_cache_hook(cache: Dict, key: str) -> Callable:
    """Return a TransformerLens hook that stores ``value`` in ``cache[key]``."""
    def hook(value: torch.Tensor, hook=None) -> None:  # noqa: ARG001
        cache[key] = value.detach()
    return hook


def make_patch_hook(source_cache: Dict, key: str, head_idx: int) -> Callable:
    """Return a hook that patches ``value[:, :, head_idx, :]`` from ``source_cache``."""
    def hook(value: torch.Tensor, hook=None) -> torch.Tensor:  # noqa: ARG001
        value[:, :, head_idx, :] = source_cache[key][:, :, head_idx, :]
        return value
    return hook


# ---------------------------------------------------------------------------
# Logit difference
# ---------------------------------------------------------------------------

def logit_diff(
    logits:           torch.Tensor,
    target_token:     int,
    distractor_token: int,
) -> float:
    """
    Compute logit(target) - logit(distractor) at the last sequence position.

    Parameters
    ----------
    logits           : Float tensor of shape [batch, seq_len, d_vocab].
    target_token     : Vocabulary index of the correct/target token.
    distractor_token : Vocabulary index of the competing distractor token.

    Returns
    -------
    float -- mean logit difference across the batch.
    """
    last = logits[:, -1, :]
    return (last[:, target_token] - last[:, distractor_token]).mean().item()


# ---------------------------------------------------------------------------
# Token normalisation
# ---------------------------------------------------------------------------

def normalize_token(
    model:   Any,
    token:   "str | int",
    role:    str = "token",
) -> int:
    """
    Convert a token string or pre-resolved int to a vocabulary index.

    Parameters
    ----------
    model : HookedTransformer (or anything with ``to_single_token``).
    token : Either a vocabulary string (e.g. " Mary") or a pre-resolved int.
    role  : Human-readable label used in error messages (e.g. "target_token").

    Returns
    -------
    int vocabulary index.

    Raises
    ------
    TypeError  -- if ``token`` is neither str nor int.
    ValueError -- if the string maps to multiple tokens (multi-token string).
    """
    if isinstance(token, int):
        return token
    if isinstance(token, str):
        try:
            return int(model.to_single_token(token))
        except Exception as exc:
            # Fall back: take the last sub-token of the string
            fallback = int(model.to_tokens(token)[0, -1].item())
            logger.debug(
                "normalize_token: '%s' is multi-token for %s -- using last sub-token %d (%s)",
                token, role, fallback, exc,
            )
            return fallback
    raise TypeError(
        f"normalize_token: {role} must be str or int, got {type(token).__name__!r}"
    )


# ---------------------------------------------------------------------------
# Head label formatting
# ---------------------------------------------------------------------------

def format_head_label(layer: int, head: int) -> str:
    """Return a compact head label like ``'L09H09'``."""
    return f"L{layer:02d}H{head:02d}"


def parse_head_label(label: str) -> Tuple[int, int]:
    """
    Parse a head label produced by :func:`format_head_label` back to (layer, head).

    Raises ValueError if the label does not match the expected format.
    """
    if not (label.startswith("L") and "H" in label):
        raise ValueError(
            f"Cannot parse head label {label!r} -- expected format 'L<layer>H<head>'"
        )
    l_part, h_part = label[1:].split("H", 1)
    return int(l_part), int(h_part)


# ---------------------------------------------------------------------------
# @stable_api decorator -- marks public API surface
# ---------------------------------------------------------------------------

def stable_api(fn: F) -> F:
    """
    Decorator that marks a function / method as part of Glassbox's stable
    public API.

    Stable API functions:
    * Will not be removed without a major version bump.
    * Will not have their signature changed in a backward-incompatible way
      without a deprecation period of at least one minor version.

    The decorator is a no-op at runtime -- it exists purely as a signal to
    readers, documentation generators, and future maintainers.
    """
    fn.__glassbox_stable__ = True  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------------------
# Deprecation helper
# ---------------------------------------------------------------------------

def deprecated(
    replacement: Optional[str] = None,
    since: str = "",
) -> Callable[[F], F]:
    """
    Decorator that emits a DeprecationWarning when the decorated callable is
    invoked.

    Parameters
    ----------
    replacement : Name of the recommended replacement (shown in warning).
    since       : Version since which the function has been deprecated.
    """
    def decorator(fn: F) -> F:
        msg_parts = [f"{fn.__qualname__} is deprecated"]
        if since:
            msg_parts.append(f"(since v{since})")
        if replacement:
            msg_parts.append(f"-- use {replacement} instead")
        msg = " ".join(msg_parts) + "."

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__deprecated__ = True  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Memory estimation (rough heuristic)
# ---------------------------------------------------------------------------

def estimate_forward_pass_memory_mb(
    n_layers:    int,
    n_heads:     int,
    d_model:     int,
    seq_len:     int,
    n_passes:    int = 3,
    dtype_bytes: int = 4,
) -> float:
    """
    Rough estimate of peak memory for ``n_passes`` forward passes (MB).

    The dominant terms are:
    * Attention patterns: [batch=1, n_heads, seq, seq] x n_layers x n_passes
    * Residual stream:    [batch=1, seq, d_model] x (n_layers+1) x n_passes

    This is a lower bound -- actual usage is higher due to intermediate
    activations and framework overhead.
    """
    attn_bytes     = n_layers * n_heads * seq_len * seq_len * dtype_bytes * n_passes
    residual_bytes = (n_layers + 1) * seq_len * d_model * dtype_bytes * n_passes
    return (attn_bytes + residual_bytes) / (1024 ** 2)
