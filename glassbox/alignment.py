import numpy as np


def fcas(circuit_a, circuit_b, n_layers_a, n_layers_b, top_k=5):
    """
    Functional Circuit Alignment Score (FCAS).

    Compares circuits from two models by normalizing head positions to
    relative depth [0, 1] and computing mean absolute deviation.

    Args:
        circuit_a:  dict {(layer, head): attribution_score}  -- model A
        circuit_b:  dict {(layer, head): attribution_score}  -- model B
        n_layers_a: total layer count of model A
        n_layers_b: total layer count of model B
        top_k:      how many top heads to compare (default 5)

    Returns:
        float in [0, 1].  1.0 = perfect alignment, 0.0 = no alignment.

    Example:
        >>> from glassbox.alignment import fcas
        >>> score = fcas(circuit_small, circuit_medium, 12, 24, top_k=5)
        >>> print(f"FCAS = {score:.3f}")   # e.g. 0.929
    """
    top_a = sorted(circuit_a.items(), key=lambda x: -abs(x[1]))[:top_k]
    top_b = sorted(circuit_b.items(), key=lambda x: -abs(x[1]))[:top_k]

    rel_a = [layer / (n_layers_a - 1) for (layer, _), _ in top_a]
    rel_b = [layer / (n_layers_b - 1) for (layer, _), _ in top_b]

    k = min(len(rel_a), len(rel_b))
    if k == 0:
        return 0.0

    return float(1.0 - np.mean([abs(rel_a[i] - rel_b[i]) for i in range(k)]))
