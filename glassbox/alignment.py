import numpy as np

def fcas(circuit_a, circuit_b, n_layers_a, n_layers_b, top_k=5):
    """Functional Circuit Alignment Score — cross-model alignment metric."""
    top_a = sorted(circuit_a.items(), key=lambda x: -abs(x[1]))[:top_k]
    top_b = sorted(circuit_b.items(), key=lambda x: -abs(x[1]))[:top_k]
    rel_a = [layer / (n_layers_a - 1) for (layer, _), _ in top_a]
    rel_b = [layer / (n_layers_b - 1) for (layer, _), _ in top_b]
    k = min(len(rel_a), len(rel_b))
    if k == 0:
        return 0.0
    return float(1.0 - np.mean([abs(rel_a[i] - rel_b[i]) for i in range(k)]))
