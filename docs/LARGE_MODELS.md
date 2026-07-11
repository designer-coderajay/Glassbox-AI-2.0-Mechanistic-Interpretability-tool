# Large Model Support — Glassbox AI v4.3.1

Complete guide for running Glassbox circuit analysis on billion-parameter models:
Llama-3-70B, Mistral-7B, Gemma-7B, Phi-3, Qwen2, and any HuggingFace transformer.

---

## Quick start

```python
from glassbox.large_model import analyze_large, estimate_memory
import torch

# Step 1: predict VRAM before loading anything
mem = estimate_memory(n_layers=80, d_model=8192, dtype=torch.bfloat16)
print(mem)
# → Parameters: 70.0B (53.2 GB at 2-byte dtype)
# → Attribution (3 passes): 53.0 GB
# → Recommended strategy: checkpoint+offload

# Step 2: load with bfloat16 (halves VRAM)
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    dtype=torch.bfloat16,
    device="cuda",
)
from glassbox import GlassboxV2
gb = GlassboxV2(model)

# Step 3: run analysis — identical to gb.analyze()
result = analyze_large(
    gb,
    prompt    = "Loan application. Annual income: €42,000. Decision:",
    correct   = " Approved",
    incorrect = " Denied",
    strategy  = "auto",          # auto-selects gradient checkpointing
    dtype     = torch.bfloat16,
)
print(result["faithfulness"]["f1"])
print(result["corruption_metadata"]["strategy"])
```

---

## Memory requirements by model

| Model | Params | VRAM (bf16) | Strategy | Notes |
|---|---|---|---|---|
| GPT-2 small | 117M | ~0.5 GB | standard | default, no special handling |
| Pythia 1B | 1B | ~2 GB | standard | fits any modern GPU |
| Llama-2-7B | 7B | ~8 GB | checkpoint | A10G or better |
| Llama-3-8B | 8B | ~8.5 GB | checkpoint | A10G or better |
| Mistral-7B | 7B | ~8 GB | checkpoint | A10G or better |
| Phi-3-mini | 3.8B | ~4 GB | checkpoint | RTX 3090 or better |
| Gemma-7B | 7B | ~8 GB | checkpoint | A10G or better |
| Llama-3-70B | 70B | ~53 GB | checkpoint+offload | A100 80GB or 2×A10G |
| Llama-2-70B | 70B | ~53 GB | checkpoint+offload | A100 80GB or 2×A10G |

The attribution pass needs approximately **3× the model weight VRAM** (clean pass + corrupted pass + gradient pass). Strategy auto-selection handles this.

---

## Strategy selection

### `standard` — models under 7B, fits in GPU
No special handling. Direct attribution patching.

### `checkpoint` — 7B–70B models on A100/H100
Gradient checkpointing per Chen et al. 2016.
- Reduces peak activation memory by √n_layers (≈4–11×)
- +33% compute cost (each checkpoint segment computed twice during backward)
- Optimal checkpoint frequency: √n_layers layers apart

### `checkpoint+offload` — 70B+ models on limited VRAM
Gradient checkpointing **plus** activation offloading to CPU RAM.
- PCIe bandwidth (~32 GB/s) adds ~10–20ms per layer
- Acceptable for compliance auditing (not real-time inference)
- Uses ZeRO-Offload approach (Rajbhandari et al. 2020)

---

## CLI usage

```bash
# Check memory before loading
glassbox-ai estimate-memory --n-layers 80 --d-model 8192 --dtype bfloat16

# Analyze with large model
glassbox-ai analyze \
    --model meta-llama/Llama-3-8B \
    --prompt "Patient presents with chest pain. Priority:" \
    --correct " Urgent" \
    --incorrect " Routine" \
    --dtype bfloat16 \
    --strategy auto
```

---

## Docker — no GPU required for small models

```bash
docker build -f scan.Dockerfile -t glassbox-scan .

# GPT-2 (CPU only — no GPU needed)
docker run --rm -v $(pwd)/output:/output \
    glassbox-scan \
    --model gpt2 \
    --prompt "Loan application. Annual income: €42,000. Decision:" \
    --correct " Approved" --incorrect " Denied" \
    --purpose "Credit risk scoring" \
    --provider "Your Organisation Name" \
    --output /output/annex_iv.pdf
```

For GPU acceleration with Docker, add `--gpus all` and ensure NVIDIA Container Toolkit is installed.

---

## Supported architectures

| Family | n_kv_heads | Norm | Notes |
|---|---|---|---|
| gpt2 | = n_heads | LayerNorm | GPT-2 small → XL |
| pythia | = n_heads | LayerNorm | EleutherAI Pythia 70M–12B |
| gpt-j | = n_heads | LayerNorm | GPT-J 6B |
| llama-2 | = n_heads | RMSNorm | Llama-2 7B/13B/70B |
| llama-3 | n_heads/4 (GQA) | RMSNorm | Llama-3-8B |
| llama-3-70b | n_heads/8 (GQA) | RMSNorm | Llama-3-70B |
| mistral | n_heads/4 (GQA) | RMSNorm | Mistral-7B |
| phi-2 | = n_heads | LayerNorm | Phi-2 |
| phi-3 | n_heads/4 (GQA) | RMSNorm | Phi-3-mini |
| gemma | n_heads/8 (GQA) | RMSNorm | Gemma 2B/7B |
| qwen2 | n_heads/4 (GQA) | RMSNorm | Qwen2-7B |

GQA models use `GQAAttentionMapper` to correctly redistribute attribution scores across query head groups. All 11 families are verified with 183 unit tests.
