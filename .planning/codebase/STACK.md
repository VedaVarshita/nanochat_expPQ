# STACK.md — Technology Stack

## Language & Runtime
- **Python** ≥ 3.10 (primary language)
- **Package manager**: `uv` (astral.sh) with `pyproject.toml`
- **JavaScript/Node**: used only by the GSD plugin (`.claude/get-shit-done/bin/gsd-tools.cjs`)

## Core Framework
- **PyTorch** `2.9.1` — the only ML framework; no Lightning, Keras, etc.
  - Explicit mixed-precision via `COMPUTE_DTYPE` (no `torch.amp.autocast`)
  - `torch.compile` used in optimizer fused kernels (`nanochat/optim.py`)
  - `torch.distributed` for DDP multi-GPU training
  - PyTorch SDPA as fallback attention (always available)

## Key Dependencies (`pyproject.toml`)
| Package | Purpose |
|---------|---------|
| `torch==2.9.1` | Core ML framework |
| `rustbpe>=0.1.0` | Fast Rust-based BPE tokenizer |
| `tiktoken>=0.11.0` | GPT-4 tokenizer (split pattern reference) |
| `tokenizers>=0.22.0` | HuggingFace tokenizers (BPE training) |
| `datasets>=4.0.0` | HuggingFace datasets (task data loading) |
| `wandb>=0.21.3` | Experiment tracking / metrics logging |
| `fastapi>=0.117.1` | Web chat server (`scripts/chat_web.py`) |
| `uvicorn>=0.36.0` | ASGI server for FastAPI |
| `kernels>=0.11.7` | Flash Attention 3 kernel loader (Hopper only) |
| `psutil>=7.1.0` | System resource monitoring |
| `jinja2` | Prompt template rendering in `nanochat/core_eval.py` |

## Optional / Hardware-Specific Features
| Feature | Requirement | Notes |
|---------|-------------|-------|
| Flash Attention 3 | Hopper GPU (sm90) | Loaded via `kernels` pkg from HF Hub |
| FP8 training | CUDA (any) | Custom `nanochat/fp8.py` (~150 lines, replaces torchao) |
| bfloat16 | CUDA sm80+ (Ampere+) | Auto-detected |
| float32 | CPU / MPS / pre-Ampere | Auto-detected fallback |

## Dev Dependencies (`[dependency-groups] dev`)
- `pytest>=8.0.0` — testing
- `matplotlib>=3.10.8` — plotting (scaling laws, etc.)
- `ipykernel>=7.1.0` — notebook support
- `transformers>=4.57.3` — reference models for evaluation
- `python-dotenv>=1.2.1` — env var loading

## Hardware Targets
| Hardware | dtype | Attention | FP8 |
|----------|-------|-----------|-----|
| H100 / Hopper (sm90) | bfloat16 | Flash Attention 3 | Yes |
| A100 / Ampere (sm80+) | bfloat16 | PyTorch SDPA | Yes |
| V100 / pre-Ampere (sm<80) | float32 | PyTorch SDPA | No |
| CPU | float32 | PyTorch SDPA | No |
| Apple Silicon (MPS) | float32 | PyTorch SDPA | No |

## Configuration
- **`NANOCHAT_DTYPE`** — override compute dtype (`float32`, `bfloat16`, `float16`)
- **`NANOCHAT_BASE_DIR`** — override data/checkpoint base dir (default `~/.cache/nanochat`)
- **`WANDB_RUN=dummy`** — disable wandb logging
- **`OMP_NUM_THREADS`** — CPU thread parallelism
- **`PYTORCH_ALLOC_CONF=expandable_segments:True`** — set at top of training scripts

## PyTorch Index Sources
```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", extra = "cpu" },   # https://download.pytorch.org/whl/cpu
  { index = "pytorch-cu128", extra = "gpu" }, # https://download.pytorch.org/whl/cu128
]
```
`cpu` and `gpu` extras are mutually exclusive conflicts in `[tool.uv]`.
