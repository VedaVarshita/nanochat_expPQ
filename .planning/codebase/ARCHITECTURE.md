# ARCHITECTURE.md — System Architecture

## Pattern
**Monolithic script-based pipeline** — no service mesh, no plugin registry, no dependency injection. Each pipeline stage is a standalone Python module invoked directly. Shared logic lives in the `nanochat/` package.

## Pipeline Stages (sequential)
```
1. Tokenizer Training     scripts/tok_train.py
         ↓
2. Base Pretraining       scripts/base_train.py   ← main focus (speedrun leaderboard)
         ↓
3. Supervised Fine-Tuning scripts/chat_sft.py
         ↓
4. RL Fine-Tuning         scripts/chat_rl.py      (optional)
         ↓
5. Inference / Chat       scripts/chat_cli.py
                          scripts/chat_web.py
```

## Core Abstractions

### GPT model (`nanochat/gpt.py`)
- `GPTConfig` dataclass — all model hyperparameters
- `GPT(nn.Module)` — transformer with:
  - `CausalSelfAttention` — RoPE, GQA support, sliding window (pattern `SSSL` or `L`)
  - `MLP` — standard 4× expansion FFN
  - `Block` — attention + MLP + residual scaling (`resid_lambdas`, `x0_lambdas`)
  - Value embeddings (VE) — early fusion on alternating layers
  - Backout + smear (autoresearch round 2 additions)
- `Linear` — custom layer that casts weights to `COMPUTE_DTYPE` on forward (replaces `autocast`)
- Single depth dial: `model_dim = depth × aspect_ratio`, `num_heads = model_dim / head_dim`

### Data Loading (`nanochat/dataloader.py`)
- `tokenizing_distributed_data_loader_*` — streaming tokenizer that reads parquet shards, tokenizes on-the-fly, yields batches with BOS token packing
- Stateful variant for checkpoint resumption

### Optimizer (`nanochat/optim.py`)
- `MuonAdamW` (single GPU) / `DistMuonAdamW` (DDP) — mixed optimizer:
  - **Muon** for matrix parameters (weight matrices)
  - **AdamW** for embeddings, unembeddings, scalars
  - Fused AdamW step via `@torch.compile`
- Adapted from modded-nanoGPT

### Inference Engine (`nanochat/engine.py`)
- `KVCache` — manages key/value cache across layers for autoregressive generation
- `Engine` — wraps model + KV cache, handles token-by-token generation
- Optional Python code execution tool via `nanochat/execution.py`

### Evaluation (`nanochat/core_eval.py`, `nanochat/loss_eval.py`)
- `evaluate_bpb()` — validation bits-per-byte (vocab-size-invariant loss metric)
- `evaluate_core()` — DCLM CORE score (22-task ensemble, calls `scripts/base_eval.py`)

### Checkpoint Manager (`nanochat/checkpoint_manager.py`)
- Saves/loads: `model_NNNNNN.pt`, `optimizer_NNNNNN.pt`, `meta_NNNNNN.json`
- Patches missing keys for backward compat with old checkpoints
- Location: `~/.cache/nanochat/base_checkpoints/<model-tag>/`

### Precision / dtype (`nanochat/common.py`)
- `COMPUTE_DTYPE` — global auto-detected dtype (bfloat16 on sm80+, float32 otherwise)
- No `torch.amp.autocast` — all casting explicit in `Linear.forward()`

## Training Loop (`scripts/base_train.py`)
```
build model on meta device → move to device → init weights
      ↓
setup optimizer (Muon + AdamW param groups)
      ↓
[optional] torch.compile model
      ↓
[optional] DDP wrap (torchrun multi-GPU)
      ↓
for step in range(num_iterations):
    gradient accumulation loop (device_batch_size × grad_accum = total_batch_size)
    optimizer.step()
    LR schedule (warmup → cosine warmdown)
    periodic: val_bpb eval, CORE eval, sample, checkpoint
```

## Web Server (`scripts/chat_web.py`)
- FastAPI + uvicorn
- Worker pool: one model copy per GPU, requests load-balanced
- Streaming SSE responses for token-by-token output
- Abuse prevention: message/char limits, temperature/top-k clamping

## Distributed Training
- `torchrun --nproc_per_node=N` → DDP via `torch.distributed`
- `compute_init()` in `nanochat/common.py` handles DDP setup + device assignment
- Single-GPU fallback: automatic gradient accumulation compensates for missing GPUs
- CPU/MPS: always single process (no multi-process DDP)

## Data Flow
```
Parquet shards (disk)
    → tokenizing_distributed_data_loader (streaming, multi-process)
    → token batches [B, T] on device
    → GPT forward → logits [B, T, V]
    → cross-entropy loss
    → backward → Muon/AdamW step
```
