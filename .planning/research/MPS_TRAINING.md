# MPS_TRAINING.md — PyTorch MPS Training Compatibility Research

**Researched against:** PyTorch 2.9.1, codebase files `common.py`, `gpt.py`, `optim.py`, `flash_attention.py`, `base_train.py`, `runcpu.sh`

## 1. MPS vs CUDA Limitations

**float32 is mandatory.** `common.py` already auto-detects this correctly — `_detect_compute_dtype()` returns `torch.float32` for the no-CUDA path. bfloat16 on MPS has storage-level support but numerically unreliable matmul. Never set `NANOCHAT_DTYPE=bfloat16` on MPS.

**Unified memory.** No separate VRAM pool. `PYTORCH_ALLOC_CONF=expandable_segments:True` (already set at line 15 of `base_train.py`) is the single most important memory tuning. On M3 Max (48+ GB), the runcpu.sh config (depth=6, seq=512, batch=32) uses an estimated 1–2 GB — comfortably fits.

**No DDP.** MPS does not support NCCL or Gloo distributed backends. `DistMuonAdamW` cannot run on MPS. The codebase correctly routes to `MuonAdamW` on non-DDP paths.

**Unsupported ops that matter:**
- `.item()` on MPS tensor forces a CPU-GPU sync (command queue drain). The training loop calls this every step via `train_loss.item()`. Unavoidable, but serializes the Metal queue.
- `dist.*` collective ops — completely unsupported on MPS.
- `torch.cuda.max_memory_allocated` — returns 0 on MPS (`lambda: 0` in `base_train.py`). Memory is invisible to the training logger on MPS.

## 2. SDPA on MPS

`F.scaled_dot_product_attention` is supported and uses a MPS-native kernel (not FlashAttention). The `is_causal=True` fast path works. `enable_gqa=True` is supported in PyTorch 2.2+. The codebase already uses correct patterns in `flash_attention.py`.

**Note:** Sliding window via explicit boolean mask materializes a full T×T tensor on MPS — no kernel-level sparsity. `runcpu.sh` uses `--window-pattern=L` (full causal, fast path). Do not add sliding window patterns without measuring cost.

## 3. `torch.compile` on MPS — Primary Risk

`torch.compile` on MPS uses the inductor backend targeting Metal. Works for simple static-shape graphs but is incomplete compared to CUDA inductor.

**Highest-risk components:** `adamw_step_fused` and `muon_step_fused` in `optim.py` are both decorated with `@torch.compile(dynamic=False, fullgraph=True)`. Two specific risks:

1. **0-D CPU tensor inputs.** The optimizer passes 0-D CPU tensors (lr, momentum scalars) into compiled graph nodes. On MPS, the compiled Metal kernel may not handle CPU-tensor-to-MPS-kernel parameter passing the same way as CUDA, potentially causing a graph break or silent eager fallback.

2. **`torch._foreach_copy_`** in Muon's param copy-back had incomplete MPS support in earlier 2.x releases. PyTorch 2.9.1 is likely fixed but is a candidate for the first fallback check.

**If step 0 hangs:** This is Metal shader compilation warmup, not a real hang. Expected wait on M3 Max at depth=6: ~30–90 seconds. Do not interrupt.

**If optimizer kernels fail to compile:** The Python function bodies run correctly in eager mode — the `@torch.compile` decorator only adds fusion. Remove it from `optim.py` temporarily to diagnose.

## 4. Transformer Ops Compatibility

| Op | Status | Notes |
|----|--------|-------|
| `F.rms_norm` | Supported (PyTorch 2.3+) | 2.9.1 is fine |
| RoPE (element-wise mul + cat) | Supported | Pure basic ops |
| `F.cross_entropy` (float32) | Supported | Model already casts logits to float32 |
| softcap (tanh) | Supported | |
| `nn.Embedding` | Supported | |
| sigmoid, relu, `.square()` | Supported | |
| `torch.Generator(device="mps")` | Supported | Used in generate() |

## 5. Performance Tips

```bash
# Add to runcpu.sh before base_train call
export OMP_NUM_THREADS=8       # Tune for M-series (M3 Max: 12+4 cores)
export PYTORCH_ALLOC_CONF="expandable_segments:True"  # Already in base_train.py
```

**Accurate timing on MPS:** Current no-op `synchronize = lambda: None` means `dt` is loose — GPU work may not be complete when timing is read. Acceptable noise for 30-min runs, not a correctness issue.

**MPS memory monitoring (not in training output):**
```python
torch.mps.current_allocated_memory() / 1e9  # GB
```

## 6. Codebase-Specific Notes

- **`muon_step_fused` bf16 path:** On MPS, `COMPUTE_DTYPE` is float32 so the `.bfloat16()` cast is never taken. Entire optimizer runs in float32. Correct.
- **`engine.py` KVCache dtype:** `dtype = torch.bfloat16 if device.type == "cuda" else torch.float32` — KV cache will be float32 on MPS. Correct.
- **`gc.freeze()` / `gc.disable()`** at step 0 — good practice for MPS too, removes GC overhead during Metal command queue hot path.
- **`tokenizing_distributed_data_loader`** uses a main-process generator, no `DataLoader` workers. Sidesteps all macOS `spawn` multiprocessing issues entirely.

## Open Questions (Need Empirical Verification at Run Time)

1. Does `@torch.compile(fullgraph=True)` on `adamw_step_fused` / `muon_step_fused` succeed on MPS in PyTorch 2.9.1?
2. Actual Metal shader warmup time for depth=6 on target hardware
3. Does `torch._foreach_copy_` have correctness issues on MPS 2.9.1?
4. Is `torch.mps.synchronize()` available in PyTorch 2.9.1?

---
*Researched: 2026-04-02*
