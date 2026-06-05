# Changes Since Fork

All modifications made to [karpathy/nanochat](https://github.com/karpathy/nanochat) in this fork (`nanochat_expPQ`), in reverse chronological order.

---

## Uncommitted (current session)

### `scripts/encoder_baseline.py` — new file
Baseline training script for `GPTEncoder` using **next-token prediction** — identical objective to `base_train.py` for a fair apples-to-apples comparison.
- Architecture: `GPTEncoder` backbone + external `stub_head = Linear(n_embd, vocab)` + softcap + cross-entropy. Only structural difference vs. `base_train.py` is that `lm_head` lives outside the backbone.
- Optimizer: MuonAdamW for encoder, AdamW for stub_head (mirrors `lm_head` treatment in `base_train.py`).
- Scaling laws, LR/momentum/weight-decay schedulers, DDP, FP8, checkpointing, wandb: identical to `base_train.py`.
- Checkpoint dir: `base_checkpoints/enc_baseline_d{depth}/`
- `--core-metric-every` and `--sample-every` default to `-1` (no generation without lm_head).
- Compare val bpb from this run against `base_train.py` at the same depth to isolate architecture effect.

### `scripts/encoder_pretrain.py` + `docs/encoder_head_design.md` — BoxedLayer target assignment (k-means style)
Added `BoxedLayer` stub class and a 20-epoch target-cache refresh schedule between the encoder hidden states and `one_hot_matrix`.
- `BoxedLayer.__call__(hidden) → (B, T) class indices`: user-implemented nearest-neighbor / custom NN. Called with `hidden.detach()` under `torch.no_grad()` — no backprop flows through it.
- **Refresh schedule**: epoch 0 and every 20th epoch rebuild `target_cache` by running BoxedLayer on each training micro-batch. The 19 epochs in between replay the frozen cache by `micro_step_global % len(target_cache)`.
- State: `target_cache`, `cache_built_for_epoch`, `micro_step_global`, `prev_dataloader_epoch`
- `docs/encoder_head_design.md` updated with BoxedLayer interface, data flow diagram, refresh schedule, and state variable table.

### `scripts/encoder_pretrain.py` — replace `stub_head` with fixed `one_hot_matrix`
Replaced the learned linear head (`stub_head = Linear(n_embd, vocab_size)`) with a user-filled fixed matrix `one_hot_matrix = zeros(NUM_LABELS, n_embd)`.
- `NUM_LABELS` is a user-configurable constant (default 128), independent of `vocab_size`
- Projection: `hidden @ one_hot_matrix.T` → `(B, T, NUM_LABELS)` logits → softcap → cross-entropy
- Matrix is `requires_grad=False` — not learned, not saved in checkpoints. User fills rows before training.
- Removed `stub_head_optimizer`, all associated `step()`/`zero_grad()`/`eval()`/`train()` calls, and the `stub_head.*` checkpoint prefix
- Open decision documented: target format (integer indices vs. one-hot float vectors) — see `docs/encoder_head_design.md`

### `docs/encoder_head_design.md` — new file
Design record comparing Option 1 (learned `stub_head`) vs Option 2 (fixed `one_hot_matrix`). Documents why Option 2 was chosen and the open target-format decision.

### `CLAUDE.md` — new file
Repository guidance for Claude Code: setup commands, common run commands, high-level architecture (complexity dial, precision model, GPT internals, MuonAdamW, inference engine, training script pattern), hardware-specific notes for this fork, and environment variables.

### `CHANGES.md` — new file
Running log of all modifications made since the fork from `karpathy/nanochat`. Updated after every change going forward.

### `nanochat/gpt.py` — `GPTEncoder` class
Added `GPTEncoder(GPT)` subclass that overrides `forward()` to stop before the `lm_head` projection and return raw hidden states of shape `(B, T, n_embd)`.  
- Full transformer trunk runs identically to `GPT` (smear → blocks → backout → final RMSNorm)
- Drops `lm_head`, softcap, and loss computation
- Supports both training and KV-cache inference
- Intended for representation learning / downstream task heads (wav2vec 2.0 / BERT-style encoder)

### `scripts/encoder_pretrain.py` — new file
Pretraining script for `GPTEncoder` using next-token cross-entropy as the pretraining signal.  
- Structurally mirrors `base_train.py` — same CLI flags, scaling laws, MuonAdamW optimizer, schedulers, checkpointing, wandb logging
- Separates the backbone (`GPTEncoder`) from the loss head (`stub_head`: thin linear → vocab)
- `compute_loss(hidden, targets)` is the designated swap-in point for custom task heads
- `stub_head` saved under `stub_head.*` prefix in checkpoints; encoder weights load with `strict=False` for easy head replacement
- `--core-metric-every` and `--sample-every` default to `-1` (disabled — no standalone `lm_head`)
- Checkpoint directory: `base_checkpoints/enc_d{depth}/`

---

## Committed — Training Infrastructure

### `fix(engine): use COMPUTE_DTYPE for KV cache instead of hardcoded bfloat16` (`8c83ae7`)
**File:** `nanochat/engine.py`  
KV cache tensors were hardcoded to `bfloat16`, causing a dtype mismatch crash on Titan V GPUs (which use `float32` as `COMPUTE_DTYPE`). Changed to use `COMPUTE_DTYPE` from `nanochat.common`.

### `fix(sft): prevent NaN loss from all-masked batches via safe-mean in V_gpt` (`39277ca`)
**Files:** `nanochat/V_gpt.py` (new), `scripts/V_chat_sft.py` (new), `runs/runeval.sh`  
When SFT packing produces a batch where all target positions are `-1` (conversations exceed `max_seq_len`), `cross_entropy` with `ignore_index=-1` and `reduction='mean'` computes `0/0 = NaN`, corrupting all model weights in one backward pass.  
- `nanochat/V_gpt.py`: patched `GPT` variant that replaces `reduction='mean'` with `sum / clamp(n_valid, 1)` so empty batches yield zero loss and zero gradients
- `scripts/V_chat_sft.py`: injects `V_gpt` via `sys.modules` before `checkpoint_manager` loads, so the fix is active without touching shared code

### `fix(chat_sft): disable optimizer state loading to prevent NaN divergence` (`309b0b2`)
**File:** `runs/runeval.sh`  
Added `--no-load-optimizer` flag to the `chat_sft` invocation in the eval script. Loading optimizer state from a checkpoint trained with a different precision setup was causing NaN divergence on Titan V.

---

## Committed — Titan V / Multi-GPU Compatibility Fixes

### `fix(runeval): disable chatcore eval in chat_sft to avoid BF16/FP32 KV cache crash on Titan V` (`e14242f`)
**File:** `runs/runeval.sh`  
Added `--no-chatcore-eval` to skip the chat-core evaluation path during SFT, which triggered the BF16/FP32 KV cache mismatch crash on Titan V hardware.

### `fix(runeval): skip sampling in base_eval to avoid BF16/FP32 KV cache dtype mismatch on Titan V` (`ac4d2a6`)
**File:** `runs/runeval.sh`  
Added `--sample-every=-1` to `base_eval` invocation to skip token sampling, which uses the KV cache path that crashes on Titan V due to dtype mismatch.

### `fix(rungpu2): disable sample-every to avoid BF16/FP32 KV cache dtype mismatch on Titan V` (`84a0925`)
**File:** `runs/rungpu2.sh`  
Set `--sample-every=-1` in the 2-GPU run script to skip sampling-based evaluation mid-training.

### `fix(rungpu2): halve device-batch-size to 16 for 2-GPU DDP` (`28ee56e`)
**File:** `runs/rungpu2.sh`  
Reduced `--device-batch-size` from 32 to 16 to prevent OOM when running DDP across 2 GPUs.

### `fix(rungpu2): separate torchrun and script args with --` (`d11e92c`)
**File:** `runs/rungpu2.sh`  
Added `--` separator between `torchrun` arguments and script arguments to fix argument parsing.

---

## Committed — New Run Scripts

### `chore: add runeval.sh for base_eval, chat_sft, and chat_cli sanity check` (`4777751`)
**File:** `runs/runeval.sh` (new)  
Sanity-check script that runs `base_eval`, `chat_sft`, and `chat_cli` sequentially to verify the full pipeline works end-to-end on the current hardware.

### `chore: add 2-GPU server run script and gitignore gsd/veda dirs` (`4f996b8`)
**Files:** `runs/rungpu2.sh` (new), `.gitignore`  
- `runs/rungpu2.sh`: `torchrun`-based script for 2-GPU distributed pretraining with hardware-specific overrides for Titan V
- `.gitignore`: added `gsd/` and `veda/` to exclude local planning and experiment directories from version control
