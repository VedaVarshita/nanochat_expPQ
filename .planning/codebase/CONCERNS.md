# CONCERNS.md — Technical Debt, Issues & Concerns

## Known TODOs in Code

### `nanochat/gpt.py`
- **Q scaling hack**: `q = q * 1.2  # sharper attention, TODO think through better` (line ~101) — empirically tuned, not principled
- **RoPE over-compute**: `rotary_seq_len = config.sequence_len * 10  # TODO make nicer?` — wasteful allocation
- **RoPE base theta**: `# TODO: bump base theta more? e.g. 100K is more common` — may affect long-context quality
- **Chunked cross-entropy**: `# TODO experiment with chunked cross-entropy?` — memory optimization opportunity

### `nanochat/tokenizer.py`
- **Regex deviation**: `\p{N}{1,2}` instead of GPT-4's `\p{N}{1,3}` — suspected harmful but unvalidated (`TODO`)
- **Inefficient prepend**: `ids.insert(0, prepend_id)  # TODO: slightly inefficient` — O(n) list insert

### `nanochat/checkpoint_manager.py`
- **Re-init on load**: `model.init_weights()  # TODO: fix model re-init` — weights re-initialized then overwritten; wasteful and fragile

### `nanochat/core_eval.py`
- **SQuAD discrepancy**: Gets 31% vs reference 37% — cause unknown

### `nanochat/engine.py`
- **dtype hack**: `# NOTE: setting the dtype here and in this way is an ugly hack` (line ~180)

### `scripts/base_train.py`
- **Termination conditions**: `# TODO: possibly also add loss explosions etc.` — no automatic abort on NaN/explosion

### `scripts/chat_eval.py`
- **Tokenizer API**: `# TODO: remake the way this works` — prompt rendering approach needs rework

## Fragile Areas

### Non-determinism in training
- Run 4 (ClimbMix) showed CORE score spread of 0.016 across 7 identical runs (0.2512–0.2677)
- Default shard order is among the "unlucky" ones — acknowledged in LEADERBOARD.md
- No seed control for data shuffling across runs

### Sliding window + SDPA incompatibility
- `--window-pattern` other than `L` causes terrible performance without FA3
- No runtime error or guard — silently produces bad results on non-Hopper hardware
- Documented warning in `base_train.py` but easy to miss

### FP8 silently ignored on CPU
- `--fp8` flag is silently ignored with just a `print0` warning on non-CUDA
- No error, which could confuse users expecting FP8 behavior

### Backward-compat patching
- Old checkpoints missing `resid_lambdas`, `x0_lambdas`, `window_pattern` keys
- Patched at load time in `checkpoint_manager.py` — will accumulate over time as architecture evolves

## Performance Concerns

### CPU/MPS path is under-optimized
- `torch.compile` benefits are limited on CPU
- No profiling or optimization done for MPS path
- `runcpu.sh` explicitly notes "you will not get far on your Macbook"

### Data loading
- Tokenization happens at training time (not pre-tokenized)
- Multi-process data loading with `Pool` in `dataset.py` but tokenization is synchronous per batch

### Optimizer
- Muon requires `nesterov=True` momentum across all parameters — may not generalize perfectly to all depth settings (though empirically works)

## Security Concerns

### Web server abuse prevention
Limits exist in `chat_web.py` (500 messages, 8000 chars/message, 32000 chars total) but:
- No authentication on the web UI
- No rate limiting beyond message count
- Intended for personal/research use only — not production-hardened

### Data download
- Downloads parquet files directly over HTTPS from HuggingFace — no integrity verification (no checksums)

## Dependency Concerns
- `torch==2.9.1` is pinned to an exact version — will lag behind PyTorch releases
- `kernels` package for FA3 pulls from HuggingFace Hub at runtime — network dependency during training startup
- The `cpu` and `gpu` torch extras point to different PyPI indexes — switching between them requires full reinstall

## Missing Features / Future Work
- RL stage (`chat_rl.py`) does not support `float16` GradScaler (noted in README)
- No multi-node distributed training support (only single-node multi-GPU via DDP)
- No gradient checkpointing option for very large models
- No mixed-depth (encoder-decoder) architectures
