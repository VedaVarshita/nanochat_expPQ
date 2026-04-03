# Requirements: nanochat-expPQ

**Defined:** 2026-04-02
**Core Value:** A reproducible local MPS baseline to fairly compare tokenizer and architecture experiments

## v1 Requirements

### Environment

- [ ] **ENV-01**: `uv sync --extra cpu` installs all dependencies on Apple Silicon without errors
- [ ] **ENV-02**: `WANDB_RUN=dummy` suppresses wandb and training runs without a wandb account
- [ ] **ENV-03**: `PYTORCH_ALLOC_CONF=expandable_segments:True` is set and MPS memory is stable across a full 5000-step run

### Tokenizer

- [ ] **TOK-01**: Dataset downloaded via `python -m nanochat.dataset -n 8` (~2B chars)
- [ ] **TOK-02**: Tokenizer trained via `scripts/tok_train` with `--max-chars=2000000000`
- [ ] **TOK-03**: `scripts/tok_eval` runs without error and reports vocab stats
- [ ] **TOK-04**: Trained tokenizer checkpoint is saved and reused for all subsequent runs (never retrain mid-experiment)

### Baseline Pretraining

- [ ] **BASE-01**: `scripts/base_train` runs to 5000 steps on MPS using the runcpu.sh config (depth=6, head-dim=64, seq-len=512, batch=32)
- [ ] **BASE-02**: Training loss is stable (no NaN/Inf) throughout the run
- [ ] **BASE-03**: val_bpb is logged every 100 steps and produces a complete loss curve
- [ ] **BASE-04**: `torch.compile` on optimizer kernels either succeeds or gracefully falls back to eager mode (documented)
- [ ] **BASE-05**: Final checkpoint saved to `~/.cache/nanochat/base_checkpoints/`

### Evaluation

- [ ] **EVAL-01**: `scripts/base_eval` completes with `--device-batch-size=1 --split-tokens=16384 --max-per-task=16`
- [ ] **EVAL-02**: DCLM CORE score is recorded (even if low — this is expected at small scale)
- [ ] **EVAL-03**: Baseline BPB curve and CORE score documented as reference for all future experiments

### Reproducibility

- [ ] **REPRO-01**: Second identical run (same seed, same config) produces val_bpb within ±0.005 of first run
- [ ] **REPRO-02**: Experiment protocol documented: seed, config flags, data version, tokenizer checkpoint

### SFT Sanity Check

- [ ] **SFT-01**: `scripts/chat_sft` runs to completion on identity conversations data
- [ ] **SFT-02**: Model generates coherent text via `chat_cli` (qualitative check only)

## v2 Requirements

### Tokenizer Experiments

- **TOKEXP-01**: Vocab size comparison — 4K vs 8K vs 16K vs 32K (controlled for bytes seen)
- **TOKEXP-02**: value_embeds on vs off × vocab size — isolate embedding parameter dominance effect
- **TOKEXP-03**: Split pattern `{1,1}` vs `{1,2}` at 8K vocab

### Architecture Experiments

- **ARCHEXP-01**: RoPE theta sweep — {500, 2000, 10000} vs baseline 100000
- **ARCHEXP-02**: Value embeddings ablation — freeze ve_gate at zero
- **ARCHEXP-03**: x0_lambdas ablation — freeze at zero
- **ARCHEXP-04**: relu² vs SwiGLU (parameter-equalized)
- **ARCHEXP-05**: resid_lambdas: flat 1.0 vs decaying schedule
- **ARCHEXP-06**: head_dim sweep — 32 / 64 / 128
- **ARCHEXP-07**: Depth/width: depth=4 vs 6 vs 8 at fixed dim=384

## Out of Scope

| Feature | Reason |
|---------|--------|
| CUDA / H100 training | This is a local MPS research fork |
| Matching speedrun leaderboard score exactly | Hardware gap makes this impossible; goal is reproducible scaled-down baseline |
| RL fine-tuning (chat_rl.py) | Not needed for pretraining research |
| Production serving (chat_web.py worker pools) | Not relevant to research goals |
| Sliding window attention patterns | Requires FA3 or new mask impl; MPS uses full causal only |
| DDP / multi-process training | MPS does not support distributed backends |
| bfloat16 on MPS | Numerically unreliable for matmul; float32 only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 – ENV-03 | Phase 1 | Pending |
| TOK-01 – TOK-04 | Phase 1 | Pending |
| BASE-01 – BASE-05 | Phase 1 | Pending |
| EVAL-01 – EVAL-03 | Phase 1 | Pending |
| REPRO-01 – REPRO-02 | Phase 1 | Pending |
| SFT-01 – SFT-02 | Phase 1 | Pending |
| TOKEXP-01 – TOKEXP-03 | Phase 2 | Pending |
| ARCHEXP-01 – ARCHEXP-07 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after initial definition*
