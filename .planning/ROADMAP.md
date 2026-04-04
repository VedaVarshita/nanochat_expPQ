# Roadmap: nanochat-expPQ

## Overview

This project forks nanoChat to run reproducible LLM pretraining experiments on Apple Silicon MPS. Phase 1 establishes a stable, reproducible baseline using the runcpu.sh config. Phase 2 sweeps tokenizer vocabulary size and split patterns against that baseline. Phase 3 ablates architecture choices — RoPE theta, value embeddings, activation functions, scalar lambdas, head dim, and depth/width — to identify which design decisions matter at this scale.

## Phases

- [ ] **Phase 1: Baseline Reproduction** - Run runcpu.sh end-to-end on MPS, document reference BPB curve and CORE score
- [ ] **Phase 2: Tokenizer Experiments** - Sweep vocab sizes (4K/8K/16K/32K), value_embeds interaction, and split pattern
- [ ] **Phase 3: Architecture Experiments** - Ablate RoPE theta, VE, x0_lambdas, activations, scalar schedules, head dim, and depth/width

## Phase Details

### Phase 1: Baseline Reproduction
**Goal**: A stable, reproducible pretraining baseline exists on MPS — reference BPB curve and CORE score documented for all future comparisons
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, TOK-01, TOK-02, TOK-03, TOK-04, BASE-01, BASE-02, BASE-03, BASE-04, BASE-05, EVAL-01, EVAL-02, EVAL-03, REPRO-01, REPRO-02, SFT-01, SFT-02
**Success Criteria** (what must be TRUE):
  1. `uv sync --extra cpu` completes without errors and training runs without a wandb account
  2. `base_train` runs to 5000 steps on MPS with no NaN/Inf loss and `PYTORCH_ALLOC_CONF=expandable_segments:True` set; torch.compile outcome (success or eager fallback) is documented
  3. val_bpb is logged every 100 steps and a complete 50-point loss curve exists in the run log
  4. A second identical run (same seed, same config) produces val_bpb within ±0.005 of the first run; seed, config flags, data version, and tokenizer checkpoint are recorded
  5. `chat_sft` completes and `chat_cli` generates coherent text (qualitative check passes)
**Plans:** 4 plans

Plans:
- [x] 01-01-PLAN.md — Environment setup + tokenizer training
- [ ] 01-02-PLAN.md — Base pretraining run 1 (5000 steps)
- [ ] 01-03-PLAN.md — Evaluation (CORE score) + SFT sanity check
- [ ] 01-04-PLAN.md — Reproducibility run + BASELINE.md documentation

**UI hint**: no

### Phase 2: Tokenizer Experiments
**Goal**: The optimal vocabulary size for this model scale is identified, and the value_embeds and split pattern interactions are measured against the Phase 1 baseline BPB curve
**Depends on**: Phase 1
**Requirements**: TOKEXP-01, TOKEXP-02, TOKEXP-03
**Success Criteria** (what must be TRUE):
  1. Four tokenizers trained (4K, 8K, 16K, 32K vocab) from the same 2B-char corpus with the same seed; BPB curves for all four exist in a single comparison table
  2. value_embeds on vs off is measured at at least two vocab sizes; the table records BPB delta to isolate embedding parameter dominance
  3. Split pattern `{1,1}` vs `{1,2}` at 8K vocab is measured and the BPB difference is recorded
  4. An optimal vocab size recommendation is made for this model scale (or a clear "too noisy to call" conclusion is documented)
**Plans**: TBD

### Phase 3: Architecture Experiments
**Goal**: An ablation table exists showing BPB delta vs Phase 1 baseline for each architecture variant, establishing which design decisions matter at this scale
**Depends on**: Phase 1
**Requirements**: ARCHEXP-01, ARCHEXP-02, ARCHEXP-03, ARCHEXP-04, ARCHEXP-05, ARCHEXP-06, ARCHEXP-07
**Success Criteria** (what must be TRUE):
  1. RoPE theta sweep ({500, 2000, 10000} vs baseline 100000) is run and BPB delta is recorded for each value
  2. Value embeddings ablation (ve_gate frozen at zero) is run; x0_lambdas behavior with and without VE is noted
  3. relu² vs SwiGLU (parameter-equalized) is run and BPB delta is recorded
  4. All scalar ablations (x0_lambdas, resid_lambdas) are run one at a time and BPB deltas are recorded
  5. head_dim sweep (32/64/128) and depth/width sweep (depth=4/6/8 at dim=384) are run and results are in the ablation table
  6. Final ablation table exists with one row per variant, BPB delta column, and a "meaningful signal?" column (threshold: >0.01 bpb single run, >0.005 bpb with two replications)
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Baseline Reproduction | 0/4 | Planning complete | - |
| 2. Tokenizer Experiments | 0/? | Not started | - |
| 3. Architecture Experiments | 0/? | Not started | - |
