---
phase: 01-baseline-reproduction
plan: 01
subsystem: infra
tags: [python, pytorch, uv, tokenizer, bpe, rustbpe, wandb, mps, huggingface]

# Dependency graph
requires: []
provides:
  - Working .venv with PyTorch 2.9.1 installed (MPS/CPU)
  - Dataset downloaded: 9 shards of base_data_climbmix (~2B chars) at ~/.cache/nanochat/base_data_climbmix/
  - BPE tokenizer checkpoint trained at ~/.cache/nanochat/tokenizer/ (tokenizer.pkl + token_bytes.pt)
  - veda/logs/ directory for training log capture
  - Verified: DummyWandb suppression with WANDB_RUN=dummy
  - Verified: PYTORCH_ALLOC_CONF=expandable_segments:True set in base_train.py at import
affects: [01-02, 01-03, 01-04, phase-02, phase-03]

# Tech tracking
tech-stack:
  added:
    - uv (package manager, creates .venv with CPython 3.10.16)
    - torch==2.9.1 (MPS/CPU, float32)
    - rustbpe==0.1.0 (fast BPE tokenizer training in Rust)
    - datasets==4.0.0, huggingface-hub==0.34.4 (dataset download)
    - wandb==0.21.3 (suppressed via DummyWandb)
  patterns:
    - WANDB_RUN=dummy activates DummyWandb in nanochat/common.py (no real wandb calls)
    - PYTORCH_ALLOC_CONF set at module import time in scripts/base_train.py
    - Tokenizer saved as .pkl+.pt pair in ~/.cache/nanochat/tokenizer/ (not a single .model file)

key-files:
  created:
    - veda/logs/.gitkeep
  modified: []

key-decisions:
  - "Tokenizer checkpoint format is ~/.cache/nanochat/tokenizer/tokenizer.pkl + token_bytes.pt (not tokenizer.model as plan assumed — rustbpe saves .pkl format)"
  - "PyTorch 2.9.1 installed (plan expected 2.10.0, uv resolved latest compatible version — fully compatible)"
  - "Tokenizer hashes recorded: tokenizer.pkl=387cfc08, token_bytes.pt=2163966b — must never be retrained (TOK-04)"

patterns-established:
  - "All scripts run with: source .venv/bin/activate && WANDB_RUN=dummy python -m scripts.*"
  - "Tokenizer is a shared artifact: reuse across all Phase 2 and Phase 3 runs"
  - "veda/logs/ is the designated directory for capturing training stdout/stderr logs"

requirements-completed: [ENV-01, ENV-02, ENV-03, TOK-01, TOK-02, TOK-03, TOK-04]

# Metrics
duration: 3min
completed: 2026-04-04
---

# Phase 01 Plan 01: Environment Setup and Tokenizer Pipeline Summary

**uv-managed venv with PyTorch 2.9.1, 9-shard HuggingFace dataset downloaded, 32768-vocab BPE tokenizer trained via rustbpe in 74s**

## Performance

- **Duration:** ~3 min (excluding dataset download ~2 min and tokenizer training ~74s)
- **Started:** 2026-04-04T04:39:56Z
- **Completed:** 2026-04-04T04:42:30Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Python environment set up: `.venv/` with PyTorch 2.9.1, all 65 packages installed via `uv sync --extra cpu`
- DummyWandb verified: `WANDB_RUN=dummy` correctly suppresses wandb (ENV-02)
- PYTORCH_ALLOC_CONF verified: `expandable_segments:True` set at import time in `scripts/base_train.py` (ENV-03)
- Dataset downloaded: 9 shards of base_data_climbmix to `~/.cache/nanochat/base_data_climbmix/` (TOK-01)
- Tokenizer trained: 32768-vocab BPE with 32503 merges from ~2B chars in 74s via rustbpe (TOK-02)
- Tokenizer evaluation passed: comparison vs GPT-2/GPT-4 vocab stats printed (TOK-03)
- Tokenizer hashes recorded for BASELINE.md (TOK-04)

## Task Commits

Each task was committed atomically:

1. **Task 1: Environment setup and tokenizer pipeline** - `50652f2` (chore)

**Plan metadata:** (to follow with docs commit)

## Files Created/Modified
- `veda/logs/.gitkeep` - Empty marker file to track logs directory in git

## Tokenizer Checkpoint Hashes (TOK-04)
These hashes are permanent — the tokenizer must NEVER be retrained:
- `~/.cache/nanochat/tokenizer/tokenizer.pkl` — SHA-256: `387cfc082b0bee45467774fd6f1310a922ad170886a58ccddcb468f275e06a6c`
- `~/.cache/nanochat/tokenizer/token_bytes.pt` — SHA-256: `2163966be0ff2e82687d92ad404418c4df0634be7f6463314f20b3ff8d0c547e`

## Decisions Made
- PyTorch 2.9.1 used (plan assumed 2.10.0; uv resolved latest compatible — no material difference for MPS/CPU float32)
- Tokenizer checkpoint is `.pkl` + `.pt` pair (not a single `.model` file as plan frontmatter suggested — this is how rustbpe saves checkpoints in this codebase version)

## Deviations from Plan

### Observation: Tokenizer path differs from plan assumption

**1. [Informational - Path] Tokenizer saves to ~/.cache/nanochat/tokenizer/ as .pkl/.pt pair**
- **Found during:** Task 1, Step 8 (checkpoint hash recording)
- **Issue:** Plan expected `~/.cache/nanochat/tokenizer.model` but rustbpe saves `~/.cache/nanochat/tokenizer/tokenizer.pkl` and `token_bytes.pt`
- **Fix:** No fix needed — this is correct codebase behavior. Hashes recorded for both files. Plan 04 BASELINE.md will document actual paths.
- **Files modified:** None
- **Verification:** `ls ~/.cache/nanochat/tokenizer/` shows both files, `python -m scripts.tok_eval` exits 0

---

**Total deviations:** 1 observation (no code changes)
**Impact on plan:** None — tokenizer works correctly, paths documented for future plans.

## Issues Encountered
None — all steps completed cleanly.

## User Setup Required
None - no external service configuration required. All dependencies installed via `uv sync --extra cpu`.

## Next Phase Readiness
- Environment fully ready for Plan 02 (base_train run)
- Tokenizer checkpoint is locked — do not retrain
- `veda/logs/` ready for training log capture
- Activate venv before any script: `source .venv/bin/activate`
- Set `WANDB_RUN=dummy` before any training script to suppress wandb

---
*Phase: 01-baseline-reproduction*
*Completed: 2026-04-04*

## Self-Check: PASSED

- FOUND: veda/logs/.gitkeep
- FOUND: 01-01-SUMMARY.md
- FOUND: .venv/
- FOUND: ~/.cache/nanochat/tokenizer/tokenizer.pkl
- FOUND commit 50652f2: chore(01-01): create veda/logs directory for training log capture
