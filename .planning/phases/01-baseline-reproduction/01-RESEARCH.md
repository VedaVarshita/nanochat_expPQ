# Phase 1: Baseline Reproduction - Research

**Researched:** 2026-04-03
**Domain:** PyTorch MPS training pipeline — tokenizer training, base pretraining, evaluation, SFT, result capture
**Confidence:** HIGH

## Summary

Phase 1 executes the `runs/runcpu.sh` pipeline on Apple Silicon MPS and documents the resulting BPB curve and CORE score as the permanent reference. The pipeline is: dataset download → tokenizer training → base_train (5000 steps, depth=6, head-dim=64) → base_eval → chat_sft → chat_cli qualitative check. Every command, flag, and output path is already specified in `runcpu.sh` and the CONTEXT.md decisions. This phase is primarily about orchestrating execution and capturing results reliably, not about writing new training code.

The main technical unknowns are (1) `torch.compile` behavior on MPS for the fused optimizer kernels, and (2) Metal shader warmup at step 0. Research confirms `torch.compile` on MPS with PyTorch 2.10.0 succeeds for both `adamw_step_fused` and `muon_step_fused` — both kernels compiled and ran correctly on the local machine. The step-0 Metal shader warmup hang (30-90s) is a known timing issue, not a failure.

The result capture strategy (tee to `veda/logs/`, grep for val_bpb, paste into `veda/docs/BASELINE.md`) is fully specified in CONTEXT.md decisions D-01 through D-07. The planner's job is to sequence these actions into atomic, verifiable tasks.

**Primary recommendation:** Follow `runcpu.sh` exactly. Do not modify training flags. Capture stdout via `tee`. Document torch.compile outcome empirically at step 0.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Baseline results live in `veda/docs/BASELINE.md` — the fork-owner docs directory
- **D-02:** BPB curve format is a markdown table: `step → val_bpb` (50 rows, one per 100-step eval). Human-readable and git-diffable.
- **D-03:** BASELINE.md also contains: final CORE score, experiment protocol (seed, config flags, data version, tokenizer checkpoint hash), torch.compile outcome (success or eager fallback)
- **D-04:** Training stdout is piped via `tee` to a log file: `python -m scripts.base_train ... | tee veda/logs/run1.txt`. Live terminal output preserved AND log file written.
- **D-05:** Log files live in `veda/logs/` — fork-owner directory, alongside docs
- **D-06:** Second reproducibility run uses `veda/logs/run2.txt` (same tee pattern)
- **D-07:** BPB values are extracted from the log file (grep `val_bpb` lines) and pasted into BASELINE.md table

### Claude's Discretion

- torch.compile strategy on MPS (proactively disable or let it attempt + document) — STATE.md flags this as highest-risk unknown; planner should handle based on `nanochat/optim.py` and available `--compile` flags
- Exact BASELINE.md structure beyond the required fields (headers, sections, formatting)
- Whether to create `veda/logs/` and `veda/docs/` directories in a setup task or inline

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENV-01 | `uv sync --extra cpu` installs all dependencies on Apple Silicon without errors | pyproject.toml verified: `cpu` extra targets PyTorch CPU index; Python 3.13.2 and uv 0.9.9 confirmed available; .venv does not yet exist — install step needed |
| ENV-02 | `WANDB_RUN=dummy` suppresses wandb; training runs without a wandb account | `DummyWandb` class in `nanochat/common.py` confirmed; `base_train.py` line 100 uses it when `args.run == "dummy"` |
| ENV-03 | `PYTORCH_ALLOC_CONF=expandable_segments:True` is set and MPS memory is stable across 5000 steps | `base_train.py` line 15 sets this env var at module import — already in the code; no extra work needed |
| TOK-01 | Dataset downloaded via `python -m nanochat.dataset -n 8` (~2B chars) | Command is verbatim from `runcpu.sh`; downloads HuggingFace dataset shards to `~/.cache/nanochat/` |
| TOK-02 | Tokenizer trained via `scripts/tok_train` with `--max-chars=2000000000` | Command is verbatim from `runcpu.sh`; saves to `~/.cache/nanochat/`; `--vocab-size` defaults to 32768 |
| TOK-03 | `scripts/tok_eval` runs without error and reports vocab stats | Command is verbatim from `runcpu.sh`; produces vocab coverage stats |
| TOK-04 | Trained tokenizer checkpoint saved and reused for all subsequent runs | `checkpoint_manager.py` saves to `~/.cache/nanochat/`; tokenizer checkpoint hash must be recorded in BASELINE.md |
| BASE-01 | `scripts/base_train` runs to 5000 steps on MPS using runcpu.sh config | Full command documented in `runcpu.sh`; depth=6, head-dim=64, seq-len=512, batch=32 |
| BASE-02 | Training loss is stable (no NaN/Inf) throughout the run | float32 on MPS is numerically stable; no bfloat16 matmul issues; monitoring via grep on log |
| BASE-03 | val_bpb is logged every 100 steps and produces a complete 50-point loss curve | `--eval-every=100` in runcpu.sh; `base_train.py` line 427 prints `Step XXXXX \| Validation bpb: X.XXXXXX` |
| BASE-04 | `torch.compile` on optimizer kernels either succeeds or gracefully falls back; documented | Research confirms both `adamw_step_fused` and `muon_step_fused` compiled and ran on MPS with PyTorch 2.10.0 — HIGH confidence compile will succeed; however step-0 warmup (30-90s) is expected |
| BASE-05 | Final checkpoint saved to `~/.cache/nanochat/base_checkpoints/` | `base_train.py` saves at `last_step`; directory is `~/.cache/nanochat/base_checkpoints/d6/` (output_dirname = `d{depth}`) |
| EVAL-01 | `scripts/base_eval` completes with `--device-batch-size=1 --split-tokens=16384 --max-per-task=16` | Command verbatim from `runcpu.sh`; downloads `eval_bundle.zip` from S3 if not cached |
| EVAL-02 | DCLM CORE score is recorded (even if low) | `base_eval.py` writes CSV to `~/.cache/nanochat/base_eval/base_model_XXXXXX.csv`; also prints CORE metric |
| EVAL-03 | Baseline BPB curve and CORE score documented as reference | Goes into `veda/docs/BASELINE.md` per D-01 through D-03 |
| REPRO-01 | Second identical run produces val_bpb within ±0.005 of first run | Same seed (torch.manual_seed(42) in `compute_init`), same config, same data version; `runcpu.sh` flags are deterministic |
| REPRO-02 | Experiment protocol documented: seed, config flags, data version, tokenizer checkpoint | Captured from run log + git hash; goes into BASELINE.md |
| SFT-01 | `scripts/chat_sft` runs to completion on identity conversations data | `runcpu.sh` includes curl download of `identity_conversations.jsonl` + chat_sft command |
| SFT-02 | Model generates coherent text via `chat_cli` (qualitative check only) | `python -m scripts.chat_cli -p "What is the capital of France?"` — model should respond with Paris |
</phase_requirements>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 (pinned in pyproject.toml) | Training engine, MPS backend | Already pinned; MPS support built |
| uv | 0.9.9 (installed) | Python environment + package management | Used by runcpu.sh; `uv sync --extra cpu` is the install command |
| rustbpe | >=0.1.0 | BPE tokenizer training | Project tokenizer library |
| datasets | >=4.0.0 | Dataset download via `nanochat.dataset -n 8` | HuggingFace data pipeline |
| wandb | >=0.21.3 | Logging (disabled via DummyWandb) | In pyproject.toml; bypassed with WANDB_RUN=dummy |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=8.0.0 | Test runner | Running existing tests in `tests/` |
| tee (Unix) | system | Stdout capture to log file | All training runs in Phase 1 |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `tee` for log capture | Python logging redirect | tee is simpler, preserves real-time terminal output AND writes log file simultaneously |
| grep for BPB extraction | Python log parser | grep is sufficient for the 50-line extraction; no script needed |

**Installation:**
```bash
uv sync --extra cpu
source .venv/bin/activate
```

**Version verification:** PyTorch 2.10.0 is pinned in pyproject.toml under `[project.optional-dependencies] cpu`. Verified 2026-04-03 by reading pyproject.toml directly.

---

## Architecture Patterns

### Recommended Project Structure

```
veda/
├── docs/
│   └── BASELINE.md       # BPB curve table, CORE score, experiment protocol
└── logs/
    ├── run1.txt           # stdout from first training run (tee output)
    └── run2.txt           # stdout from reproducibility run
```

`~/.cache/nanochat/` (outside repo):
```
~/.cache/nanochat/
├── base_checkpoints/
│   └── d6/               # checkpoint files from base_train (depth=6)
├── base_eval/
│   └── base_model_XXXXXX.csv   # CORE score CSV from base_eval
├── eval_bundle/          # DCLM eval bundle (auto-downloaded)
└── identity_conversations.jsonl   # SFT data (auto-downloaded)
```

### Pattern 1: tee-based Stdout Capture

**What:** Pipe training command through `tee` to write a log file while displaying in terminal.
**When to use:** Every training run (run1, run2) and eval runs.
**Example:**
```bash
# Source: CONTEXT.md D-04/D-05
WANDB_RUN=dummy python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=dummy \
    2>&1 | tee veda/logs/run1.txt
```

Note: `2>&1` is required — `print0` goes to stdout but warnings/errors go to stderr. Without it, compiler warnings and error traces will not be captured in the log file.

### Pattern 2: BPB Extraction from Log

**What:** grep the log file for validation BPB lines after training completes.
**When to use:** After each training run to build the BASELINE.md table.
**Example:**
```bash
# Source: CONTEXT.md D-07
grep "Validation bpb" veda/logs/run1.txt
# Output format: "Step 00100 | Validation bpb: 1.234567"
# Produces 51 lines: step 0, 100, 200, ..., 5000
```

### Pattern 3: torch.compile Outcome Documentation

**What:** Detect whether torch.compile succeeds or falls back to eager mode; record in BASELINE.md.
**When to use:** At beginning of run, before step 0.
**Detection approach:** Look for compile-related messages in the log. A successful compile produces Metal shader compilation messages at step 0. A fallback produces a warning like "torch.compile failed" or "Falling back to eager". The 30-90s pause at step 0 is normal Metal shader warmup — it is NOT a hang.

### Pattern 4: Directory Setup (inline)

**What:** Create `veda/logs/` and `veda/docs/` before running training.
**When to use:** Task 1 (setup), before any training command.
**Example:**
```bash
mkdir -p veda/logs
# veda/docs/ already exists (contains LEADERBOARD_CPU.md)
```

### Anti-Patterns to Avoid

- **Redirecting stderr to /dev/null:** This hides torch.compile warnings and error traces. Always use `2>&1 | tee` not `| tee`.
- **Running base_eval before base_train checkpoint exists:** base_eval loads from `~/.cache/nanochat/base_checkpoints/d6/` — this directory only exists after base_train writes its final checkpoint at step 5000.
- **Retraining the tokenizer for run2:** TOK-04 requires the tokenizer is never retrained. Run2 reuses the existing tokenizer checkpoint.
- **Interrupting at step 0:** The Metal shader warmup looks like a hang but it is not. Wait it out (up to 90s).
- **Setting --core-metric-every during training:** runcpu.sh uses `--core-metric-every=-1`. Enabling it adds significant time per 2000-step cycle. CORE score is measured separately via base_eval.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Wandb suppression | Custom logging bypass | `DummyWandb` (already in `nanochat/common.py`) | Already implemented; `WANDB_RUN=dummy` flag activates it |
| BPB curve extraction | Python log parser script | `grep "Validation bpb" veda/logs/run1.txt` | 50 lines; grep is sufficient |
| Checkpoint management | Custom save/load | `nanochat/checkpoint_manager.py` | Already handles save/load with metadata JSON |
| MPS memory configuration | Manual torch allocator settings | `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"` | Already set in `base_train.py` line 15 |
| Environment setup | Manual pip installs | `uv sync --extra cpu` | Handles all dependencies including correct PyTorch CPU build |

**Key insight:** All infrastructure already exists in the codebase. Phase 1 is execution and documentation, not infrastructure building.

---

## Common Pitfalls

### Pitfall 1: Step 0 Apparent Hang

**What goes wrong:** Training appears frozen for 30-90 seconds at the start of step 0.
**Why it happens:** Apple's Metal GPU framework compiles shaders on first use. Both `torch.compile` (model forward pass) and the MPS backend itself trigger compilation at step 0. This is one-time; subsequent steps are fast.
**How to avoid:** Do not interrupt. Do not Ctrl-C. Wait the full 90 seconds.
**Warning signs:** No output for >30s at step 0. This is normal.

### Pitfall 2: torch.compile Fused Kernel Status

**What goes wrong:** Assuming compile succeeded without verification; documenting wrong outcome in BASELINE.md.
**Why it happens:** On some MPS configurations, torch.compile may emit warnings but still succeed (partial compilation). The Inductor warning about "Not enough SMs for max_autotune_gemm" appears on MPS — this is expected and benign.
**How to avoid:** After run1 completes, search the log for "compile" or "eager fallback" messages: `grep -i "compile\|eager\|fallback\|inductor" veda/logs/run1.txt`. Record what was found, not what was expected.
**Warning signs:** "W torch/_inductor" messages in log — these are warnings, not failures.

### Pitfall 3: Missing 2>&1 in tee Command

**What goes wrong:** Log file captures only stdout; stderr (compiler warnings, Python warnings, exceptions) missing from log.
**Why it happens:** Python writes logging output to stderr (the `ColoredFormatter` in `nanochat/common.py` uses `logging.StreamHandler()` which defaults to stderr).
**How to avoid:** Always use `2>&1 | tee veda/logs/run1.txt`, not `| tee veda/logs/run1.txt`.
**Warning signs:** Log file exists but is much shorter than terminal output.

### Pitfall 4: val_bpb Count Mismatch

**What goes wrong:** Expecting exactly 50 rows in the BPB table but getting 51 (step 0 included).
**Why it happens:** `base_train.py` line 421 evaluates `if last_step or step % args.eval_every == 0` — this triggers at step 0, 100, 200, ..., 5000. That is 51 evaluations (including step 0 and step 5000).
**How to avoid:** The BASELINE.md table should have 51 rows: step 0 through step 5000 inclusive. Decision D-02 says "50 rows" but the code produces 51. Include all rows in the table; document the count.
**Warning signs:** `grep "Validation bpb" veda/logs/run1.txt | wc -l` returns 51, not 50.

### Pitfall 5: base_eval Downloads eval_bundle on First Run

**What goes wrong:** base_eval appears to hang during evaluation — it is actually downloading `eval_bundle.zip` from S3.
**Why it happens:** `base_eval.py` line 115 checks `if not os.path.exists(eval_bundle_dir)` and downloads the bundle if missing. This is a one-time ~100MB download.
**How to avoid:** Expect a download at the start of the first base_eval run. Do not interrupt.
**Warning signs:** `base_eval` output shows "Downloading https://karpathy-public.s3..." — this is expected.

### Pitfall 6: Reproducibility Run Must Not Retrain Tokenizer

**What goes wrong:** Run2 starts from the `runcpu.sh` top and re-downloads data / retrains tokenizer.
**Why it happens:** runcpu.sh always runs the full pipeline from scratch.
**How to avoid:** Run2 executes only `base_train` and `base_eval` — the tokenizer step is skipped. The tokenizer checkpoint already exists from Run1.
**Warning signs:** `tok_train` appearing in run2 log output.

---

## Code Examples

Verified patterns from codebase inspection:

### BPB Log Line Format (from base_train.py line 427)

```python
# Source: scripts/base_train.py line 427
print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
```

Produces lines like:
```
Step 00100 | Validation bpb: 1.234567
Step 00200 | Validation bpb: 1.198432
...
Step 05000 | Validation bpb: 1.089123
```

Extraction command:
```bash
grep "Validation bpb" veda/logs/run1.txt
```

### Tokenizer Checkpoint Hash Capture

The tokenizer is saved to `~/.cache/nanochat/`. To record the checkpoint hash for BASELINE.md:
```bash
sha256sum ~/.cache/nanochat/tokenizer.model 2>/dev/null || \
  ls -la ~/.cache/nanochat/ | grep -i tok
```

### CORE Score Location (from base_eval.py lines 288-296)

```python
# Source: scripts/base_eval.py lines 288-298
output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
```

The CSV and the terminal output both contain the CORE metric:
```
CORE metric: 0.0234
Results written to: ~/.cache/nanochat/base_eval/base_model_005000.csv
```

### Checkpoint Output Directory (from base_train.py lines 155-156)

```python
# Source: scripts/base_train.py lines 155-156
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"  # e.g. d6
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
```

For depth=6 without `--model-tag`: `~/.cache/nanochat/base_checkpoints/d6/`

### Full runcpu.sh base_train Command (source of truth)

```bash
# Source: runs/runcpu.sh
WANDB_RUN=dummy python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Newton-Schulz orthogonalization in Muon | Polar Express Sign Method | Recent (optim.py references arxiv 2505.16932) | Better convergence for Muon; already in codebase |
| AdamW-only optimizer | Combined MuonAdamW (Muon for matrices, AdamW for embeddings) | Recent | Matrix parameters get orthogonal updates; already in codebase |

**Deprecated/outdated:**
- `chat_rl.py`: RL fine-tuning — explicitly out of scope per REQUIREMENTS.md
- `chat_web.py`: Production serving — explicitly out of scope

---

## Open Questions

1. **Exact val_bpb row count**
   - What we know: runcpu.sh sets `--eval-every=100` and `--num-iterations=5000`; `base_train.py` evaluates at `step % args.eval_every == 0` starting from step 0, plus at `last_step` (step 5000).
   - What's unclear: Whether step 0 and step 5000 are distinct eval points or one of them might be skipped.
   - Recommendation: After Run1, count `grep "Validation bpb" veda/logs/run1.txt | wc -l` and document the actual count. BASELINE.md should include all rows regardless.

2. **torch.compile outcome scope**
   - What we know: Local test confirmed both `adamw_step_fused` and `muon_step_fused` compile successfully on MPS with PyTorch 2.10.0. The Inductor SMs warning is benign.
   - What's unclear: Whether the full model compile (`torch.compile(model, dynamic=False)` at base_train.py line 246) succeeds without fallback for the specific model graph.
   - Recommendation: Treat compile as expected to succeed. At step 0, note the warmup duration and any compile-related log lines in BASELINE.md.

3. **SFT data download URL availability**
   - What we know: `runcpu.sh` downloads `identity_conversations.jsonl` from `karpathy-public.s3.us-west-2.amazonaws.com`.
   - What's unclear: Whether this S3 URL is still publicly accessible.
   - Recommendation: The curl command is in runcpu.sh. If it fails, flag for human resolution — no fallback exists for SFT-01.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All scripts | Yes | 3.13.2 | — |
| uv | ENV-01 | Yes | 0.9.9 | — |
| .venv (to be created) | All scripts | No (not yet) | — | Run `uv sync --extra cpu` |
| PyTorch (in venv) | BASE-01, EVAL-01, SFT-01 | No (not yet) | 2.10.0 pinned | Run `uv sync --extra cpu` |
| MPS backend | ENV-03, BASE-01 | Yes | PyTorch 2.10.0 `mps_available=True` | — |
| torch.compile on MPS | BASE-04 | Yes | Confirmed working | Eager fallback (auto) |
| Internet access | TOK-01, EVAL-01, SFT-01 | Assumed | — | Required (no offline fallback) |

**Missing dependencies with no fallback:**
- Internet access is required to download dataset (~2B chars from HuggingFace), eval_bundle.zip from S3, and identity_conversations.jsonl from S3.

**Missing dependencies with fallback:**
- `.venv` — must be created with `uv sync --extra cpu` (Wave 0 task).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0.0+ |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `python -m pytest tests/ -v -m "not slow"` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | uv sync completes | smoke | `uv sync --extra cpu && python -c "import torch"` | N/A (shell command) |
| ENV-02 | WANDB_RUN=dummy suppresses wandb | unit | `python -c "import os; os.environ['WANDB_RUN']='dummy'; from nanochat.common import DummyWandb; w=DummyWandb(); w.log({'x':1}); print('OK')"` | N/A (inline) |
| ENV-03 | PYTORCH_ALLOC_CONF set in base_train | unit | `python -c "import scripts.base_train" 2>&1 \| grep -q "PYTORCH_ALLOC_CONF" \|\| python -c "import scripts.base_train; import os; assert os.environ.get('PYTORCH_ALLOC_CONF')=='expandable_segments:True'"` | N/A (code inspection) |
| BASE-01 through BASE-05 | Training run completes | integration | Manual (5000-step training run; ~30 min; not automatable in <30s) | manual-only |
| EVAL-01, EVAL-02 | base_eval completes with CORE score | integration | Manual (downloads + evaluates; not automatable in <30s) | manual-only |
| REPRO-01 | Run2 val_bpb within ±0.005 of Run1 | integration | Manual (requires two full training runs) | manual-only |
| SFT-01, SFT-02 | chat_sft completes, chat_cli coherent | integration | Manual (qualitative check) | manual-only |

**Justification for manual-only:** All core requirements (BASE, EVAL, REPRO, SFT) require full training runs of 5000+ steps (~30 min each). These cannot be run in <30s and have no meaningful short-form proxy. They are validated by human inspection of log files and BASELINE.md content.

### Sampling Rate

- **Per task commit:** `python -m pytest tests/ -v -m "not slow"` (existing tests; confirms no regressions)
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** BASELINE.md exists with complete BPB table, CORE score, and experiment protocol; run2 val_bpb diff verified.

### Wave 0 Gaps

- [ ] `.venv` does not exist — must be created by `uv sync --extra cpu` before any training can run
- [ ] `veda/logs/` directory does not exist — must be created before tee commands

*(No new test files needed — existing infrastructure covers what can be automated; training runs are inherently manual)*

---

## Project Constraints (from CLAUDE.md)

No `CLAUDE.md` file exists in the repository root. Constraints come from `PROJECT.md`:

- **Hardware:** Apple Silicon MPS only — no CUDA, no multi-GPU DDP
- **dtype:** float32 only (bfloat16 not stable on MPS for this codebase)
- **Scale:** Small model configs only — full speedrun configs would take days on MPS
- **Baseline fidelity:** Experiments must be comparable to a fixed baseline run; randomness seeding matters
- **Out of scope:** CUDA training, RL fine-tuning, production serving, sliding window attention, DDP, bfloat16

---

## Sources

### Primary (HIGH confidence)

- `runs/runcpu.sh` — Canonical pipeline; exact training commands verified by reading file
- `scripts/base_train.py` — Training loop, log format, checkpoint paths, compile behavior verified by reading file
- `scripts/base_eval.py` — CORE evaluation, CSV output paths verified by reading file
- `nanochat/common.py` — DummyWandb, compute_init, print0 verified by reading file
- `nanochat/optim.py` — adamw_step_fused and muon_step_fused kernel implementations verified by reading file
- `pyproject.toml` — Dependency versions and uv extras verified by reading file
- `.planning/phases/01-baseline-reproduction/01-CONTEXT.md` — All locked decisions read directly

### Secondary (MEDIUM confidence)

- Local empirical test: `adamw_step_fused` and `muon_step_fused` compiled and ran successfully on MPS with PyTorch 2.10.0 — confirmed by running test code
- Local empirical test: `torch.compile` on MPS returns successfully for simple function — confirmed
- Inductor warning "Not enough SMs for max_autotune_gemm" on MPS — observed during testing, is benign

### Tertiary (LOW confidence)

- Step 0 Metal warmup duration (30-90s) — sourced from STATE.md; plausible but not directly measured in this session

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions read directly from pyproject.toml; uv/Python confirmed available
- Architecture patterns: HIGH — all patterns derived from reading actual source files
- Pitfalls: HIGH for code-based pitfalls (val_bpb count, tee pattern); MEDIUM for torch.compile (local test passed but full model compile not tested)
- Environment availability: HIGH — confirmed by running `python3 --version`, `uv --version`, and torch MPS test code

**Research date:** 2026-04-03
**Valid until:** 2026-05-03 (stable stack — PyTorch version pinned)
