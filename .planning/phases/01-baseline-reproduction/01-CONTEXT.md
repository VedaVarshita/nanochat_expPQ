# Phase 1: Baseline Reproduction - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Execute the full `runs/runcpu.sh` pipeline on Apple Silicon MPS — tokenizer training → base pretraining (5000 steps, depth=6, head-dim=64) → base_eval → SFT sanity check — and document the resulting BPB curve and CORE score as the permanent reference for all future tokenizer and architecture experiments.

The exact model config, training flags, and pipeline order are fixed by `runs/runcpu.sh`. This phase clarifies HOW results are captured and documented, not what's trained.

</domain>

<decisions>
## Implementation Decisions

### Baseline Artifact Format
- **D-01:** Baseline results live in `veda/docs/BASELINE.md` — the fork-owner docs directory
- **D-02:** BPB curve format is a markdown table: `step → val_bpb` (50 rows, one per 100-step eval). Human-readable and git-diffable.
- **D-03:** BASELINE.md also contains: final CORE score, experiment protocol (seed, config flags, data version, tokenizer checkpoint hash), torch.compile outcome (success or eager fallback)

### Run Output Capture
- **D-04:** Training stdout is piped via `tee` to a log file: `python -m scripts.base_train ... | tee veda/logs/run1.txt`. Live terminal output preserved AND log file written.
- **D-05:** Log files live in `veda/logs/` — fork-owner directory, alongside docs
- **D-06:** Second reproducibility run uses `veda/logs/run2.txt` (same tee pattern)
- **D-07:** BPB values are extracted from the log file (grep `val_bpb` lines) and pasted into BASELINE.md table

### Claude's Discretion
- torch.compile strategy on MPS (proactively disable or let it attempt + document) — STATE.md flags this as highest-risk unknown; planner should handle based on `nanochat/optim.py` and available `--compile` flags
- Exact BASELINE.md structure beyond the required fields (headers, sections, formatting)
- Whether to create `veda/logs/` and `veda/docs/` directories in a setup task or inline

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reference Pipeline
- `runs/runcpu.sh` — Canonical MPS pipeline: exact flags for tok_train, base_train (depth=6, head-dim=64, seq-len=512, batch=32, 5000 iters), base_eval, chat_sft. This is the source of truth for all training commands.

### Requirements
- `.planning/REQUIREMENTS.md` — All Phase 1 requirements (ENV-01 through SFT-02). Every plan must map to at least one requirement ID.

### Project Context
- `.planning/PROJECT.md` — Hardware constraints (MPS only, float32), out-of-scope items, key decisions
- `.planning/STATE.md` — Known risks: torch.compile on MPS optimizer kernels (adamw_step_fused, muon_step_fused); step 0 Metal shader warmup hang (~30-90s — do not interrupt)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `nanochat/common.py` — `DummyWandb` class: makes all `wandb_run.log(...)` calls no-ops when `WANDB_RUN=dummy`. No wandb account needed.
- `nanochat/common.py` — `print0` for rank-0-only logging (safe for single-process MPS runs)
- `nanochat/checkpoint_manager.py` — Saves checkpoints to `~/.cache/nanochat/base_checkpoints/` by default (overridable via `NANOCHAT_BASE_DIR`)

### Established Patterns
- `PYTORCH_ALLOC_CONF=expandable_segments:True` is set at the top of training scripts — already in the codebase convention
- All training scripts use `argparse` with `--` prefix flags; `-1` means "disabled" for `--foo-every` interval args
- `runcpu.sh` uses `--core-metric-every=-1` (disables CORE score during training, runs it separately via `base_eval.py`)
- Meta device init pattern in `base_train.py`: `torch.device("meta")` → `to_empty()` → `init_weights()` — avoids CPU→MPS copy overhead

### Integration Points
- `scripts/tok_train.py` writes tokenizer checkpoint to `~/.cache/nanochat/` — this checkpoint must be available before `base_train.py` runs
- `scripts/base_train.py` → `scripts/base_eval.py` → `scripts/chat_sft.py` → `scripts/chat_cli.py` — strictly sequential pipeline
- `veda/` directory exists in the repo — `veda/docs/` and `veda/logs/` are new subdirectories to create

</code_context>

<specifics>
## Specific Ideas

- tee pattern for log capture: `python -m scripts.base_train [flags] | tee veda/logs/run1.txt`
- Extract BPB values: `grep "val_bpb" veda/logs/run1.txt` to get the 50 data points for the markdown table
- BASELINE.md lives at `veda/docs/BASELINE.md`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-baseline-reproduction*
*Context gathered: 2026-04-03*
