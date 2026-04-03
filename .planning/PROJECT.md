# nanochat-expPQ: Language Pretraining Experiments

## What This Is

A personal fork of the nanoChat open-source LLM training repo, used to run language pretraining experiments on Apple Silicon (MPS). The immediate goal is to reproduce the baseline training pipeline locally (following `runs/runcpu.sh`) as a verified foundation, then systematically test novel ideas around tokenization and model architecture.

## Core Value

A reproducible local baseline on MPS that can be used to fairly compare experimental changes against a known starting point.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Environment set up: `uv sync --extra cpu`, venv working on MPS
- [ ] Tokenizer trained on ~2B chars using `nanochat.dataset` + `scripts/tok_train`
- [ ] Small model (6-layer, head-dim=64) trains to completion following `runcpu.sh` config
- [ ] Training loss curve is stable and metrics (bpb) are logged
- [ ] Base eval (`scripts/base_eval`) runs and produces a DCLM CORE score
- [ ] SFT fine-tune completes on identity conversations
- [ ] Model can generate coherent text via `chat_cli` (sanity check)

### Out of Scope

- GPU (CUDA) training — this is a local MPS/CPU research fork
- Matching the H100 speedrun leaderboard score exactly — hardware gap makes this impossible; goal is a reproducible scaled-down baseline
- RL fine-tuning (chat_rl.py) — not needed for pretraining experiments
- Production serving (chat_web.py worker pools) — not relevant to research goals

## Context

- Upstream repo: nanoChat community/open-source version
- Reference run: `runs/runcpu.sh` — tuned for ~30 min on M3 Max, depth=6, head-dim=64, seq-len=512, 5000 iterations
- Dataset: downloaded via `python -m nanochat.dataset -n 8` (pulls ~2B chars)
- Hardware: Apple Silicon MPS, float32, PyTorch SDPA (no Flash Attention, no FP8)
- After baseline: experiments will focus on (1) tokenizer changes (vocab size, strategies) and (2) architecture tweaks (attention patterns, depth/width, novel ideas from autoresearch)

## Constraints

- **Hardware**: Apple Silicon MPS only — no CUDA, no multi-GPU DDP
- **dtype**: float32 (bfloat16 not stable on MPS for this codebase)
- **Scale**: Small model configs only — full speedrun configs would take days on MPS
- **Baseline fidelity**: Experiments must be comparable to a fixed baseline run; randomness seeding matters

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use `runcpu.sh` as baseline config | It's the repo's own MPS-tuned reference; eliminates guesswork about what "works" | — Pending |
| Small model (depth=6) not tiny | Tiny/debug runs don't produce meaningful loss signals for comparing experiments | — Pending |
| Reproduce before experimenting | Can't tell if a change helps without a known-good baseline to compare against | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-02 after initialization*
