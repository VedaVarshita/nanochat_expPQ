---
phase: 1
slug: baseline-reproduction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-04
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0.0+ |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `python -m pytest tests/ -v -m "not slow"` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds (automated tests only; training runs are manual) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -v -m "not slow"`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green AND BASELINE.md exists with complete BPB table, CORE score, and experiment protocol
- **Max feedback latency:** ~10 seconds (automated); training runs are manual-only

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| ENV-01 | 01 | 1 | ENV-01 | smoke | `uv sync --extra cpu && python -c "import torch"` | N/A (shell) | ⬜ pending |
| ENV-02 | 01 | 1 | ENV-02 | unit | `python -c "import os; os.environ['WANDB_RUN']='dummy'; from nanochat.common import DummyWandb; w=DummyWandb(); w.log({'x':1}); print('OK')"` | N/A (inline) | ⬜ pending |
| ENV-03 | 01 | 1 | ENV-03 | unit | Code inspection: `grep -r "PYTORCH_ALLOC_CONF" scripts/base_train.py` | N/A (code) | ⬜ pending |
| TOK-01–04 | 02 | 1 | TOK-01,02,03,04 | integration | Manual — tokenizer training run (~5 min) | manual-only | ⬜ pending |
| BASE-01–05 | 03 | 2 | BASE-01,02,03,04,05 | integration | Manual — `base_train` 5000 steps (~30 min) | manual-only | ⬜ pending |
| EVAL-01–03 | 04 | 2 | EVAL-01,02,03 | integration | Manual — `base_eval` + CORE score capture | manual-only | ⬜ pending |
| REPRO-01–02 | 05 | 3 | REPRO-01,02 | integration | Manual — Run2 + val_bpb diff check | manual-only | ⬜ pending |
| SFT-01–02 | 06 | 3 | SFT-01,02 | integration | Manual — `chat_sft` + `chat_cli` qualitative | manual-only | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `.venv` created by `uv sync --extra cpu` — no training can run without this
- [ ] `veda/logs/` directory exists before tee commands

*Existing pytest infrastructure covers all automatable requirements. Training runs (BASE, EVAL, REPRO, SFT) are inherently manual (~30 min each).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| base_train completes 5000 steps, no NaN/Inf | BASE-01–05 | ~30 min training run; no short proxy | Run `python -m scripts.base_train ...`; inspect log for NaN/Inf; verify 50 val_bpb rows |
| base_eval produces CORE score | EVAL-01–03 | Downloads + evaluates; not <30s | Run `python -m scripts.base_eval`; verify CORE score in output CSV |
| Run2 val_bpb within ±0.005 of Run1 | REPRO-01–02 | Requires two full training runs | Diff Run1 vs Run2 grep output; check max delta ≤0.005 |
| chat_sft completes, chat_cli coherent | SFT-01–02 | Qualitative coherence check | Run `python -m scripts.chat_sft`; run `python -m scripts.chat_cli`; verify coherent output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s (automated tasks)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
