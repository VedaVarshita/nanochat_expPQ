# STRUCTURE.md — Directory Layout & Organization

## Root Layout
```
nanochat_expPQ/
├── nanochat/           # Core library — shared across all pipeline stages
├── scripts/            # Entry points — one file per pipeline stage
├── tasks/              # Eval/SFT task definitions (ARC, MMLU, GSM8K, etc.)
├── runs/               # Shell scripts for full pipeline runs
├── tests/              # pytest test suite
├── dev/                # Development docs, tools, and assets
├── veda/               # Fork-owner additions (docs, experiments)
│   └── docs/           # e.g. LEADERBOARD_CPU.md
├── .planning/          # GSD project planning (created by /gsd:new-project)
│   └── codebase/       # This codebase map
├── .claude/            # Claude Code config + GSD plugin
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked dependency versions
└── README.md           # Main documentation
```

## `nanochat/` — Core Library
```
nanochat/
├── __init__.py             # empty
├── common.py               # COMPUTE_DTYPE, device init, logging, utilities
├── gpt.py                  # GPT nn.Module (transformer, attention, MLP)
├── optim.py                # MuonAdamW / DistMuonAdamW optimizer
├── dataloader.py           # Streaming tokenizing distributed data loader
├── dataset.py              # ClimbMix dataset download/read utilities
├── tokenizer.py            # BPE tokenizer wrapper (GPT-4 style)
├── checkpoint_manager.py   # Save/load checkpoints with backward compat patches
├── engine.py               # KVCache + Engine for efficient inference
├── flash_attention.py      # FA3 / SDPA unified interface
├── fp8.py                  # Minimal FP8 training (~150 lines, no torchao)
├── core_eval.py            # DCLM CORE score evaluation (22 tasks)
├── loss_eval.py            # Validation bits-per-byte evaluation
├── execution.py            # Python code execution tool for LLM
├── report.py               # Nanochat report writing utilities
└── ui.html                 # ChatGPT-style web UI (served by chat_web.py)
```

## `scripts/` — Entry Points
```
scripts/
├── base_train.py   # Pretraining (main script — speedrun target)
├── base_eval.py    # Base model eval: CORE score, bpb, samples
├── tok_train.py    # BPE tokenizer training
├── tok_eval.py     # Tokenizer compression rate evaluation
├── chat_sft.py     # Supervised fine-tuning
├── chat_rl.py      # Reinforcement learning fine-tuning
├── chat_eval.py    # Chat model evaluation tasks
├── chat_cli.py     # CLI chat interface
└── chat_web.py     # FastAPI web chat server
```

## `tasks/` — Eval & SFT Tasks
```
tasks/
├── common.py       # TaskMixture, TaskSequence base classes
├── arc.py          # ARC science questions (multiple choice)
├── mmlu.py         # MMLU broad topics (multiple choice)
├── gsm8k.py        # Grade school math (8K problems)
├── humaneval.py    # Python coding task
├── smoltalk.py     # SmolTalk HF dataset (SFT conversations)
├── spellingbee.py  # Letter spelling/counting task
└── customjson.py   # Load arbitrary JSONL conversations as task
```

## `runs/` — Shell Scripts
```
runs/
├── speedrun.sh     # 8×H100 GPT-2 speedrun (current leaderboard reference)
├── runcpu.sh       # CPU/MPS example (d6 model, ~30 min on M3 Max)
├── miniseries.sh   # Full miniseries of compute-optimal models
└── scaling_laws.sh # Scaling law experiments
```

## `dev/` — Development Assets
```
dev/
├── LEADERBOARD.md              # GPU leaderboard docs and run history
├── gen_synthetic_data.py       # Synthetic identity data generation
└── repackage_data_reference.py # Pretraining data shard generation reference
```

## `veda/docs/` — Fork Owner Additions
```
veda/docs/
└── LEADERBOARD_CPU.md          # CPU/MPS pretraining leaderboard breakdown
```

## `tests/`
```
tests/
├── test_engine.py              # Engine/KVCache unit tests (mock model)
└── test_attention_fallback.py  # FA3 vs SDPA equivalence tests
```

## Key File Paths for Development
| Task | File |
|------|------|
| Change model architecture | `nanochat/gpt.py` |
| Change optimizer | `nanochat/optim.py` |
| Change training loop | `scripts/base_train.py` |
| Change data loading | `nanochat/dataloader.py` |
| Change dataset source | `nanochat/dataset.py` |
| Change dtype/device handling | `nanochat/common.py` |
| Change inference | `nanochat/engine.py` |
| Add eval task | `tasks/<name>.py` |
| Change CORE eval | `nanochat/core_eval.py` |
| Reference speedrun | `runs/speedrun.sh` |
| Reference CPU run | `runs/runcpu.sh` |

## Naming Conventions
- Snake_case for all Python files and functions
- `base_*` prefix for pretraining stage scripts/functions
- `chat_*` prefix for SFT/RL/inference stage scripts/functions
- `tok_*` prefix for tokenizer stage scripts/functions
- Model checkpoints: `model_NNNNNN.pt`, `meta_NNNNNN.json` (step number zero-padded to 6 digits)
- Model tags: `d<depth>` e.g. `d24`, `d26` (used as checkpoint directory names)
