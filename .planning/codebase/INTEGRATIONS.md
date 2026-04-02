# INTEGRATIONS.md — External Services & APIs

## Data Sources

### ClimbMix-400B (Primary Pretraining Dataset)
- **URL**: `https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main`
- **Format**: Parquet shards (`shard_NNNNN.parquet`), up to shard 06542
- **Download**: `python -m nanochat.dataset -n <N>` fetches N shards on demand
- **Location**: `~/.cache/nanochat/base_data_climbmix/`
- **Legacy**: FineWeb-EDU-100B at `~/.cache/nanochat/base_data/` (deprecated Mar 4, 2026)

### SmolTalk (SFT Dataset)
- **Source**: HuggingFace `datasets` library, loaded dynamically in `tasks/smoltalk.py`
- **Used in**: `scripts/chat_sft.py`

### Identity Conversations (optional SFT)
- **URL**: `https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl`
- **Used in**: `runs/runcpu.sh` example

## Experiment Tracking

### Weights & Biases (wandb)
- **Project**: `nanochat`
- **Disable**: `--run dummy` or `WANDB_RUN=dummy`
- **Key metrics**: `val_bpb`, `core_metric`, `total_training_time`, `total_training_flops`, `train/mfu`, `train/tok_per_sec`
- All training scripts (`base_train.py`, `chat_sft.py`, `chat_rl.py`) log to wandb

## ML Kernels

### Flash Attention 3
- **Loader**: `kernels` package (`from kernels import get_kernel`)
- **Kernel**: `varunneal/flash-attention-3` pulled from HuggingFace Hub
- **Requirement**: Hopper GPU (sm90), bfloat16 dtype
- **Fallback**: PyTorch SDPA (automatic, no config needed)
- **Env**: `HF_HUB_DISABLE_PROGRESS_BARS=1` set when loading

## Evaluation Benchmarks (CORE metric)

The DCLM CORE score is an ensemble of ~22 tasks loaded via HuggingFace `datasets`:
- ARC (easy/challenge), MMLU, HellaSwag, WinoGrande, PIQA, SIQA, OpenBookQA, BoolQ, SQuAD, and others
- Auto-loaded when `--core-metric-every N > 0` during training
- See `nanochat/core_eval.py`

## No Auth / Database / Infrastructure
- No OAuth, no auth providers
- No databases — all state is file-based (`.pt` checkpoints, `.json` metadata, `.parquet` data)
- No message queues or cloud infrastructure
- Web UI (`scripts/chat_web.py`) is a fully self-contained FastAPI server
