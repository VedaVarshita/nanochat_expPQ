#!/bin/bash

# Baseline reproduction run on 2x NVIDIA Titan V (GPU 1 and 2).
# Matches runcpu.sh exactly except: torchrun for 2-GPU DDP, GPU selection via CUDA_VISIBLE_DEVICES.
#
# Run as:
#   screen -S run1 bash runs/rungpu2.sh
#
# Detach: Ctrl+A D
# Reattach: screen -r run1
# Check progress: tail -f veda/logs/run1.txt

set -e

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1,2
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p veda/logs

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Dataset + tokenizer (skip if already done)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    python -m nanochat.dataset -n 8
    python -m scripts.tok_train --max-chars=2000000000
fi
python -m scripts.tok_eval

# Pretraining: 5000 steps on 2x Titan V via DDP
# total-batch-size=16384 split across 2 GPUs => device-batch-size=32 each (32*512*2 = 32768 tokens/step)
torchrun --nproc_per_node=2 -m scripts.base_train -- \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --num-iterations=5000 \
    --run=$WANDB_RUN \
    2>&1 | tee veda/logs/run1.txt
