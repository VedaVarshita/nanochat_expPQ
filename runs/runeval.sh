#!/bin/bash

# Eval + SFT run on the server (follows rungpu2.sh pretraining).
# Runs base_eval, then chat_sft, then a quick chat_cli sanity check.
#
# Run as:
#   screen -S eval bash runs/runeval.sh
#
# Detach: Ctrl+A D
# Check progress: tail -f veda/logs/eval1.txt

set -e

export CUDA_VISIBLE_DEVICES=1,2
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p veda/logs

source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Step 1: DCLM CORE eval on base model (EVAL-01, EVAL-02, EVAL-03)
echo "=== base_eval ==="
CUDA_VISIBLE_DEVICES=1 python -m scripts.base_eval \
    --device-batch-size=1 \
    --split-tokens=16384 \
    --max-per-task=16 \
    --eval bpb,core \
    2>&1 | tee veda/logs/eval1.txt

# Step 2: Download SFT data and fine-tune (SFT-01, SFT-02)
echo "=== chat_sft ==="
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --nproc_per_node=2 -m scripts.chat_sft -- \
    --max-seq-len=512 \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --chatcore-every=-1 \
    --load-optimizer=0 \
    --num-iterations=1500 \
    --run=$WANDB_RUN \
    2>&1 | tee veda/logs/sft1.txt

# Step 3: Sanity check — model should answer basic questions
echo "=== chat_cli sanity check ==="
CUDA_VISIBLE_DEVICES=1 python -m scripts.chat_cli \
    -p "What is the capital of France?" \
    2>&1 | tee -a veda/logs/sft1.txt
