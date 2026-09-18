"""
Linear probe evaluation for GPTEncoder checkpoints.

Loads a frozen encoder checkpoint, trains a fresh Linear(n_embd -> vocab_size)
probe head on next-token prediction, and reports bits-per-byte — the same metric
as base_train.py and encoder_baseline.py.

This gives an apples-to-apples representation quality comparison between:
  enc_baseline_d6  — trained with next-token prediction (standard objective)
  enc_d6           — trained with BoxedLayer clustering objective

The encoder is fully frozen; only the probe head is trained. Val bpb before and
after probe training tells you how much useful next-token information is linearly
decodable from each encoder's representations.

Usage (single GPU, no torchrun):
    python -m scripts.encoder_probe \\
        --checkpoint enc_baseline_d6 --step 5000 --label "Baseline"

    python -m scripts.encoder_probe \\
        --checkpoint enc_d6 --step 5000 --label "BoxedLayer"
"""

import os
import math
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPTConfig, GPTEncoder, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import get_base_dir, autodetect_device_type, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_checkpoint

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Linear probe evaluation for GPTEncoder checkpoints")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="checkpoint dir name under base_checkpoints/ (e.g. enc_d6 or enc_baseline_d6)")
parser.add_argument("--step", type=int, required=True, help="checkpoint step to load")
parser.add_argument("--label", type=str, default=None, help="display label for printed results")
parser.add_argument("--probe-steps", type=int, default=1000,
                    help="steps to train the linear probe head (default 1000)")
parser.add_argument("--probe-lr", type=float, default=1e-3,
                    help="AdamW LR for probe head (default 1e-3)")
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--max-seq-len", type=int, default=512)
parser.add_argument("--eval-tokens", type=int, default=524288,
                    help="tokens to use for val bpb evaluation")
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
args = parser.parse_args()

label = args.label or args.checkpoint

# -----------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
if device_type == "cuda":
    device = torch.device("cuda:0")   # single GPU — probe doesn't need DDP
else:
    device = torch.device(device_type)
print(f"[{label}] Device: {device} | dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
pad_vocab = ((vocab_size + 63) // 64) * 64

# -----------------------------------------------------------------------------
# Load checkpoint and reconstruct frozen encoder
base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.checkpoint)
print(f"[{label}] Loading: {checkpoint_dir} @ step {args.step}")

model_data, _, meta_data = load_checkpoint(
    checkpoint_dir, args.step, device, load_optimizer=False, rank=0
)

config = GPTConfig(**meta_data["model_config"])
print(f"[{label}] n_layer={config.n_layer} n_embd={config.n_embd} n_head={config.n_head}")

encoder = GPTEncoder(config)
encoder.to_empty(device=device)
encoder.init_weights()
# strict=False: baseline checkpoints include stub_head.* keys; encoder_pretrain does not
missing, unexpected = encoder.load_state_dict(
    {k: v for k, v in model_data.items() if not k.startswith("stub_head.")},
    strict=False, assign=True,
)
if missing:
    print(f"[{label}] WARNING: missing keys in checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
del model_data

# Freeze all encoder parameters — only the probe head trains
for p in encoder.parameters():
    p.requires_grad_(False)
encoder.eval()
n_embd = config.n_embd

# -----------------------------------------------------------------------------
# Fresh linear probe head — same init as lm_head / stub_head
probe_head = Linear(n_embd, pad_vocab, bias=False).to(device)
nn.init.normal_(probe_head.weight, mean=0.0, std=0.001)
if COMPUTE_DTYPE != torch.float16:
    probe_head = probe_head.to(dtype=COMPUTE_DTYPE)

probe_optimizer = torch.optim.AdamW(
    probe_head.parameters(),
    lr=args.probe_lr,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)

# -----------------------------------------------------------------------------
def probe_logits(hidden):
    softcap = 15.0
    logits = probe_head(hidden)[..., :vocab_size].float()
    return softcap * torch.tanh(logits / softcap)

def evaluate_bpb():
    probe_head.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_bytes = torch.tensor(0.0, device=device)
    val_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device
    )
    eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
    with torch.no_grad():
        for i, (xv, yv) in enumerate(val_loader):
            if i >= eval_steps:
                break
            hidden = encoder(xv)
            logits = probe_logits(hidden)
            per_token_loss = F.cross_entropy(
                logits.view(-1, vocab_size), yv.view(-1),
                ignore_index=-1, reduction='none',
            )
            mask = yv.view(-1) != -1
            total_loss += per_token_loss[mask].sum()
            total_bytes += token_bytes[yv.view(-1)[mask]].sum()
    bpb = (total_loss / total_bytes / math.log(2)).item()
    probe_head.train()
    return bpb

# -----------------------------------------------------------------------------
# Evaluate before training — shows how much info is already linearly readable
bpb_before = evaluate_bpb()
print(f"[{label}] Val bpb  (probe untrained): {bpb_before:.6f}")

# Train probe
print(f"[{label}] Training probe for {args.probe_steps} steps (lr={args.probe_lr})...")
train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device
)
probe_head.train()
for step, (x, y) in enumerate(train_loader):
    if step >= args.probe_steps:
        break
    with torch.no_grad():
        hidden = encoder(x)
    loss = F.cross_entropy(
        probe_logits(hidden).view(-1, vocab_size), y.view(-1), ignore_index=-1
    )
    loss.backward()
    probe_optimizer.step()
    probe_optimizer.zero_grad(set_to_none=True)
    if (step + 1) % 100 == 0:
        print(f"[{label}]   step {step+1:04d}/{args.probe_steps} | train loss: {loss.item():.4f}")

bpb_after = evaluate_bpb()

# -----------------------------------------------------------------------------
print()
print(f"{'─'*56}")
print(f"  Checkpoint : {args.checkpoint} @ step {args.step}")
print(f"  Probe steps: {args.probe_steps}")
print(f"  Val bpb before probe : {bpb_before:.6f}")
print(f"  Val bpb after  probe : {bpb_after:.6f}  ← comparison number")
print(f"{'─'*56}")
