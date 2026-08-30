"""
Pretrain GPTEncoder (hidden-state-only backbone, no lm_head projection).

The encoder produces (B, T, n_embd) hidden states. A thin stub head is attached
here purely to give a gradient signal during pretraining; you will swap it out
for your real task head before fine-tuning.

Run from root directory:
    python -m scripts.encoder_pretrain

Distributed:
    torchrun --nproc_per_node=8 -m scripts.encoder_pretrain

CPU/Macbook smoke-test:
    python -m scripts.encoder_pretrain --depth=4 --max-seq-len=512 \
        --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 \
        --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager
from scripts.BI import FFUFeaturizer

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, GPTEncoder, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.flash_attention import HAS_FA3
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments — identical to base_train.py so benchmark runs are comparable
parser = argparse.ArgumentParser(description="Pretrain GPTEncoder backbone")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"])
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-flops", type=float, default=-1.0)
parser.add_argument("--target-param-data-ratio", type=float, default=12)
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=-1)
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.008)
parser.add_argument("--weight-decay", type=float, default=0.28)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
parser.add_argument("--resume-from-step", type=int, default=-1)
# Evaluation
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=80*524288)
parser.add_argument("--core-metric-every", type=int, default=-1, help="disabled by default for encoder (no lm_head)")
parser.add_argument("--sample-every", type=int, default=-1, help="disabled by default for encoder (no lm_head)")
parser.add_argument("--save-every", type=int, default=-1)
# Output
parser.add_argument("--model-tag", type=str, default=None)
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

from nanochat.flash_attention import USE_FA3
if USE_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Recommend --window-pattern L")
    print0("!" * 80)

# -----------------------------------------------------------------------------
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the GPTEncoder
# GPTEncoder is a subclass of GPT with the lm_head projection removed.
# It outputs (B, T, n_embd) hidden states instead of logits.

def build_encoder_meta(depth):
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPTEncoder(config)
    return model_meta

# Build the encoder on meta device, move to target device, initialize weights
encoder = build_encoder_meta(args.depth)
model_config = encoder.config
model_config_kwargs = asdict(model_config)
print0(f"GPTEncoder config:\n{json.dumps(model_config_kwargs, indent=2)}")
encoder.to_empty(device=device)
encoder.init_weights()

# -----------------------------------------------------------------------------
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  USER CONFIGURATION: set NUM_LABELS to your target class count.          │
# │  Must be <= vocab_size. Stub default is 128.                             │
# └──────────────────────────────────────────────────────────────────────────┘
NUM_LABELS = 128  # TODO: CHANGE THIS to your actual number of label classes 

n_embd = model_config.n_embd

# one_hot_matrix: fixed (not learned) projection matrix.
# Shape: (NUM_LABELS, n_embd). Row i = target representation for class i.
# Fill this manually before training:
#   one_hot_matrix[class_id] = some_vector_of_shape_(n_embd,)
# Gradients flow through the encoder regardless of this matrix's values.
one_hot_matrix = torch.zeros(NUM_LABELS, n_embd, dtype=COMPUTE_DTYPE, device=device) #TODO - from Boxed layer

class BoxedLayer:
    """
    Maps encoder hidden states to integer class indices (B, T) in [0, NUM_LABELS).

    Called under torch.no_grad() using hidden.detach() — no backprop flows through here.
    Outputs become the 'targets' tensor for cross-entropy via one_hot_matrix.

    Active on epoch 0 and every 20th epoch (k-means style refresh).
    Between refresh epochs, previously cached outputs are replayed.
    """
    def __init__(self):
        self.relu_flag = False  # TODO: set as needed for FFUFeaturizer
        # d = per-sample input dim, k = number of output classes
        # reshape in __call__ yields (B*n_embd, T) patches → d=T, k=NUM_LABELS
        d = args.max_seq_len   # TODO: confirm d matches FFUFeaturizer's expected input dim
        k = NUM_LABELS
        self.feat = FFUFeaturizer(d, k, device=device, label=False, log_label=False,
                                  n=1, softmax=True, dtype=torch.float)

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, n_embd) — detached, no grad
        print(hidden.shape)
        x_unfold = hidden
        # Reshape: treat each (embedding-dim, batch) slice as a sample with T features
        # (B, T, n_embd) → transpose → (B, n_embd, T) → view → (B*n_embd, T)
        patch_vec = x_unfold.transpose(1, 2).contiguous().view(
            x_unfold.shape[0] * x_unfold.shape[2], x_unfold.shape[1]
        )
        return self.feat.update(patch_vec, Z=None, Y=None, relu_flag=self.relu_flag)

boxed_layer = BoxedLayer()

# Target cache — stores BoxedLayer outputs per micro-step for one full refresh epoch.
# Replayed for the 19 epochs between refreshes.
target_cache: list[torch.Tensor] = []   # (B, T) cpu tensors, one per micro-step
cache_built_for_epoch: int = -1         # dataloader epoch when cache was last built
micro_step_global: int = 0             # global micro-step counter for cache indexing
prev_dataloader_epoch: int = 0         # to detect epoch transitions

def compute_loss(hidden, targets, reduction='mean'):
    """
    Compute cross-entropy loss from encoder hidden states via one_hot_matrix.

    Args:
        hidden:  (B, T, n_embd) — encoder output
        targets: (B, T) integer class indices in [0, NUM_LABELS)
                 ─── OPEN DECISION ───────────────────────────────────────
                 Currently expects integer indices (F.cross_entropy default).
                 If you want to supply one-hot float vectors (B, T, NUM_LABELS),
                 change F.cross_entropy to soft cross-entropy or MSE here.
                 See docs/encoder_head_design.md for the trade-offs.
                 ──────────────────────────────────────────────────────────
        reduction: 'mean' or 'none'
    Returns:
        scalar loss (or per-token tensor if reduction='none')
    """
    logits = hidden @ one_hot_matrix.T          # (B, T, NUM_LABELS)
    logits = logits.float()
    softcap = 15.0
    logits = softcap * torch.tanh(logits / softcap)
    loss = F.cross_entropy(
        logits.view(-1, NUM_LABELS),
        targets.view(-1),
        ignore_index=-1,
        reduction=reduction,
    )
    return loss

# -----------------------------------------------------------------------------
# Resume from checkpoint if requested
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"enc_d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    encoder.load_state_dict(model_data, strict=False, assign=True)
    del model_data

# -----------------------------------------------------------------------------
# FP8 (same logic as base_train.py)
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

        def fp8_module_filter(mod, fqn):
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(encoder, config=fp8_config, module_filter_fn=fp8_module_filter)
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling)")

@contextmanager
def disable_fp8_ctx(modules):
    """Temporarily swap Float8Linear -> Linear for BF16 evaluation."""
    fp8_locations = []
    for root_mod in modules:
        for name, module in root_mod.named_modules():
            if 'Float8' in type(module).__name__:
                parent_name, attr_name = (name.rsplit('.', 1) if '.' in name else ('', name))
                parent = root_mod.get_submodule(parent_name) if parent_name else root_mod
                fp8_locations.append((parent, attr_name, module))
    if not fp8_locations:
        yield
        return
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(fp8_module.in_features, fp8_module.out_features, bias=fp8_module.bias is not None, device="meta", dtype=fp8_module.weight.dtype)
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)
    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile both modules
orig_encoder = encoder
encoder = torch.compile(encoder, dynamic=False)
# one_hot_matrix is a plain tensor — no compile needed

# -----------------------------------------------------------------------------
# Scaling laws — identical math to base_train.py
# Note: encoder has no lm_head, so num_scaling_params uses transformer_matrices only.
def build_ref_meta(depth):
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim, window_pattern=args.window_pattern)
    with torch.device("meta"):
        return GPT(config)  # full GPT to get comparable scaling param counts

param_counts = orig_encoder.num_scaling_params()
print0("GPTEncoder parameter counts:")
for k, v in param_counts.items():
    print0(f"  {k:24s}: {v:,}")
num_params = param_counts['total']
num_flops_per_token = orig_encoder.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

def get_scaling_params(m):
    c = m.num_scaling_params()
    return c['transformer_matrices'] + c['lm_head']  # GPT has lm_head; one_hot_matrix is not counted

d12_ref_full = build_ref_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref_full)
B_REF = 2**19

num_scaling_params = param_counts['transformer_matrices']  # encoder has no lm_head
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

batch_lr_scale = (total_batch_size / B_REF) ** 0.5 if total_batch_size != B_REF else 1.0
if batch_lr_scale != 1.0:
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}")

weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}")

# -----------------------------------------------------------------------------
# Optimizer: encoder backbone uses MuonAdamW (same as base_train.py)
# one_hot_matrix is fixed — no optimizer needed for it
optimizer = orig_encoder.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)
if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# DataLoaders
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Iteration count and schedulers
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
else:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")

total_tokens = total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    return 0.97

def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Validation: average cross-entropy loss using BoxedLayer targets
# (bpb is not applicable here — targets are cluster labels, not vocab tokens)
def evaluate_encoder_bpb(encoder_model, val_loader, eval_steps):
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0, device=device)
    encoder_model.eval()
    with torch.no_grad():
        for i, (xv, _) in enumerate(val_loader):
            if i >= eval_steps:
                break
            hidden = encoder_model(xv)
            targets = boxed_layer(hidden.detach())   # (B, T) class indices
            loss = compute_loss(hidden, targets, reduction='none')  # (B*T,)
            total_loss += loss.sum()
            total_tokens += loss.numel()
    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / total_tokens).item()
    encoder_model.train()
    return avg_loss

# -----------------------------------------------------------------------------
# Training loop
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0.0
    total_training_time = 0.0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Evaluate val bpb
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8_ctx([orig_encoder]):
            val_bpb = evaluate_encoder_bpb(orig_encoder, val_loader, eval_steps)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time, "val/bpb": val_bpb})

    # Save checkpoint: encoder state_dict only (one_hot_matrix is user-managed, not saved)
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        combined_state = orig_encoder.state_dict()
        save_checkpoint(
            checkpoint_dir,
            step,
            combined_state,
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {"min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss, "total_training_time": total_training_time},
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Single training step
    synchronize()
    t0 = time.time()

    # BoxedLayer target refresh schedule (k-means style):
    # Epoch 0 and every 20th epoch: rebuild target_cache from BoxedLayer outputs.
    # Other epochs: replay frozen cache.
    current_dataloader_epoch = dataloader_state_dict['epoch']
    if current_dataloader_epoch != prev_dataloader_epoch:
        prev_dataloader_epoch = current_dataloader_epoch
    is_refresh_epoch = (current_dataloader_epoch % 20 == 0)
    if is_refresh_epoch and current_dataloader_epoch != cache_built_for_epoch:
        # Start of a new refresh epoch — clear old cache and reset micro-step counter
        target_cache.clear()
        micro_step_global = 0
        cache_built_for_epoch = current_dataloader_epoch

    for micro_step in range(grad_accum_steps):
        # Encoder forward → hidden states
        hidden = encoder(x)                                  # (B, T, n_embd)

        # Determine targets via BoxedLayer (no backprop) or replay from cache
        if is_refresh_epoch:
            with torch.no_grad():
                targets_for_loss = boxed_layer(hidden.detach())  # (B, T) class indices
            target_cache.append(targets_for_loss.cpu())           # store for future epochs
        else:
            if not target_cache:
                raise RuntimeError(
                    "target_cache is empty. Ensure epoch 0 (a refresh epoch) "
                    "completes before non-refresh epochs are used."
                )
            cache_idx = micro_step_global % len(target_cache)
            targets_for_loss = target_cache[cache_idx].to(device)
        micro_step_global += 1

        loss = compute_loss(hidden, targets_for_loss)        # scalar; grads flow through hidden
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # Step optimizers
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    encoder.zero_grad(set_to_none=True)
    # -------------------------------------------------------------------------

    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    eta_str = ""
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        eta_str = f" | eta: {(num_iterations - step) * avg_time_per_step / 60:.1f}m"
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        wandb_run.log({"step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time, "train/loss": debiased_smooth_loss, "train/lrm": lrm, "train/dt": dt, "train/tok_per_sec": tok_per_sec, "train/mfu": mfu})

    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1
    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# -----------------------------------------------------------------------------
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

from nanochat.report import get_report
get_report().log(section="GPTEncoder pretraining", data=[
    user_config,
    {"Number of encoder parameters": num_params, "FLOPs per token": f"{num_flops_per_token:e}", "Training tokens": total_tokens, "DDP world size": ddp_world_size},
    {"Min val bpb": min_val_bpb if val_bpb is not None else None, "Final val bpb": val_bpb, "MFU %": f"{mfu:.2f}%"},
])

wandb_run.finish()
compute_cleanup()
