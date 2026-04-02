# CONVENTIONS.md — Code Style & Patterns

## Language Style
- **Python 3.10+** — uses dataclasses, walrus operator, type hints where helpful
- No strict type annotations throughout — types used where they add clarity
- Snake_case everywhere (files, functions, variables)
- Module-level docstrings on all files explaining purpose and usage

## Code Organization

### Modules are self-contained
Each `nanochat/` module imports only from `nanochat/` or stdlib/third-party. No circular imports. Scripts import from `nanochat/` and `tasks/` but never the reverse.

### Flat over nested
No deep class hierarchies. `GPT` → `Block` → `CausalSelfAttention` / `MLP` is the deepest nesting. No abstract base classes, no factories, no dependency injection.

### Single global config
`COMPUTE_DTYPE` and `COMPUTE_DTYPE_REASON` are module-level globals in `nanochat/common.py`, imported everywhere. No config objects passed around.

## Model Code Patterns

### Custom Linear layer (dtype casting)
```python
# nanochat/gpt.py
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), ...)
```
All matmuls cast to `COMPUTE_DTYPE` this way — no `autocast` context manager.

### Meta device initialization
```python
# scripts/base_train.py
with torch.device("meta"):
    model = GPT(config)           # 1. shapes only, no data
model.to_empty(device=device)     # 2. allocate storage, garbage data
model.init_weights()              # 3. proper initialization
```
Used to avoid CPU → GPU copies on large models.

### `print0` for distributed logging
```python
from nanochat.common import print0
print0("Only rank 0 prints this")
```
All training scripts use `print0` instead of `print` to avoid duplicate output in DDP.

### DummyWandb for no-logging mode
```python
wandb_run = DummyWandb() if args.run == "dummy" else wandb.init(...)
wandb_run.log({"loss": loss})  # no-op when dummy
```

## CLI / Argument Patterns

All training scripts use `argparse` with `--` prefix flags:
```python
parser.add_argument("--depth", type=int, default=20)
parser.add_argument("--fp8", action="store_true")
parser.add_argument("--eval-every", type=int, default=250)  # -1 = disable
```
Convention: `-1` value means "disabled" for any `--foo-every` interval argument.

`torchrun` is used for multi-GPU: arguments after `--` are passed to the Python script:
```bash
torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=26 --fp8
```

## Error Handling
- Minimal defensive coding — internal code is trusted
- `assert` used freely for shape/invariant checks during development
- Gradient scaler enabled only for `float16` (`base_train.py` checks dtype explicitly)
- FP8: silently ignored on non-CUDA with a `print0` warning
- Flash Attention 3: silently falls back to SDPA with a printed warning block

## Checkpoint Backward Compatibility
New model parameters are patched in on load via `_patch_missing_keys()` and `_patch_missing_config_keys()` in `nanochat/checkpoint_manager.py`. This allows loading old checkpoints after architecture changes without breaking.

## Shell Scripts
- All `runs/*.sh` scripts are self-contained and runnable top-to-bottom
- `runs/speedrun.sh` is the canonical reference — always reflects current SOTA
- Scripts use `source .venv/bin/activate` and assume `uv` is installed
- `OMP_NUM_THREADS=1` set before `torchrun` to avoid thread contention

## Optimizer Parameter Groups
Matrix parameters → Muon; embeddings, unembeddings, scalars → AdamW. This split is explicit in `base_train.py` where param groups are assembled:
```python
# matrices: weight tensors with ndim >= 2 (excluding embeddings)
# scalars: resid_lambdas, x0_lambdas
# embeddings: wte
# unembedding: lm_head.weight
```

## Wandb Logging Convention
- `step` = optimization step number
- `total_training_time` = wall-clock seconds of training steps only (excludes eval)
- `total_training_flops` = cumulative FLOPs
- `val_bpb` = validation bits-per-byte (lower is better)
- `core_metric` = DCLM CORE score (higher is better, target > 0.256525)
