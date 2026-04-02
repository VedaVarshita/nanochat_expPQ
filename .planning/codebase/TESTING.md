# TESTING.md — Test Structure & Practices

## Framework
- **pytest** ≥ 8.0.0 (dev dependency)
- Config in `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  python_classes = ["Test*"]
  python_functions = ["test_*"]
  markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
  ```

## Test Files
```
tests/
├── test_engine.py              # Engine/KVCache unit tests
└── test_attention_fallback.py  # FA3 vs SDPA equivalence tests
```

## Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Skip slow tests
python -m pytest tests/ -v -m "not slow"

# Specific file
python -m pytest tests/test_engine.py -v
python -m pytest tests/test_attention_fallback.py -v -s
```

## `test_engine.py` — Engine Unit Tests
Tests the `Engine` and `KVCache` classes without a real model:

```python
@dataclass
class MockConfig:
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128

class MockModel:
    """Returns uniform logits — different samples should differ with temp > 0."""
```

Pattern: mock model returning uniform logits, verify KV cache correctness and sampling behavior.

## `test_attention_fallback.py` — Attention Equivalence Tests
Split into two classes due to hardware constraints:

**`TestFA3VsSDPA`** (requires Hopper GPU + bfloat16):
- Runs same inputs through both FA3 and SDPA implementations
- Asserts outputs are numerically close (`atol=1e-2, rtol=1e-2`)
- Tests training (no KV cache) and inference (with KV cache) paths

**`TestSDPAOnly`** (runs anywhere: CUDA, CPU, MPS):
- Tests SDPA fallback path in isolation
- Uses device-appropriate dtype (float32 on CPU/MPS)

Helper utilities:
```python
def set_impl(impl):   # Force 'fa3', 'sdpa', or None (auto)
def run_both_impls(fn):  # Run fn with both impls, return both outputs
def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):  # With helpful diffs
```

## Coverage Gaps
- No tests for training loop (`scripts/base_train.py`)
- No tests for optimizer (`nanochat/optim.py`)
- No tests for data loader (`nanochat/dataloader.py`)
- No tests for tokenizer (`nanochat/tokenizer.py`)
- No tests for checkpoint save/load (`nanochat/checkpoint_manager.py`)
- No integration tests for full pipeline
- No tests for SFT or RL stages

The test suite is intentionally minimal — nanochat is validated primarily through training runs and leaderboard results, not unit tests.

## Mocking Strategy
- Real models are not loaded in tests — `MockModel` / `MockConfig` replace them
- No database or network mocks needed (no such dependencies in the test paths)
- `flash_attention._override_impl` is set directly to test both attention paths

## CI
No CI configuration found (no `.github/workflows/`, no `.travis.yml`, etc.). Tests are run manually.
