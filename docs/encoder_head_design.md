# Encoder Head Design Decision

Design record for the output projection layer in `scripts/encoder_pretrain.py`.

---

## Decisions Made

A log of every design choice made during this session, in order.

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Output projection head type | **Option 2 — fixed `one_hot_matrix`** over Option 1 (learned `stub_head`) | User controls class geometry; `NUM_LABELS` is independent of `vocab_size`; no optimizer overhead |
| 2 | `one_hot_matrix` dimensions | **`(NUM_LABELS, n_embd)`** — user-defined label count, not vocab-sized | Label space is a separate domain (not token vocabulary) |
| 3 | Target format (integer indices vs. one-hot float vectors) | **Open / undecided** | To be resolved when the task head and dataset are finalized — see Open Decision section below |
| 4 | BoxedLayer architecture | **User's custom NN** (nearest-neighbor style) | Implementation is domain-specific; scaffold provided as a stub |
| 5 | BoxedLayer output | **Hard class indices `(B, T)` int** → become the `targets` tensor for cross-entropy | Matches `F.cross_entropy` integer-index API |
| 6 | Backprop through BoxedLayer | **No** — called with `hidden.detach()` under `torch.no_grad()` | BoxedLayer is an assignment oracle, not a learned layer |
| 7 | Target refresh schedule | **Option B — cache for 20 epochs (k-means style)**: epoch 0 + every 20th epoch rebuilds `target_cache`; 19 epochs in between replay the frozen cache | Cheaper per training step than re-running BoxedLayer every batch; mirrors k-means EM |

---

## The problem

`GPTEncoder` outputs hidden states of shape `(B, T, n_embd)`. To compute a cross-entropy training signal, these hidden states must be projected to a logit vector of size `(B, T, NUM_LABELS)`. Two approaches were considered.

---

## Option 1 — Learned linear head (`stub_head`)

A standard `nn.Linear(n_embd, NUM_LABELS)` layer with learned weights. During training, the head parameters are updated by a separate `AdamW` optimizer alongside the encoder's `MuonAdamW`.

**Pros:**
- The head adapts to the encoder's output distribution during pretraining.
- Closest to standard LM pretraining (equivalent to `base_train.py` with a different output size).
- Cross-entropy loss has a valid gradient signal immediately, even before the user intervenes.

**Cons:**
- The user must swap out the head before fine-tuning — the learned weights carry the pretraining projection, not the task-specific representation.
- Adds optimizer state and parameter count.
- The head may "absorb" alignment work that should be done by the encoder itself.

---

## Option 2 — Fixed user-filled matrix (`one_hot_matrix`) ← **CHOSEN**

A plain `torch.Tensor` of shape `(NUM_LABELS, n_embd)`, initialized to zeros, with `requires_grad=False`. The user fills in the rows manually before training. The projection is:

```python
logits = hidden @ one_hot_matrix.T   # (B, T, NUM_LABELS)
```

No optimizer touches this matrix. Gradients flow through `hidden` (and therefore through the encoder), but not into `one_hot_matrix`.

**Why chosen:**
- `NUM_LABELS` is user-defined and independent of `vocab_size` — the label space is a separate domain (e.g., phoneme classes, speaker IDs, semantic categories).
- The user controls exactly what each class representation looks like — rows can encode prior knowledge about class geometry.
- No optimizer overhead; no second `param_group` to manage.
- Pretraining trains the encoder to produce hidden states that are close (in dot-product space) to the user-specified class vectors.

**Trade-off:**
- With an all-zeros matrix (the default stub), all logits are zero → uniform distribution → loss is `log(NUM_LABELS)` nats with near-zero gradients. The user **must** fill `one_hot_matrix` with meaningful vectors before training begins for gradients to be informative.

---

## Open decision: target format

Currently `compute_loss` expects `targets` as integer class indices `(B, T)` and uses `F.cross_entropy` with `ignore_index=-1`. This is the standard hard-label format.

**If you want soft targets** (e.g., probability distributions over classes, or continuous regression targets):

- Replace `F.cross_entropy(logits.view(-1, NUM_LABELS), targets.view(-1), ...)` with a soft cross-entropy:
  ```python
  log_probs = F.log_softmax(logits, dim=-1)
  loss = -(targets_float * log_probs).sum(dim=-1).mean()
  ```
- Or use MSE if targets are real-valued embeddings:
  ```python
  loss = F.mse_loss(logits, targets_float)
  ```

See `compute_loss` in [scripts/encoder_pretrain.py](../scripts/encoder_pretrain.py) — the decision point is marked with an `OPEN DECISION` comment.

---

## BoxedLayer — target assignment between hidden states and one_hot_matrix

A `BoxedLayer` sits between the encoder hidden states and `one_hot_matrix`. Its role is to assign each hidden vector to a class index, producing the `targets` tensor that `compute_loss` uses.

### Interface

```python
class BoxedLayer:
    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, n_embd) — detached, no grad
        # returns: (B, T) integer tensor, values in [0, NUM_LABELS)
        ...
```

The user implements `__call__` with their own nearest-neighbor / lookup logic. The layer is called under `torch.no_grad()` with `hidden.detach()` — **no gradient flows through it**.

### Data flow

```
encoder(x)  →  hidden (B, T, n_embd)   ←── grad flows here
                    │
              hidden.detach()
                    │
               boxed_layer(...)          ←── no grad (disconnected branch)
                    │
             targets (B, T) int          ←── discrete integers, not in compute graph
                    │
        compute_loss(hidden, targets)   ←── hidden still has grad
                    │
              one_hot_matrix.T → logits → cross_entropy → loss.backward()
```

### Gradient path clarification

Targets are **integers** — discrete values with no gradient. They are not part of the computation graph. `F.cross_entropy` uses them only to select which class to penalize; the gradient flows exclusively through the prediction side.

The actual backprop path is:

```
loss.backward()
      │
  cross_entropy   ← targets just pick which class to penalize (discrete, no grad)
      │
  logits (B, T, NUM_LABELS)
      │
  hidden @ one_hot_matrix.T   ← one_hot_matrix is fixed, no grad
      │
  hidden (B, T, n_embd)       ← gradient arrives here
      │
  encoder(x)                  ← gradient flows all the way back through transformer
```

**BoxedLayer is never in the gradient path.** It runs to produce the integer labels, then steps aside entirely. The reason `hidden.detach()` is passed into BoxedLayer is not to protect `hidden` from receiving gradients (it still does, via the loss path) — it is to prevent BoxedLayer itself from being part of the computation graph so PyTorch does not attempt to backprop through it.

BoxedLayer influences *what the encoder is trained toward* (by choosing which class index is the target each step) but is **never differentiated**.

### Refresh schedule (k-means style)

- **Epoch 0 and every 20th epoch** (`current_epoch % 20 == 0`): BoxedLayer runs on each training micro-batch. Outputs are stored in `target_cache` (a list of `(B, T)` cpu tensors, one per micro-step).
- **Epochs 1–19, 21–39, …**: `target_cache` is frozen. Each micro-step retrieves its targets by `target_cache[micro_step_global % len(target_cache)]`.

This mirrors k-means EM: fixed assignments for N steps, then re-cluster.

### State variables

| Variable | Purpose |
|---|---|
| `target_cache` | List of `(B, T)` cpu tensors, one per micro-step of the last refresh epoch |
| `cache_built_for_epoch` | Dataloader epoch when cache was last built (prevents double-clear mid-epoch) |
| `micro_step_global` | Global micro-step counter for cache indexing; resets at start of each refresh epoch |
| `prev_dataloader_epoch` | Detects epoch transitions from `dataloader_state_dict['epoch']` |

### What you must implement

Fill in `BoxedLayer.__call__` in [scripts/encoder_pretrain.py](../scripts/encoder_pretrain.py) with your custom NN / nearest-neighbor logic before training.
