# Encoder Head Design Decision

Design record for the output projection layer in `scripts/encoder_pretrain.py`.

---

## Decisions Made

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Output projection head type | **Fixed `boxed_layer.U` matrix** (not a learned `stub_head`) | User controls class geometry; `NUM_LABELS` is independent of `vocab_size`; no optimizer overhead |
| 2 | Label space dimensions | **`(NUM_LABELS, n_embd)`** — user-defined label count, not vocab-sized | Label space is a separate domain (not token vocabulary) |
| 3 | Target format | **Hard integer indices `(B, T)`** → `F.cross_entropy` integer-index API | Sufficient for cluster assignment; soft targets remain an open option |
| 4 | BoxedLayer input | **`wte` embeddings** (input token embeddings, before transformer) | Breaks circularity — see Circularity Problem below |
| 5 | BoxedLayer output | **Hard class indices `(B, T)` int** → become `targets` for cross-entropy | Matches `F.cross_entropy` integer-index API |
| 6 | Backprop through BoxedLayer | **No** — called under `torch.no_grad()` | BoxedLayer is an assignment oracle, not a learned layer |
| 7 | Target refresh schedule | **Every step** — BoxedLayer runs on each micro-batch | `FFUFeaturizer.feat.update` is fast matrix accumulation; no reason to cache |

---

## The problem

`GPTEncoder` outputs hidden states of shape `(B, T, n_embd)`. To compute a cross-entropy training signal, these must be projected to logits of size `(B, T, NUM_LABELS)`. A fixed projection matrix `U` (shape `(NUM_LABELS, n_embd)`) is used — user-controlled, not learned.

`U` is maintained by `FFUFeaturizer` inside `BoxedLayer`, which runs on each micro-batch and returns updated cluster directions.

---

## The Circularity Problem (and why input embeddings fix it)

**Original broken design (hidden states as input):**

```
U  ← FFUFeaturizer(encoder_hidden_states)   # U = directions in hidden-state space
targets ← argmax(hidden @ U.T)              # cluster = which U-direction is dominant
loss = cross_entropy(hidden @ U.T, targets) # train encoder to align hidden with U
```

`U` chases the encoder's hidden states; the encoder is trained toward `U`. The fixed point is mode collapse: every token maps to one dominant direction → loss → 0, representations carry no information about the input text.

**Fixed design (wte embeddings as input):**

```
input_emb ← encoder.transformer.wte(x)     # token embeddings, BEFORE transformer
U  ← FFUFeaturizer(input_emb)              # U = directions in INPUT embedding space
targets ← argmax(input_emb @ U.T)          # which input cluster does this token fall in?
loss = cross_entropy(hidden @ U.T, targets) # encoder trained to predict input cluster
                                            #   from its deep representations
```

`U` is now derived from `wte` — the embedding table output, the raw token representation before any transformer layer. This matches the pattern from Coates & Ng (2012) and Krähenbühl et al. (2015): clusters are always computed from input patches/features, never from the model's own intermediate activations. The encoder has to actually learn something about the input structure to predict the cluster assignments.

---

## Data flow

```
x (B, T) token IDs
    │
    ├── orig_encoder.transformer.wte(x) → input_emb (B, T, n_embd)    [no grad, input space]
    │         │
    │    boxed_layer(input_emb)                             [no grad, FFUFeaturizer update]
    │         │
    │    targets (B, T) int                                 [discrete, not in compute graph]
    │
    └── encoder(x) → hidden (B, T, n_embd)                 [grad flows here]
              │
    compute_loss(hidden, targets)
              │
    hidden @ boxed_layer.U.T → logits → cross_entropy → loss.backward()
```

**Gradient path:**

```
loss.backward()
      │
  cross_entropy   ← targets select which class to penalize (discrete, no grad)
      │
  logits (B, T, NUM_LABELS)
      │
  hidden @ U.T    ← U is fixed per step (output of FFUFeaturizer, no grad)
      │
  hidden (B, T, n_embd)   ← gradient arrives here
      │
  encoder(x)              ← gradient flows all the way back through transformer
```

BoxedLayer is **never in the gradient path**. It runs on `input_emb` (detached wte output) to produce integer labels, then steps aside entirely.

---

## Open decision: target format

Currently `compute_loss` expects integer class indices `(B, T)` and uses `F.cross_entropy` with `ignore_index=-1`.

**If you want soft targets** (e.g., probability distributions over classes):
```python
log_probs = F.log_softmax(logits, dim=-1)
loss = -(targets_float * log_probs).sum(dim=-1).mean()
```

**If you want regression targets** (real-valued embeddings):
```python
loss = F.mse_loss(logits, targets_float)
```

See `compute_loss` in [scripts/encoder_pretrain.py](../scripts/encoder_pretrain.py) — the decision point is marked with an `OPEN DECISION` comment.

---

## Literature grounding

The design follows Coates & Ng (2012) "An Analysis of Single-Layer Networks in Unsupervised Feature Learning" and Krähenbühl et al. (2015) "Data-dependent Initializations of Convolutional Neural Networks":

- **Clusters are always computed from inputs** (raw patches / token embeddings), never from the model's own hidden representations.
- **U is computed before or concurrently with training** — not after the encoder has already transformed the input.
- The encoder is trained to **predict input-space structure from deep features**, which forces it to learn the input distribution rather than collapsing to self-consistent trivial solutions.

Key quote (Krähenbühl et al.): *"The first layer's rough destination can be computed from a few thousand unlabeled patches in seconds, and gradient descent is instead asked to rediscover it from noise."*
