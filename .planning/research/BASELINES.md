# BASELINES.md — Small-Scale LLM Pretraining Baselines Research

**Confidence:** HIGH — all findings derived from primary sources in the repo

## 1. Your d6 runcpu Config Parameters

- `model_dim=384`, ~10.6M transformer matrix params, ~23.2M scaling params, ~73.5M total (heavily embedding-dominated due to value_embeds)
- 5000 iters × 16,384 batch = 81.9M tokens → token:scaling_param ratio ≈ 3.53 (vs compute-optimal ~10.5×)
- Total training FLOPs ≈ 6.4×10^15 — about 156× below the minimum nanoChat uses for scaling law experiments (1×10^18)
- **The model is massively undertrained at this config.** You are measuring a snapshot on the loss curve, not a converged model. This is expected and fine for relative comparison purposes.

## 2. Scale for Research Signal: d6 vs d12

The nanoChat community's research standard is **d12, not d6**. The README explicitly states: "For quick experimentation my favorite scale is to train a 12-layer model." Autoresearch results from d12 "easily translated" to d24. d6→d12 transfer is less reliable.

- **d6 use case:** Detecting catastrophically bad ideas (does the loss explode? does the metric degrade by >5%?)
- **d12 use case:** Reliable directional signal for architectural and tokenizer comparisons
- For your local MPS setup, d6 is the practical starting point given time constraints; validate anything interesting at d12 before drawing conclusions

## 3. DCLM CORE Score

- Formula: `mean((accuracy - random_baseline) / (1 - random_baseline))` across ~22 tasks (ARC, MMLU, SQuAD variants, etc.)
- GPT-2 (1.6B) reference: 0.256525
- At d6/5000 steps with `--max-per-task=16`: expect **~0.00–0.12, extremely noisy** — not a usable decision metric at this scale
- CORE is only meaningful at adequate training (token:param ratio ≥5×), with `--max-per-task=-1`, averaged over multiple runs
- Use CORE only as a final sanity check after BPB looks good

## 4. BPB is Your Primary Metric

BPB (bits-per-byte) is tokenizer-invariant, smooth, logged every 100 steps, and correlates well with capability. The nanoChat LEADERBOARD explicitly says: "val_loss number is a great, smooth metric to track relative performance w.r.t. and has less noise than CORE."

- At d6/5000 steps: expect val_bpb in the **0.85–1.00** range
- At GPT-2 scale (d24, ClimbMix), current SOTA: **0.718**
- BPB is only comparable when the **validation set and tokenizer are held fixed** — explicitly noted when ClimbMix replaced FineWeb-EDU as training data

## 5. Minimum Compute for Meaningful Signal

| Goal | FLOPs needed | Achievable at d6/5000 steps? |
|------|-------------|------------------------------|
| Directional BPB comparison | ~10^17–10^18 | Marginal (6.4×10^15) |
| Reliable CORE comparisons | ~10^18–10^19 | No |
| Reliable architecture comparison | ~10^17–10^18 | Yes (BPB curve shape) |

At MPS with fixed data and fixed seed, run-to-run variance should be <0.005 bpb. Treat any difference **>0.01 bpb** as meaningful signal. Differences <0.005 bpb are noise.

## 6. Reproducibility Best Practices

- Training is mildly non-deterministic on CUDA (Run 4 in the leaderboard showed 0.016 CORE spread across 7 identical runs). MPS behavior is untested but likely more deterministic due to deterministic Metal kernels.
- **Never retrain the tokenizer mid-experiment** — keep the trained tokenizer checkpointed and reused across all runs
- **Track the full BPB curve** (all eval points), not just the final value — curve shape reveals training dynamics differences
- Fix the random seed for weight initialization and data shuffling across all comparison runs
- When comparing tokenizer variants, control for bytes seen (not steps) — see TOKENIZER.md

## Recommended Baseline Protocol

1. Run `runcpu.sh` exactly as-is → establish reference BPB curve
2. Re-run with same seed → confirm reproducibility (delta should be <0.005 bpb)
3. All subsequent experiments: compare BPB curve shape and final val_bpb against this baseline
4. Only run CORE eval on experiments where BPB improvement is ≥0.01 from baseline

---
*Researched: 2026-04-02*
