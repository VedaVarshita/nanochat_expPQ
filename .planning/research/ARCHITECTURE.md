# ARCHITECTURE.md — Transformer Architecture Variants Research

**Confidence:** MEDIUM-HIGH overall; based on codebase analysis + training data through August 2025

## 1. RoPE Base Theta — Highest Signal, 1-Line Change

The codebase uses `theta=100000`. At seq_len=512, the lowest-frequency RoPE channels barely rotate at any theta above ~5K — they contribute no positional signal at this sequence length. The theta value is essentially unconstrained and untested at this scale.

**Recommended experiment:** Sweep `{500, 2000, 10000}` vs baseline `100000` — 1 line change per run, very high signal-to-effort ratio.

## 2. Value Embeddings (VE) Ablation

The alternating-layer ResFormer-style value embeddings are a significant and custom component. They blend via `ve_gate` into the first 12 residual channels and interact with `x0_lambdas`.

**Ablation:** Freeze `ve_gate` weights at zero → clean removal of VE contribution. Track `x0_lambdas` alongside — they interact with what the VE gate "sees" via early residual channels.

## 3. relu² vs SwiGLU

relu² was specifically adopted in the modded-nanoGPT lineage (which this codebase descends from) because it interacts better with Muon's orthogonal update structure. SwiGLU advantages are well-established at large scale but less clear at small scale with Muon.

**Experiment:** SwiGLU at parameter parity — adjust intermediate dim to `(8/3) × d_model` instead of `4 × d_model` to match parameter count. Worth doing to validate the relu² choice.

## 4. Head Dimension Sweep

At depth=6, dim=384: `head_dim=128` gives only 3 heads — potentially too few for attention diversity. `--head-dim` CLI flag already exists.

**Informative comparison:** head_dim=32 (12 heads) vs head_dim=64 (6 heads, baseline) vs head_dim=128 (3 heads).

## 5. GQA at Small Scale — Skip for Now

GQA is an inference-efficiency technique. At 6 layers, seq_len=512, the KV cache is tiny. The parameter reduction from fewer KV heads would need to be re-invested elsewhere for a fair comparison. Skip until inference efficiency matters.

## 6. Scalar Ablations (Fast, Informative)

Each requires 2–3 lines of init change. Ablate systematically, not simultaneously — they interact.

- `resid_lambdas`: flat 1.0 vs current decaying schedule
- `x0_lambdas`: freeze at zero (also affects what VE gate sees)
- `backout_lambda`: freeze at 0
- `smear_lambda`: freeze at 0

## 7. Depth vs Width Tradeoffs

All hold dim≈384 for fair comparison. Well-studied in Kaplan 2020 + Chinchilla 2022 literature.

| Config | depth | aspect_ratio | dim |
|--------|-------|-------------|-----|
| Shallow-wide | 4 | 96 | 384 |
| Baseline | 6 | 64 | 384 |
| Deep-narrow | 8 | 48 | 384 |

## 8. MPS Constraint

Window pattern must stay `--window-pattern=L` (full causal). Sliding window via SDPA requires FA3 or a new mask implementation. All architecture experiments must use `L`.

## Experiment Priority

**Tier 1 — 1–3 line changes, highest expected signal:**
1. RoPE theta sweep: {500, 2000, 10000} vs baseline 100000
2. VE ablation: freeze ve_gate at zero
3. x0_lambdas ablation: freeze at zero
4. relu² vs SwiGLU (parameter-equalized)
5. resid_lambdas: flat 1.0 vs decaying schedule

**Tier 2 — Existing CLI flags, structural comparisons:**
6. head_dim sweep: 32 / 64 (baseline) / 128
7. Depth/width: depth=4 vs 6 vs 8 at fixed dim=384
8. backout and smear ablations

**Tier 3 — New code, higher effort:**
9. Differential Attention (~60 lines)
10. Parallel MLP+Attention block (~30 lines)

## Signal Threshold

Non-determinism baseline is ±0.008 CORE, ±~0.003–0.005 bpb estimated.
- Single run: claim win only if improvement >0.01 bpb
- Two replications: >0.005 bpb is meaningful

---
*Researched: 2026-04-02*
