# TOKENIZER.md — BPE Tokenizer Experimentation Research

## 1. Vocabulary Size: What to Compare

**Recommended range: 4K, 8K, 16K, 32K**

- Skip 1K — near byte-level, pathological behavior
- The `runcpu` baseline config (depth=6, ~10M non-embedding transformer params) has embedding tables that **dominate** parameter count at large vocab sizes:
  - At 32K vocab: `wte + lm_head + 3 value_embed tables ≈ 37M+ params` vs ~10M transformer matrix params
  - The `sqrt(N)` heuristic for optimal vocab size points to ~3K–4K for a ~10M param model
- Current 32K default may be too large for this model scale — the embedding table is the model
- Most informative experimental range: **4K vs 8K vs 16K vs 32K**

## 2. Sequence Length and Information Density

- Key variable is **compression ratio** (bytes per token):
  - ~2.5 bytes/token at 8K vocab
  - ~4.5 bytes/token at 64K vocab
- For a fixed 512-token window, a 64K tokenizer sees ~80% more bytes per context than an 8K tokenizer
- The codebase's `token_bytes.pt` / `evaluate_bpb()` infrastructure is designed for exactly this — bpb normalizes by bytes, not tokens

## 3. Tokenization Family

- Stay with **byte-level BPE + regex split** (current approach)
- Do not mix in character-level or SentencePiece-without-byte-fallback in the same study — changes too many confounding variables
- Byte-level BPE guarantees lossless encoding of any UTF-8 input regardless of vocab size

## 4. Tokenizer Training Data Requirements

- BPE frequency statistics stabilize at ~100M–1B characters for English-dominant data
- Current 2B char budget is more than sufficient; reducing it would not materially affect vocab quality
- **Keep tokenizer training data fixed** while varying vocab size to isolate the vocab variable

## 5. Split Patterns

- The `\p{N}{1,2}` number-split rule is the only pattern parameter with any validated effect in the codebase
- Rule of thumb: use `{1,1}` for vocab ≤8K, `{1,2}` for 8K–32K, `{1,3}` for 32K+
- This is explicitly unvalidated across vocab sizes (marked as TODO in code) — worth testing as a secondary variable after vocab size is understood

## 6. Fair Comparison Methodology

- **Always use bpb, never raw perplexity** — the codebase already enforces this; perplexity is vocab-size-dependent
- **Control for bytes seen, not steps**: a 4K tokenizer at 2.0 bytes/token and a 32K tokenizer at 3.5 bytes/token see different amounts of text in the same number of steps — adjust step counts proportionally, or report on a per-byte basis
- **Acknowledge parameter count differences**: value_embeds scale with vocab size and are dominant at small model scale — either hold architecture fixed and accept this, or **disable value_embeds for vocab experiments** to isolate the tokenizer effect
- Fix random seed across vocab experiments for a fair comparison of loss curves

## Key Experiment Recommendations

| Priority | Experiment | What it tests |
|----------|-----------|---------------|
| 1 | 8K vs 16K vs 32K vocab, same architecture | Optimal vocab size for this model scale |
| 2 | Fix step count per byte, not per token | Whether compression ratio affects apparent improvement |
| 3 | value_embeds on vs off × vocab size | Whether value_embeds interact with vocab scale |
| 4 | Split pattern `{1,1}` vs `{1,2}` at 8K | Number tokenization effect |

---
*Researched: 2026-04-02*
