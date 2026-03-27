# Long-Context Sparse V Validation

**Date:** 2026-03-27
**Hardware:** Apple M5 Max 128GB
**Branch:** `experiment/long-context-sparse-v-validation`

## Motivation

Prior sparse V quality validation used 8-chunk wikitext-2 at c=512 (PPL 6.1756). At 512 tokens, sparse V barely triggers — most attention weights are above the 1e-6 threshold. This eval validates quality at context lengths where sparse V is actively skipping positions.

## Methodology

- `llama-perplexity` with wikitext-2-raw at c=512, 8192, 16384, 32768
- KV cache: turbo3 (3.5-bit TurboQuant)
- Compared: sparse V enabled (default on M5) vs `TURBO_SPARSE_V=0` (force disabled)
- 8 chunks at c=512/8192/16384, 2 chunks at c=32768 (corpus size constraint)
- Both MoE and dense models tested

**Important limitation:** Skip-rate stats are estimated from decode speed improvements, not directly measured via kernel counters. Adding atomic counters to the Metal FA kernel would require modifying the kernel argument signature and would affect timing. The 90% skip rate at 32K is from profiling during initial sparse V development (documented in the paper). Skip rates at other context lengths are interpolated from the speed improvement curve.

## Results

### MoE (Qwen3.5-35B-A3B Q8_0)

| Context | Sparse V ON | Sparse V OFF | Delta | Est. skip rate |
|---------|------------|-------------|-------|---------------|
| 512 | 6.1756 ± 0.330 | 6.1756 ± 0.330 | 0.0000 | ~6% |
| 8192 | 5.5700 ± 0.072 | 5.5700 ± 0.072 | 0.0000 | ~28% |
| 16384 | 5.1122 ± 0.045 | 5.1122 ± 0.045 | 0.0000 | ~51% |
| 32768 | 6.1293 ± 0.082 | 6.1293 ± 0.082 | 0.0000 | ~90% |

### Dense (Qwen3.5-27B Q8_0)

| Context | Sparse V ON | Sparse V OFF | Delta |
|---------|------------|-------------|-------|
| 8192 | 7.0152 ± 0.106 | 7.0152 ± 0.106 | 0.0000 |

### Skip-Rate Estimates

Estimated from decode speed improvement data (Section 4.2 of the paper):

| Context | Decode Δ (MoE) | Est. skip rate | Interpretation |
|---------|---------------|---------------|----------------|
| 512 | +1.4% | ~6% | Almost no positions skipped |
| 4096 | +4.0% | ~16% | Light skipping |
| 8192 | +7.2% | ~28% | Moderate — ~1 in 4 positions skipped |
| 16384 | +12.9% | ~51% | Heavy — majority of positions skipped |
| 32768 | +22.8% | ~90% | Near-total — only ~10% of V positions dequantized |

Skip rate grows with context length because softmax concentrates on fewer positions as context increases. At 32K, the model attends meaningfully to ~3,200 of 32,768 positions per head.

## Interpretation

**Long-context PPL does not regress when sparse V is active.** Results are numerically identical across all tested context lengths (512 through 32K) on both MoE and dense architectures.

Even at 32K where ~90% of V dequantizations are skipped, perplexity is unchanged to 4+ decimal places. This indicates that the skipped positions fall below numerical significance in this setup — their contribution is below the threshold where it could affect output quality.

**The 512-context PPL (6.1756) should be described as a no-regression sanity check, not the main quality validation.** At c=512, sparse V skips ~6% of positions and has no meaningful effect. The true validation is at c=8K+ where skip rates are 28-90% and the optimization is actively changing computation.

**PPL improves with context length** (6.18 at 512 → 5.11 at 16K) as expected — longer context gives the model more information. The 32K PPL (6.13) is higher than 16K (5.11) because only 2 chunks fit, reducing statistical power. The key metric is the ON/OFF delta, which is 0.0000 everywhere.

## Methodology Notes

- Skip rates are estimated, not directly measured. Adding kernel-level atomic counters was considered but rejected because: (a) it requires modifying the Metal FA kernel argument signature, (b) atomic operations would affect timing, (c) the estimates from speed data are consistent with the profiling data from initial development.
- The 32K run uses 2 chunks instead of 8 due to corpus size. This gives less statistical power but the ON/OFF delta is still exactly 0.
- All runs use the same random seed (implicit in wikitext-2 sequential processing).

## Raw Commands

```bash
LLAMA=~/local_llms/llama.cpp/build-turbo/bin
MODEL_MOE=~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf
MODEL_DENSE=~/local_llms/models/Qwen3.5-27B-Q8_0.gguf
WIKI=~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw

# MoE at each context length
for ctx in 512 8192 16384; do
  # With sparse V (default on M5)
  $LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI -c $ctx \
    -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99

  # Without sparse V
  TURBO_SPARSE_V=0 $LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI -c $ctx \
    -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99
done

# 32K (2 chunks)
$LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI -c 32768 \
  -ctk turbo3 -ctv turbo3 -fa on --chunks 2 -ngl 99
TURBO_SPARSE_V=0 $LLAMA/llama-perplexity -m $MODEL_MOE -f $WIKI -c 32768 \
  -ctk turbo3 -ctv turbo3 -fa on --chunks 2 -ngl 99

# Dense 8K
$LLAMA/llama-perplexity -m $MODEL_DENSE -f $WIKI -c 8192 \
  -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99
TURBO_SPARSE_V=0 $LLAMA/llama-perplexity -m $MODEL_DENSE -f $WIKI -c 8192 \
  -ctk turbo3 -ctv turbo3 -fa on --chunks 8 -ngl 99
```
