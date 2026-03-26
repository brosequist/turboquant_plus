# Experiment: Layer-Adaptive Extended Context + Decode Speed

Branch: `experiment/layer-adaptive-extended-ctx`

## Layer-Adaptive Mode 2: Context Scaling (2K-16K)

| Context | Mode 2 turbo3 tok/s | q8_0 tok/s | Ratio |
|---------|-------------------|-----------|-------|
| 2048 | 4688 | 4762 | 0.984x |
| 4096 | 3045 | 3105 | 0.981x |
| 8192 | 2294 | 2349 | 0.977x |
| 16384 | 1735 | 1778 | 0.976x |

**Holds at 97-98% across all tested depths.** Layer-adaptive mode 2 is safe for long context.

## Decode Speed (M5 Max, Qwen3.5-35B-A3B, 4K context)

| Cache Type | Prompt tok/s | Decode tok/s | KV Cache Size |
|-----------|-------------|-------------|---------------|
| turbo3 | 237.6 | 76.3 | 17.5 MiB |
| q8_0 | 246.1 | 84.0 | 42.5 MiB |
| ratio | 0.97x | **0.91x** | 0.41x |

**Decode is 91% of q8_0 on M5 Max.** The 2.4x smaller KV cache partially compensates for the more complex dequant.

### vs External Tester Reports

| Tester | Hardware | Context | Decode ratio |
|--------|----------|---------|-------------|
| Us (M5 Max) | Apple M5 Max 128GB | 4K | 0.91x |
| @tarruda | M1 Ultra 128GB | 4K | ~0.65x (17→11 tok/s) |
| Anon | M1 Max 64GB | 42K | 0.36x (4 vs 11 tok/s) |

The decode gap is MUCH worse on older hardware (M1 vs M5). Likely causes:
1. M1 lacks Tensor API (`has tensor = false`) — may use slower code path
2. M1 has lower memory bandwidth per GPU core
3. The 42K context test has a much larger KV cache to scan per token

### External Benchmark: Mario (M1 Max 64GB, 32K prompt, main TOT)

| Config | KV Cache | Prefill tok/s | Decode tok/s | vs q8_0 decode |
|--------|----------|--------------|-------------|----------------|
| llama.cpp q8_0 | 8-bit | 442 | 41.8 | baseline |
| **llama.cpp turbo3** | **3.5-bit** | **417** | **34.6** | **0.83x** |
| mlx-vlm fp16 | 16-bit | 488 | 42.6 | 1.02x |
| mlx-vlm q8 uniform | 8-bit | 480 | 32.8 | 0.78x |
| mlx-vlm TurboQuant 4-bit | 4-bit | 471 | 13.1 | 0.31x |
| mlx-vlm TurboQuant 3.5-bit | 3.5-bit | 450 | 7.9 | 0.19x |

**Key takeaways:**
1. Our llama.cpp turbo3 decode (34.6) is **4.4x faster** than mlx-vlm's turbo 3.5-bit (7.9)
2. Our 0.83x decode ratio beats even mlx-vlm's q8 uniform (0.78x)
3. Prefill is near-parity: 417 vs 442 = 0.94x
4. The optimized dequant on main TOT is working — earlier reports of 0.36x were on older code

### Note on 1K Timing Unreliability

The 1024-context measurements show unrealistic numbers (millions of tok/s) because the total compute time is sub-millisecond. Metal's async command buffer dispatch means the CPU timer reports the submission time, not the actual GPU execution time. Use 2K+ context with 2+ chunks for reliable measurements.
