# Optimizing Gaussian Elimination on the ASUS Tinker Board 2

A performance study of pivotless Gaussian elimination on the ASUS Tinker Board 2 (Rockchip RK3399: 2× Cortex-A72 + 4× Cortex-A53, ARMv8 NEON), completed for CS 317 Software Performance Optimization at Whitman College.

The algorithm, the operation count, and the input matrices stay fixed. Only the implementation changes, so every speedup comes from how well the code uses the hardware:

1. Baseline row-major implementation
2. Transposed memory layout
3. Cache tiling (32×32 tiles, sized for the 32 KB L1 cache)
4. Tiling + OpenMP row-level parallelism across all six cores
5. Tiling + OpenMP + NEON SIMD (two doubles per 128-bit register)

Full write-up: [Optimizing Gaussian Elimination on the ASUS Tinker Board 2](Gaussian-Elimination-ARM/gaussian-arm/docs/Gaussian-Elimination-Report.pdf)

## Results

Mean of 10 runs per configuration on a 4 GB Tinker Board 2 running TinkerOS (Debian), compiled with `g++ -O3 -march=armv8-a+simd -fopenmp`, using identical fixed-seed random matrices for every version.

| Variant | 256×256 | 512×512 | 1024×1024 | Speedup at 1024 |
|---|---:|---:|---:|---:|
| Baseline | 118 ms | 945 ms | 7,760 ms | 1.00× |
| + Transposition | 91 ms | 731 ms | 5,810 ms | 1.34× |
| + Cache tiling | 62 ms | 498 ms | 3,822 ms | 2.03× |
| + OpenMP | 26 ms | 203 ms | 1,441 ms | 5.39× |
| + NEON | 19 ms | 154 ms | 1,098 ms | **7.07×** |

## What profiling showed

Profiling used `perf`, `gprof`, and ARM Streamline.

- **Baseline:** about 70% of runtime in the innermost loop, an L1 miss rate of about 22%, no compiler auto-vectorization, and only one core in use.
- **Transposition:** column accesses became contiguous, cutting runtime by about 25% at n = 1024.
- **Cache tiling:** the L1 miss rate fell from 22% to 8%, a further 1.47–1.52× speedup.
- **OpenMP:** the largest single gain, 2.38–2.65× over the tiled version.
- **NEON:** a further 24–27% reduction, with SIMD utilization around 85% at large sizes. Gains are capped because each 128-bit register holds only two doubles.
- **Code structure:** a header-only, inlined build ran slightly faster than a modular one (154 ms vs. 158 ms at n = 512), a small effect next to memory and parallelism changes.

**Takeaway:** on this board, Gaussian elimination is memory-bound before it is compute-bound. Improving locality first made the later parallel and SIMD gains possible.

## About the code

The code in `src/` re-implements each variant as described in the report. The tiled, OpenMP, and NEON versions go further than the report's description of tiling: they also group pivot steps into panels of 32 and copy each 32×32 block of pivot rows into a contiguous buffer, so the block stays in L1 while it updates every remaining row. The timings above are the original measurements from the study, not results from this code.

## Build and run

```bash
cd Gaussian-Elimination-ARM/gaussian-arm
make
./bin/baseline 512 10
OMP_NUM_THREADS=6 ./bin/openmp 512 10
OMP_NUM_THREADS=6 ./bin/neon 512 10
make benchmark   # runs n = 256, 512, 1024 and writes benchmarks/local_results.csv
```

Requires GCC or Clang with OpenMP. On non-ARM machines, `neon` builds a scalar fallback and labels its output accordingly. Output columns: `version,n,runs,avg_ms,best_ms,worst_ms,checksum`.

## Notes

- **No pivoting.** The study isolates performance rather than numerical stability, so this is not a general-purpose solver. Inputs are diagonally shifted to avoid tiny pivots, which reduces but does not eliminate the risk.
- Timing covers elimination only. Matrix generation and copying are excluded; the transposed variant's transpose is included.
- Tiling processes pivot steps and columns in 32×32 tiles. Unlike production LU libraries, it does not use matrix-multiply (BLAS-3) kernels.
- All five versions produce bit-for-bit identical matrices, checked element by element at sizes from 1 to 1024 on x86-64 and on ARM64 with NEON, at several thread counts. The benchmark's checksum is only a quick sanity check.

## Files

All project files are in `Gaussian-Elimination-ARM/gaussian-arm/`:

- `src/common.hpp`: matrix type, input generation, and benchmark harness
- `src/*.cpp`: one file per variant
- `benchmarks/report_results.csv`: timings from the report
- `benchmarks/run.sh`: benchmark runner
- `docs/Gaussian-Elimination-Report.pdf`: the full report
