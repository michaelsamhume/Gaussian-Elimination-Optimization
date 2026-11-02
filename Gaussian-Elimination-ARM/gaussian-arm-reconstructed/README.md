# Gaussian Elimination Optimization on the ASUS Tinker Board 2

**Reconstructed C++ implementations based on an 11-page project report.** These files are **not the original source code** used to produce the published benchmark measurements. They are independently reconstructed from the report's baseline listing and descriptions of the optimization techniques.

## Project

Compare pivotless Gaussian elimination implementations on the ASUS Tinker Board 2 (RK3399, dual Cortex-A72 + quad Cortex-A53) using:

1. Baseline row-major implementation
2. Transposed memory layout
3. Cache tiling (`TILE_SIZE=32`)
4. Tiling + OpenMP row-level parallelism
5. Tiling + OpenMP + ARMv8 NEON two-double SIMD

The baseline follows the report's listing. Optimized implementations are plausible reconstructions, **not verbatim reproductions**. In particular, transposition and blocking details are under-specified in the report, so these are representative designs and may perform differently from the reported results.

## Build and run

```bash
make
./bin/baseline 512 10
./bin/tiled 512 10
OMP_NUM_THREADS=6 ./bin/openmp 512 10
OMP_NUM_THREADS=6 ./bin/neon 512 10
make benchmark
```

`make benchmark` runs sizes 256, 512, and 1024 and writes `benchmarks/local_results.csv`. Requires GCC or Clang with OpenMP support for the parallel targets. On non-ARM machines, `neon` builds a scalar fallback and explicitly labels its results accordingly. For comparable ARM results, run on an ARMv8 board and use consistent thread counts, power, thermal, and compiler settings.

Output columns: `version,n,runs,avg_ms,best_ms,worst_ms,checksum`. Checksums help detect large numerical differences, but they are not a substitute for a full matrix-by-matrix correctness test.

## Reported results (historical, **not reproduced here**)

| Variant | 256×256 | 512×512 | 1024×1024 |
|---|---:|---:|---:|
| Baseline | 118 ms | 945 ms | 7760 ms |
| Transposed | 91 ms | 731 ms | 5810 ms |
| Tiled | 62 ms | 498 ms | 3822 ms |
| Tiled + OpenMP | 26 ms | 203 ms | 1441 ms |
| Tiled + OpenMP + NEON | 19 ms | 154 ms | 1098 ms |

Values transcribed from the accompanying report, not measured from this reconstructed code. The reported 1024×1024 ratio is 7760/1098 ≈ **7.07×**, though the report describes it as approximately 6×.

## Numerical and performance limitations

- **No pivoting**, matching the report's performance-focused setup; this is not suitable as a general-purpose numerically stable solver.
- Inputs are pseudorandom and diagonal-shifted to reduce tiny pivots, but this does not guarantee stability.
- Matrix generation and copying are excluded from the timed elimination; transposition in the transposed variant occurs inside its timed function.
- `TILE_SIZE=32` blocks the inner loop; it does not implement a fully blocked matrix factorization.
- The NEON variant uses `float64x2_t` on AArch64 to match `double` matrix storage; the report mentions float32 intrinsics as examples, so this is an implementation choice.
- No original profiling traces or build environment are available. Actual performance may differ substantially.

## Files

- `src/common.hpp`: shared matrix, input, and benchmark harness
- `src/*.cpp`: individual variants
- `benchmarks/report_results.csv`: historical numbers from the PDF
- `benchmarks/run.sh`: local benchmark runner
- `docs/`: original research report (if distributed with this package)

## Attribution

Reconstructed from *Optimizing Gaussian Elimination on the ASUS Tinker Board 2*. Preserve the original report and clearly label these source files as reconstructions if publishing this repository.
