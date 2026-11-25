#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks
OUT="${1:-benchmarks/local_results.csv}"
echo 'version,n,runs,avg_ms,best_ms,worst_ms,checksum' > "$OUT"
for n in 256 512 1024; do
  for version in baseline transposed tiled openmp neon; do
    "./bin/$version" "$n" 10 >> "$OUT"
  done
done
echo "Wrote $OUT (local measurements, not the original report's data)"
