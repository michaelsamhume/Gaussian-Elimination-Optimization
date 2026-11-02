#include "common.hpp"
#include <omp.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
// Double-precision vectorization on ARMv8 NEON (two doubles per 128-bit vector).
// Non-ARM platforms compile a scalar fallback for portability; it is NOT a NEON benchmark.
void eliminate(Matrix& A) {
    const int n=A.n;
    for(int k=0;k<n-1;++k) {
        const double pivot=A(k,k);
        const double* pivot_row=&A.data[static_cast<size_t>(k)*n];
        #pragma omp parallel for schedule(static)
        for(int i=k+1;i<n;++i) {
            double* row=&A.data[static_cast<size_t>(i)*n];
            const double factor=row[k]/pivot;
            for(int jb=k;jb<n;jb+=TILE_SIZE) {
                const int end=std::min(n,jb+TILE_SIZE);
                int j=jb;
#if HAVE_NEON && defined(__aarch64__)
                const float64x2_t f=vdupq_n_f64(factor);
                for(;j+1<end;j+=2) {
                    const float64x2_t x=vld1q_f64(row+j);
                    const float64x2_t y=vld1q_f64(pivot_row+j);
                    vst1q_f64(row+j,vsubq_f64(x,vmulq_f64(f,y)));
                }
#endif
                for(;j<end;++j) row[j]-=factor*pivot_row[j];
            }
        }
    }
}
int main(int argc,char** argv) {
#if HAVE_NEON && defined(__aarch64__)
    return run_benchmark(argc,argv,"neon");
#else
    std::cerr << "NOTICE: NEON unavailable; using scalar fallback.\n";
    return run_benchmark(argc,argv,"neon_scalar_fallback");
#endif
}
