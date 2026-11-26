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
// Cache-blocked, OpenMP-parallel elimination (as in openmp.cpp) with double-precision
// ARMv8 NEON vectorization (two doubles per 128-bit vector) in every row update.
// Non-ARM platforms compile a scalar fallback for portability; it is NOT a NEON benchmark.

// row[j] -= factor * pivot_row[j] for j in [j0, j1).
// NEON multiplies, then subtracts (not fused), matching the scalar arithmetic exactly.
// The two rows never overlap, which __restrict tells the compiler.
static inline void row_update(double* __restrict row,const double* __restrict pivot_row,
                              double factor,int j0,int j1) {
    int j=j0;
#if HAVE_NEON && defined(__aarch64__)
    const float64x2_t f=vdupq_n_f64(factor);
    for(;j+1<j1;j+=2) {
        const float64x2_t x=vld1q_f64(row+j);
        const float64x2_t y=vld1q_f64(pivot_row+j);
        vst1q_f64(row+j,vsubq_f64(x,vmulq_f64(f,y)));
    }
#endif
    for(;j<j1;++j) row[j]-=factor*pivot_row[j];
}

void eliminate(Matrix& A) {
    const int n=A.n;
    const int B=TILE_SIZE;
    std::vector<double> F(static_cast<size_t>(n)*B);
    std::vector<double> D(static_cast<size_t>(B)*B);
    std::vector<double> P(static_cast<size_t>(B)*(n+B));
    for(int kb=0;kb<n-1;kb+=B) {
        const int kend=std::min(kb+B,n-1);
        const int ntiles=(n-kend+B-1)/B;
        #pragma omp parallel
        {
            // 1. Panel pivot rows (full width).
            for(int k=kb;k<kend;++k) {
                const double pivot=A(k,k);
                const double* pivot_row=&A(k,0);
                #pragma omp for schedule(static)
                for(int i=k+1;i<kend;++i) {
                    double* row=&A(i,0);
                    row_update(row,pivot_row,row[k]/pivot,k,n);
                }
            }
            // Pack the finished pivot rows.
            #pragma omp for schedule(static)
            for(int k=kb;k<kend;++k) {
                for(int j=kb;j<kend;++j) D[static_cast<size_t>(k-kb)*B+(j-kb)]=A(k,j);
                for(int t=0;t<ntiles;++t) {
                    const int jb=kend+t*B, w=std::min(B,n-jb);
                    double* p=&P[(static_cast<size_t>(t)*B+(k-kb))*B];
                    for(int jj=0;jj<w;++jj) p[jj]=A(k,jb+jj);
                }
            }
            // 2. Multipliers and panel columns for the remaining rows.
            #pragma omp for schedule(static)
            for(int i=kend;i<n;++i) {
                double* row=&A(i,kb);
                double* f=&F[static_cast<size_t>(i)*B];
                for(int k=0;k<kend-kb;++k) {
                    const double* d=&D[static_cast<size_t>(k)*B];
                    const double factor=row[k]/d[k];
                    f[k]=factor;
                    row_update(row,d,factor,k,kend-kb);
                }
            }
            // 3. Remaining columns, one packed tile at a time.
            for(int t=0;t<ntiles;++t) {
                const int jb=kend+t*B, w=std::min(B,n-jb);
                const double* tile=&P[static_cast<size_t>(t)*B*B];
                #pragma omp for schedule(static) nowait
                for(int i=kend;i<n;++i) {
                    double* row=&A(i,jb);
                    const double* f=&F[static_cast<size_t>(i)*B];
                    for(int k=0;k<kend-kb;++k)
                        row_update(row,tile+static_cast<size_t>(k)*B,f[k],0,w);
                }
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
