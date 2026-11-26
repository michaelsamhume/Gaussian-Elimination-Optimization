#include "common.hpp"
#include <omp.h>
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
// Cache-blocked elimination (same tiling and packing as tiled.cpp) with OpenMP across rows.
// Within each phase the rows being updated are independent, so they are split across threads.
// One parallel region per panel keeps thread start-up cost low.
// Apply nk panel updates to one tile of a row: row[jj] -= f[k] * tile[k][jj].
// The row, the packed tile and the multipliers never overlap, which __restrict tells the
// compiler so it can skip runtime overlap checks on these short loops.
static inline void apply_tile(double* __restrict row,const double* __restrict tile,
                              const double* __restrict f,int nk,int w) {
    for(int k=0;k<nk;++k) {
        const double factor=f[k];
        const double* __restrict p=tile+static_cast<size_t>(k)*TILE_SIZE;
        for(int jj=0;jj<w;++jj) row[jj]-=factor*p[jj];
    }
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
            // 1. Panel pivot rows; the barrier after each step publishes the next pivot row.
            for(int k=kb;k<kend;++k) {
                const double pivot=A(k,k);
                #pragma omp for schedule(static)
                for(int i=k+1;i<kend;++i) {
                    const double factor=A(i,k)/pivot;
                    for(int j=k;j<n;++j) A(i,j)-=factor*A(k,j);
                }
            }
            // Pack the finished pivot rows (one row per iteration).
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
                    for(int jj=k;jj<kend-kb;++jj) row[jj]-=factor*d[jj];
                }
            }
            // 3. Remaining columns, one packed tile at a time. Threads never share a row within a
            //    tile and the packed tiles are read-only here, so no barrier is needed between tiles.
            for(int t=0;t<ntiles;++t) {
                const int jb=kend+t*B, w=std::min(B,n-jb);
                const double* tile=&P[static_cast<size_t>(t)*B*B];
                #pragma omp for schedule(static) nowait
                for(int i=kend;i<n;++i) {
                    apply_tile(&A(i,jb),tile,&F[static_cast<size_t>(i)*B],kend-kb,w);
                }
            }
        }
    }
}
int main(int argc,char** argv) { return run_benchmark(argc,argv,"openmp"); }
