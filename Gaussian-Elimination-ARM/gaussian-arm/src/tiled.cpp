#include "common.hpp"
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
// Cache-blocked elimination with TILE_SIZE x TILE_SIZE tiles.
// Pivot steps are handled in panels of TILE_SIZE. The panel's pivot rows are copied ("packed")
// into contiguous TILE_SIZE x TILE_SIZE tiles; without packing, pivot rows sit a power-of-two
// distance apart in memory and evict each other from the L1 cache. Every remaining row then
// receives all of the panel's updates one tile at a time, so each packed tile (8 KB of doubles
// at 32) stays in L1 while it is reused across all remaining rows, and each row is streamed
// through the cache once per panel instead of once per pivot step.
// Each element receives the same updates, with the same multipliers, in the same order as the
// baseline, so the result is bitwise identical.
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
    std::vector<double> F(static_cast<size_t>(n)*B);      // multipliers for the current panel
    std::vector<double> D(static_cast<size_t>(B)*B);      // packed pivot rows, panel columns
    std::vector<double> P(static_cast<size_t>(B)*(n+B));  // packed pivot rows, remaining columns
    for(int kb=0;kb<n-1;kb+=B) {
        const int kend=std::min(kb+B,n-1);                // this panel's pivot steps: kb..kend-1
        const int ntiles=(n-kend+B-1)/B;
        // 1. Bring the panel's own pivot rows up to date (full width), as in the baseline.
        for(int k=kb;k<kend;++k) {
            const double pivot=A(k,k);
            for(int i=k+1;i<kend;++i) {
                const double factor=A(i,k)/pivot;
                for(int j=k;j<n;++j) A(i,j)-=factor*A(k,j);
            }
        }
        // Pack the finished pivot rows: D holds columns kb..kend-1, P holds columns kend..n-1
        // as ntiles contiguous B x B tiles.
        for(int k=kb;k<kend;++k) {
            for(int j=kb;j<kend;++j) D[static_cast<size_t>(k-kb)*B+(j-kb)]=A(k,j);
            for(int t=0;t<ntiles;++t) {
                const int jb=kend+t*B, w=std::min(B,n-jb);
                double* p=&P[(static_cast<size_t>(t)*B+(k-kb))*B];
                for(int jj=0;jj<w;++jj) p[jj]=A(k,jb+jj);
            }
        }
        // 2. Remaining rows: compute the panel's multipliers, updating the panel columns as we go.
        for(int i=kend;i<n;++i) {
            double* row=&A(i,kb);                          // row[jj] is A(i,kb+jj)
            double* f=&F[static_cast<size_t>(i)*B];
            for(int k=0;k<kend-kb;++k) {
                const double* d=&D[static_cast<size_t>(k)*B];  // d[jj] is A(kb+k,kb+jj)
                const double factor=row[k]/d[k];
                f[k]=factor;
                for(int jj=k;jj<kend-kb;++jj) row[jj]-=factor*d[jj];
            }
        }
        // 3. Remaining rows, remaining columns: apply all panel updates one packed tile at a time.
        for(int t=0;t<ntiles;++t) {
            const int jb=kend+t*B, w=std::min(B,n-jb);
            const double* tile=&P[static_cast<size_t>(t)*B*B];
            for(int i=kend;i<n;++i)
                apply_tile(&A(i,jb),tile,&F[static_cast<size_t>(i)*B],kend-kb,w);
        }
    }
}
int main(int argc,char** argv) { return run_benchmark(argc,argv,"tiled"); }
