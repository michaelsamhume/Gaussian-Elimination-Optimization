#include "common.hpp"
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
// Cache-block the row-major j dimension, preserving elimination dependencies.
void eliminate(Matrix& A) {
    const int n=A.n;
    for(int k=0;k<n-1;++k) {
        const double pivot=A(k,k);
        for(int i=k+1;i<n;++i) {
            const double factor=A(i,k)/pivot;
            for(int jb=k;jb<n;jb+=TILE_SIZE) {
                const int end=std::min(n,jb+TILE_SIZE);
                for(int j=jb;j<end;++j) A(i,j)-=factor*A(k,j);
            }
        }
    }
}
int main(int argc,char** argv) { return run_benchmark(argc,argv,"tiled"); }
