#include "common.hpp"
// Reconstructed transpose-based layout: columns of the original become contiguous rows.
void eliminate(Matrix& A) {
    const int n=A.n;
    Matrix T(n);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) T(j,i)=A(i,j);
    for(int k=0;k<n-1;++k) {
        const double pivot=T(k,k);
        for(int i=k+1;i<n;++i) {
            const double factor=T(k,i)/pivot;
            for(int j=k;j<n;++j) T(j,i)-=factor*T(j,k);
        }
    }
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) A(i,j)=T(j,i);
}
int main(int argc,char** argv) { return run_benchmark(argc,argv,"transposed"); }
