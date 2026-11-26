#include "common.hpp"
#include <vector>
// Transposed layout: T(j,i) = A(i,j), so each column of A is a contiguous row of T.
// The loops are reordered so the innermost loop walks contiguous memory in T.
void eliminate(Matrix& A) {
    const int n=A.n;
    Matrix T(n);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) T(j,i)=A(i,j);
    std::vector<double> factor(n);
    for(int k=0;k<n-1;++k) {
        const double pivot=T(k,k);
        // Compute all multipliers for step k before column k is updated.
        for(int i=k+1;i<n;++i) factor[i]=T(k,i)/pivot;
        for(int j=k;j<n;++j) {
            const double pivot_entry=T(j,k);
            double* col=&T.data[static_cast<size_t>(j)*n];
            for(int i=k+1;i<n;++i) col[i]-=factor[i]*pivot_entry;
        }
    }
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) A(i,j)=T(j,i);
}
int main(int argc,char** argv) { return run_benchmark(argc,argv,"transposed"); }
