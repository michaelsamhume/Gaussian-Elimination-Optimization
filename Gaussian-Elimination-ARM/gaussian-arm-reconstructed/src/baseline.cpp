#include "common.hpp"
void eliminate(Matrix& A) {
    const int n=A.n;
    for (int k=0; k<n-1; ++k) {
        const double pivot=A(k,k);
        for (int i=k+1; i<n; ++i) {
            const double factor=A(i,k)/pivot;
            for (int j=k; j<n; ++j) A(i,j)-=factor*A(k,j);
        }
    }
}
int main(int argc,char** argv) { return run_benchmark(argc,argv,"baseline"); }
