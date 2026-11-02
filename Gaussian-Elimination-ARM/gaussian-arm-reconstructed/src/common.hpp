#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Matrix {
    int n;
    std::vector<double> data;
    explicit Matrix(int size) : n(size), data(static_cast<size_t>(size) * size) {}
    double& operator()(int i, int j) { return data[static_cast<size_t>(i)*n+j]; }
    const double& operator()(int i, int j) const { return data[static_cast<size_t>(i)*n+j]; }
};

inline Matrix make_random_matrix(int n, uint64_t seed = 0xC0FFEEULL) {
    Matrix A(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j) A(i,j)=dist(rng);
    for (int i=0; i<n; ++i) A(i,i)+=(A(i,i)>=0.0 ? 5.0 : -5.0);
    return A;
}

inline double checksum(const Matrix& A) {
    double result=0.0;
    for (double x : A.data) result+=x;
    return result;
}

// All implementations operate on the same row-major Matrix, except the transposed variant.
// Pivoting is intentionally omitted to match the research report's performance focus.
void eliminate(Matrix& A);

inline int run_benchmark(int argc, char** argv, const char* label) {
    int n = argc>1 ? std::atoi(argv[1]) : 512;
    int runs = argc>2 ? std::atoi(argv[2]) : 10;
    if (n<1 || n>10000 || runs<1 || runs>10000) {
        std::cerr << "Usage: " << argv[0] << " [matrix_size 1..10000] [runs 1..10000]\n";
        return 1;
    }
    const Matrix input=make_random_matrix(n);
    double total=0.0, best=1e300, worst=0.0, last_checksum=0.0;
    for (int r=0; r<runs; ++r) {
        Matrix A=input;
        auto t0=std::chrono::steady_clock::now();
        eliminate(A);
        auto t1=std::chrono::steady_clock::now();
        last_checksum=checksum(A);
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
        total+=ms; best=std::min(best,ms); worst=std::max(worst,ms);
    }
    std::cout << std::fixed << std::setprecision(6)
              << label << "," << n << "," << runs << "," << total/runs
              << "," << best << "," << worst << "," << last_checksum << "\n";
    return 0;
}
