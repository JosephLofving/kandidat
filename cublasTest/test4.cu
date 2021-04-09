#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N (1290)

void MatrixMulOnHost(float* M, float* N, float* P, int Width) {
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            double sum = 0;
            for (int k = 0; k < Width; ++k) {
                double a = M[i * Width + k];
                double b = N[k * width + j];
                sum += a*b;
            }
            P[i * Width + j] = sum;
        }
    }
}

__global__
void KernelFunc(...) {

}

int main() {
    dim3 DimGrid(100, 50);
    dim3 DimBlock(4, 8, 8);
    KernelFunc<<<DimGrid, DimBlock>>>(...);

    return 0;
}