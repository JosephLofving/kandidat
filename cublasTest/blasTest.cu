#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define DATATYPE float

// Fill values using curand
void init_vals(DATATYPE *in, int N) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, 1234ULL);
    curandGenerateUniform(prng, in, N);
    curandDestroyGenerator(prng);
}

// cuBLAS call
void cublas_matmul(const DATATYPE *A, const DATATYPE *B, DATATYPE *C,
    const int m, const int n, const int k) {

    int lda = m; int ldb = k; int ldc = m;
    const DATATYPE alpha = 1;
    const DATATYPE beta = 0;

    // STEP 1: Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // STEP 2: Call cuBLAS command
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        &alpha, A, lda, B, ldb, &beta, C, ldc);

    // STEP 3: Destroy handle
    cublasDestroy(handle);
}

int main() {
    // Create device vectors (memory allocation)
    int MATRIX_M = 16;
    int MATRIX_N = 16;
    int MATRIX_K = 16;

    thrust::device_vector<DATATYPE> d_A(MATRIX_M * MATRIX_K);
    thrust::device_vector<DATATYPE> d_B(MATRIX_K * MATRIX_N);
    thrust::device_vector<DATATYPE> d_C(MATRIX_M * MATRIX_N);

    // Initialize values using cuRAND
    init_vals(thrust::raw_pointer_cast(d_A.data()), MATRIX_M * MATRIX_K);
    init_vals(thrust::raw_pointer_cast(d_B.data()), MATRIX_K * MATRIX_N);

    // Call cuBLAS
    cublas_matmul(thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()),
        MATRIX_M, MATRIX_N, MATRIX_K);

    // Copy back results to host
    thrust::host_vector<DATATYPE> h_A(MATRIX_M * MATRIX_K);
    thrust::host_vector<DATATYPE> h_B(MATRIX_K * MATRIX_N);
    thrust::host_vector<DATATYPE> h_C_computed(MATRIX_M * MATRIX_N);
    h_A = d_A;
    h_B = d_B;
    h_C_computed = d_C;

    for (int row = 0; row < MATRIX_M; row++) {
        for (int col = 0; col < MATRIX_K; col++) {
            std::cout << h_A[row+col*MATRIX_M] << " ";
        }
        std::cout << ";" << std::endl;
    }
    std::cout << "\n\n";

    for (int row = 0; row < MATRIX_K; row++) {
        for (int col = 0; col < MATRIX_N; col++) {
            std::cout << h_B[row+col*MATRIX_K] << " ";
        }
        std::cout << ";" << std::endl;
    }
    std::cout << "\n\n";

    for (int row = 0; row < MATRIX_M; row++) {
        for (int col = 0; col < MATRIX_N; col++) {
            std::cout << h_C_computed[row+col*MATRIX_M] << " ";
        }
        std::cout << ";" << std::endl;
    }

    // for (int i = 0; i < MATRIX_M*MATRIX_N; i++) {
    //     std::cout << h_C_computed[i] << " ";
    // }

    std::cout << std::endl;
}