#include <cblas.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>

void printMatrix(double *mat, int width, int height, int matAmt) {
    for (int matNum = 0; matNum < matAmt; matNum++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                std::cout << mat[row+col*height] << " ";
            }
            std::cout << ";" << std::endl;
        }
        std::cout << "\n";
    }
}


int main(int argc, char*argv[]) {
    cublasStatus_t stat;
    cudaError cudaStatus;
    cusolverStatus_t cusolverStatus;
    cublasHandle_t handle;

    int matWidth = 3;
    int matAmt   = 2;

    // Declare arrays on host
    double* A[matAmt];
    double* B[matAmt];

    double[] A1 = {2, -1, 1, 1, 1, 2, 1, -1, 3};
    double[] A2 = {4, -2, 2, 2, 2, 4, 2, -2, 6};

    double[] B1 = {2, 3, -10};
    double[] B2 = {4, 6, -20};

    A[0] = A1; A[1] = A2;
    B[0] = B1; B[1] = B2;

    // double *A, *B; // A - NxN matrix, B1 - auxiliary N-vect, B=A*B - N-vector of RHS, all on the host

    // Declare arrays on device
    double *d_A, *d_B;//, *d_Work; // Coeff matris, RHS, workspace
    int *d_pivot, *d_info;//, Lwork; // Pivots, info, workspace size
    int info_gpu = 0;

    // Prepare memory on host
    A  = (double*)malloc(matAmt*matWidth*matWidth*sizeof(double));
    B  = (double*)malloc(matAmt*matWidth*sizeof(double));

    for (int i = 0; i < matAmt*matWidth*matWidth; i++) {
        A[i] = rand()/(double)RAND_MAX;
    }

    for (int i = 0; i < matAmt; i++) {
        for (int j = 0; j < matWidth; j++) {
            B[i*matWidth+j] = A[i*matWidth*matWidth + j*matWidth];
        }
    }

    std::cout << "A:\n";
    printMatrix(A, matWidth, matWidth, matAmt);
    std::cout << "\nB:\n";
    printMatrix(B, 1, matWidth, matAmt);

    cudaStatus = cudaGetDevice(0);

    // Prepare memory on the device
    cudaStatus = cudaMalloc((void**)&d_A, matAmt*matWidth*matWidth*sizeof(double));
    cudaStatus = cudaMalloc((void**)&d_B,          matAmt*matWidth*sizeof(double));
    cudaStatus = cudaMalloc((void**)&d_pivot,      matAmt*matWidth*sizeof(int));
    cudaStatus = cudaMalloc((void**)&d_info,                matAmt*sizeof(int));

    cudaStatus = cudaMemcpy(d_A, A, matAmt*matWidth*matWidth*sizeof(double), cudaMemcpyHostToDevice); // Copy d_A <- A
    cudaStatus = cudaMemcpy(d_B, B,          matAmt*matWidth*sizeof(double), cudaMemcpyHostToDevice); // Copy d_B <- B

    // BATCHED?
    // cusolverStatus = cusolverDnSgetrf_bufferSize(handle, matWidth, matWidth, d_A, matWidth, &Lwork); // Compute buffer size and prepare memory

    // cudaStatus = cudaMalloc((void**)&d_Work, matAmt*Lwork*sizeof(double));

    stat = cublasDgetrfBatched(handle, matWidth, &d_A, matWidth, d_pivot, d_info, matAmt);
    stat = cublasDgetrsBatched(handle, CUBLAS_OP_N, matWidth, 1, &d_A, matWidth, d_pivot, &d_B, matWidth, d_info, matAmt);

    cudaStatus = cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost); // d_info -> info_gpu
    cudaStatus = cudaMemcpy(B, d_B, matWidth*sizeof(double), cudaMemcpyDeviceToHost); // d_B -> B

    printf("\nX:\n");
    printMatrix(B, 1, matWidth, matAmt);
    printf("\n");

    // Free memory
    cudaStatus = cudaFree(d_A);
    cudaStatus = cudaFree(d_B);
    cudaStatus = cudaFree(d_pivot);
    cudaStatus = cudaFree(d_info);
    // cudaStatus = cudaFree(d_Work);

    free(A); free(B);

    stat = cublasDestroy(handle);

    cudaStatus = cudaDeviceReset();

    return 0;
}