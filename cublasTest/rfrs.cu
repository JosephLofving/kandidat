// set cuda api_failures stop
// print *((@global double *)d_B)[0]


#include <cblas.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <csignal>

#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);\
      exit(-1);}} while(0)

void printMatrix(double *mat, int width, int height, int matAmt) {
    for (int matNum = 0; matNum < matAmt; matNum++) {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                std::cout << mat[row+col*height] << " ";
            }
            std::cout << ";" << std::endl;
        }
    std::cout << std::endl;
    }
}


int main(int argc, char*argv[]) {
    cublasStatus_t stat;

    cudaError cudaStatus;
    cusolverStatus_t cusolverStatus;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    // cudaStream_t streamArray;
    // cudaStreamCreate(&streamArray);
    // cublasSetStream(handle, streamArray);

    int matWidth = 3;
    int matAmt   = 2;

    // Declare arrays on host
    // double **h_A[matAmt]; double **h_B[matAmt];


    double h_A[matWidth*matWidth*matAmt] = {2, -1, 1, 1, 1, 2, 1, -1, 3,
                                            4, -2, 2, 2, 2, 4, 2, -2, 6};

    double h_B[matAmt*matWidth] = {2, 3, -10,
                                   4, 6, -20};


    // double *A, *B; // A - NxN matrix, B1 - auxiliary N-vect, B=A*B - N-vector of RHS, all on the host

    // Declare arrays on device
    // double **d_A, **d_B;//, *d_Work; // Coeff matris, RHS, workspace
    double *d_A, *d_B;
    int *d_pivot, *d_info;//, Lwork; // Pivots, info, workspace size
    int info_gpu = 0;

    cudaStatus = cudaMalloc((void**)&d_A, matWidth*matWidth*matAmt*sizeof(double));
    cudaStatus = cudaMalloc((void**)&d_B,          matWidth*matAmt*sizeof(double));

    double **h_A_pArr = new double *[matAmt];
    double **h_B_pArr = new double *[matAmt];

    double test = 45.0;

    for (int i = 0; i < matAmt; i++) {
        cudaStatus = cudaMemcpy(&d_A[i*matWidth*matWidth], &h_A[i*matWidth*matWidth], matWidth*matWidth*sizeof(double), cudaMemcpyHostToDevice);
        h_A_pArr[i] = &test;
        // h_A_pArr[i] = &d_A[i*matWidth*matWidth];
    }

    for (int i = 0; i < matAmt; i++) {
        cudaStatus = cudaMemcpy(&d_B[i*matWidth], &h_B[i*matWidth], matWidth*sizeof(double), cudaMemcpyHostToDevice);
        h_B_pArr[i] = &d_B[i*matWidth];
    }

    double **d_A_pArr;
    double **d_B_pArr;

    cudaStatus = cudaMalloc((void**)&d_A_pArr, matAmt*sizeof(double *));
    cudaStatus = cudaMalloc((void**)&d_B_pArr, matAmt*sizeof(double *));


    cudaStatus  = cudaMemcpy(d_A_pArr, h_A_pArr, matAmt*sizeof(double *), cudaMemcpyHostToDevice);
    // cudaStatus  = cudaMemcpy(d_A_pArr[1], h_A_pArr[1], matAmt*sizeof(double *), cudaMemcpyHostToDevice);
    cudaStatus  = cudaMemcpy(d_B_pArr, h_B_pArr, matAmt*sizeof(double *), cudaMemcpyHostToDevice);
    // cudaStatus  = cudaMemcpy(d_B_pArr[1], h_B_pArr[1], matAmt*sizeof(double *), cudaMemcpyHostToDevice);

    // Prepare memory on host
    // A  = (double*)malloc(matAmt*matWidth*matWidth*sizeof(double));
    // B  = (double*)malloc(matAmt*matWidth*sizeof(double));

    // for (int i = 0; i < matAmt*matWidth*matWidth; i++) {
    //     A[i] = rand()/(double)RAND_MAX;
    // }

    // for (int i = 0; i < matAmt; i++) {
    //     for (int j = 0; j < matWidth; j++) {
    //         B[i*matWidth+j] = A[i*matWidth*matWidth + j*matWidth];
    //     }
    // }

    // std::cout << "A:\n";
    // printMatrix(h_A1, matWidth, matWidth);
    // std::cout << std::endl;
    // printMatrix(h_A2, matWidth, matWidth);
    // std::cout << "\nB:\n";
    // printMatrix(h_B1, 1, matWidth);
    // std::cout << std::endl;
    // printMatrix(h_B2, 1, matWidth);

                //cudaStatus = cudaGetDevice(0);

    // Prepare memory on the device
    // cudaStatus = cudaMalloc((void**)&d_A1, matWidth*matWidth*sizeof(double));
    // cudaStatus = cudaMalloc((void**)&d_A2, matWidth*matWidth*sizeof(double));
    // cudaStatus = cudaMalloc((void**)&d_B1,          matWidth*sizeof(double));
    // cudaStatus = cudaMalloc((void**)&d_B2,          matWidth*sizeof(double));
    cudaStatus = cudaMalloc((void**)&d_pivot,   matAmt*matWidth*sizeof(int));
    cudaStatus = cudaMalloc((void**)&d_info,                    sizeof(int));

    // cudaStatus = cudaMemcpy(d_A1, h_A1, matWidth*matWidth*sizeof(double), cudaMemcpyHostToDevice); // Copy d_A <- A
    // cudaStatus = cudaMemcpy(d_A2, h_A2, matWidth*matWidth*sizeof(double), cudaMemcpyHostToDevice); // Copy d_A <- A
    // cudaStatus = cudaMemcpy(d_B1, h_B1,          matWidth*sizeof(double), cudaMemcpyHostToDevice); // Copy d_B <- B
    // cudaStatus = cudaMemcpy(d_B2, h_B2,          matWidth*sizeof(double), cudaMemcpyHostToDevice); // Copy d_B <- B

    // double **h_A = new double *[matAmt];
    // double **h_B = new double *[matAmt];

    // double *d_A; double *d_B;

    // cudaStatus = cudaMalloc((void**)&d_A, matWidth*matWidth)

    // double *h_A[matAmt]; double *h_B[matAmt];
    // double *d_A[matAmt]; double *d_B[matAmt];

    // h_A[0] = d_A1; h_A[1] = d_A2;
    // h_B[0] = d_B1; h_B[1] = d_B2;

    // cudaStatus = cudaMalloc((void**)&d_A, matAmt*sizeof(double*));
    // cudaStatus = cudaMalloc((void**)&d_B, matAmt*sizeof(double*));

    // cudaStatus = cudaMemcpy(d_A, h_A, matAmt*sizeof(double*), cudaMemcpyHostToDevice);
    // cudaStatus = cudaMemcpy(d_B, h_B, matAmt*sizeof(double*), cudaMemcpyHostToDevice);

    // BATCHED?
    // cusolverStatus = cusolverDnSgetrf_bufferSize(handle, matWidth, matWidth, d_A, matWidth, &Lwork); // Compute buffer size and prepare memory

    // cudaStatus = cudaMalloc((void**)&d_Work, matAmt*Lwork*sizeof(double));

    // int *h_info;

    CUBLAS_CALL( cublasDgetrfBatched(handle, matWidth, d_A_pArr, matWidth, d_pivot, d_info, matAmt) ); // h_A -> d_A?
    CUBLAS_CALL( cublasDgetrsBatched(handle, CUBLAS_OP_N, matWidth, 1, d_A_pArr, matWidth, d_pivot, d_B_pArr, matWidth, d_info, matAmt) );

    // stat = cublasDgetrfBatched(handle, matWidth, h_A, matWidth, d_pivot, d_info, matAmt); // h_A -> d_A?
    // stat = cublasDgetrsBatched(handle, CUBLAS_OP_N, matWidth, 1, h_A, matWidth, d_pivot, h_B, matWidth, d_info, matAmt);

    cudaStatus = cudaDeviceSynchronize();

    // cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost); // d_info -> info_gpu
    cudaStatus = cudaMemcpy(h_B, d_B, matWidth*sizeof(double), cudaMemcpyDeviceToHost); // d_B -> B

    printMatrix(h_B, 1, matWidth, 2);

    // printf("\nX:\n");
    // printMatrix(h_B1, 1, matWidth);
    // printf("\n");
    // printMatrix(h_B2, 1, matWidth);
    // printf("\n");

                // 0x555559d78748 <_Z14getrf_semiwarpIddLi2ELi3ELb1EEviPKPT_iPiS4_i+520>:  0x00010c0c

    // Free memory
    cudaStatus = cudaFree(d_A);
    cudaStatus = cudaFree(d_A_pArr);
    cudaStatus = cudaFree(d_B);
    cudaStatus = cudaFree(d_B_pArr);
    cudaStatus = cudaFree(d_pivot);
    cudaStatus = cudaFree(d_info);
    // cudaStatus = cudaFree(d_Work);

    free(h_A); free(h_B);

    stat = cublasDestroy(handle);

    cudaStatus = cudaDeviceReset();

    return 0;
}