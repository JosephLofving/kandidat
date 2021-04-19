#include "computeTMatrix.h"

template <typename T>
void check(T result, char const *const func, const char *const file,
                     int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                        static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

// Configurable parameters
// Dimension of matrix
// #define N 4
// #define batchSize 2

// Wrapper around malloc
// Clears the allocated memory to 0
// Terminates the program if malloc fails
// void* xmalloc(size_t size) {
//     void* ptr = malloc(size);
//     if (ptr == NULL) {
//         printf("> ERROR: malloc for size %zu failed..\n", size);
//         exit(EXIT_FAILURE);
//     }
//     memset(ptr, 0, size);
//     return ptr;
// }

// void initSetAMatrix(cuDoubleComplex* mat, double factor) {
// 	double toSet[N*N] = {4, 3, 9, 3, 7, 7, 0, 5, 8, 6, 1, 8, 9, 4, 2, 9};
//     for (int i = 0; i < N*N; i++) {
//         mat[i] = make_cuDoubleComplex(toSet[i]*factor, 0); // Scale each element by the factor and set it
//     }
// }

// void initSetBMatrix(cuDoubleComplex* mat, double factor) {
// 	double toSet[N*N] = {53, 36, 16, 50, 74, 59, 33, 67, 78, 69, 47, 61, 191, 148, 79, 166};
//     for (int i = 0; i < N*N; i++) {
//         mat[i] = make_cuDoubleComplex(toSet[i]*factor, 0); // Scale each element by the factor and set it
//     }
// }

// Print column-major matrix
// void printMatrix(cuDoubleComplex* mat, int width, int height) {
//     for (int i = 0; i < height; i++) {
//         for (int j = 0; j < width; j++) {
//             printf("%6.3f ", cuCreal(mat[(j * height) + i]));
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

__device__
void computeTMatrixCUBLAS(cuDoubleComplex* h_Tarray,
         			cuDoubleComplex* h_Farray,
		 			cuDoubleComplex* h_Varray,
		 			int N, int mSize) {

    const int batchSize{ 1 };
    if (true) {
        const int batchSize = mSize;
    }

	// cuBLAS variables
    cublasStatus_t status;
    cublasHandle_t handle;

    // Host variables
    

    // cuDoubleComplex* h_Farray;
    cuDoubleComplex* h_Fptr_array[batchSize];

    // cuDoubleComplex* h_Varray;
    cuDoubleComplex* h_Vptr_array[batchSize];

    // Device variables
    cuDoubleComplex* d_Farray;
    cuDoubleComplex** d_Fptr_array;

    cuDoubleComplex* d_Varray;
    cuDoubleComplex** d_Vptr_array;

    int* d_pivotArray;
    int* d_trfInfo;
    int d_trsInfo;

    // Initialize cuBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("> ERROR: cuBLAS initialization failed\n");
        return (EXIT_FAILURE);
    }

    // Allocate memory for host variables
    // h_Farray = (cuDoubleComplex*)xmalloc(batchSize * matSize);
    // h_Varray = (cuDoubleComplex*)xmalloc(batchSize * matSize);

    // Allocate memory for device variables
    checkCudaErrors(cudaMalloc((void**)&d_Farray, batchSize * matSize));
    checkCudaErrors(cudaMalloc((void**)&d_Varray, batchSize * matSize));
    checkCudaErrors(
            cudaMalloc((void**)&d_pivotArray, N * batchSize * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_trfInfo, batchSize * sizeof(int)));
    checkCudaErrors(
            cudaMalloc((void**)&d_Fptr_array, batchSize * sizeof(cuDoubleComplex*)));
    checkCudaErrors(
            cudaMalloc((void**)&d_Vptr_array, batchSize * sizeof(cuDoubleComplex*)));

    // for (int i = 0; i < batchSize; i++) {
    //     initSetAMatrix(h_Farray + (i * N*N), (double)(i+1)); // Create matrices scaled by factors 1, 2, ...
    // }

    // printMatrix(h_Farray, N, N);

    // for (int i = 0; i < batchSize; i++) {
    //     initSetBMatrix(h_Varray + (i * N*N), (double)(i+1)); // Create matrices scaled by factors 1, 2, ...
    // }

    // printMatrix(h_Varray, N, N);

    // Copy data to device from host
    checkCudaErrors(cudaMemcpy(d_Farray, h_Farray, batchSize * matSize,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Varray, h_Varray, batchSize * matSize,
                               cudaMemcpyHostToDevice));

    // Create pointer array for matrices
    for (int i = 0; i < batchSize; i++) h_Fptr_array[i] = d_Farray + (i * N*N);
    for (int i = 0; i < batchSize; i++) h_Vptr_array[i] = d_Varray + (i * N*N);

    // Copy pointer array to device memory
    checkCudaErrors(cudaMemcpy(d_Fptr_array, h_Fptr_array,
                               batchSize * sizeof(cuDoubleComplex*),
							   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Vptr_array, h_Vptr_array,
							   batchSize * sizeof(cuDoubleComplex*),
							   cudaMemcpyHostToDevice));

    // Perform LU decomposition
    status = cublasZgetrfBatched(handle, N, d_Fptr_array, N, d_pivotArray,
								 d_trfInfo, batchSize);

	// Calculate the T matrix
    status = cublasZgetrsBatched(handle, CUBLAS_OP_N, N, N, d_Fptr_array, N,
                                 d_pivotArray, d_Vptr_array, N, &d_trsInfo,
								 batchSize);

    // Copy data to host from device
    checkCudaErrors(cudaMemcpy(h_Tarray, d_Varray, batchSize * matSize,
                               cudaMemcpyDeviceToHost));

    // printMatrix(h_Varray, N, N);

    // Free device variables
    checkCudaErrors(cudaFree(d_Fptr_array));
    checkCudaErrors(cudaFree(d_Vptr_array));
    checkCudaErrors(cudaFree(d_trfInfo));
    checkCudaErrors(cudaFree(d_pivotArray));
    checkCudaErrors(cudaFree(d_Farray));
    checkCudaErrors(cudaFree(d_Varray));

    // Free host variables
    // if (h_Farray) free(h_Farray);
    // if (h_Varray) free(h_Varray);

    // Destroy cuBLAS handle
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("> ERROR: cuBLAS uninitialization failed...\n");
    }
}
