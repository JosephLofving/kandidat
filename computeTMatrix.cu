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

#define chkCudaErr(val) check((val), #val, __FILE__, __LINE__)

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

void computeTMatrixCUBLAS(cuDoubleComplex* d_Tarray,
         			cuDoubleComplex* d_Farray,
		 			cuDoubleComplex* d_Varray,
		 			int matLength, int TLabLength) {

    int batchSize = TLabLength;
	// cuBLAS variables
    cublasStatus_t status;
    cublasHandle_t handle;

    // Host variables
    cuDoubleComplex** h_Fptr_array;
    cuDoubleComplex** h_Vptr_array;

    h_Fptr_array = (cuDoubleComplex**)malloc(batchSize * sizeof(cuDoubleComplex*));
    h_Vptr_array = (cuDoubleComplex**)malloc(batchSize * sizeof(cuDoubleComplex*));

    // Device variables
    cuDoubleComplex** d_Fptr_array;
    cuDoubleComplex** d_Vptr_array;

    int* d_pivotArray;
    int* d_trfInfo;
    int  d_trsInfo;

    // Initialize cuBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("> ERROR: cuBLAS initialization failed\n");
    }

    // Allocate memory for device variables
    chkCudaErr(cudaMalloc((void**)&d_pivotArray, matLength * TLabLength * sizeof(int)));
    chkCudaErr(cudaMalloc((void**)&d_trfInfo, TLabLength * sizeof(int)));
    chkCudaErr(cudaMalloc((void**)&d_Fptr_array, TLabLength * sizeof(cuDoubleComplex*)));
    chkCudaErr(cudaMalloc((void**)&d_Vptr_array, TLabLength * sizeof(cuDoubleComplex*)));

    // Create pointer array for matrices
    for (int i = 0; i < TLabLength; i++) h_Fptr_array[i] = d_Farray + (i * matLength * matLength);
    for (int i = 0; i < TLabLength; i++) h_Vptr_array[i] = d_Varray + (i * matLength * matLength);

    // Copy pointer array to device memory
    chkCudaErr(cudaMemcpy(d_Fptr_array, h_Fptr_array,
                               TLabLength * sizeof(cuDoubleComplex*),
							   cudaMemcpyHostToDevice));
    chkCudaErr(cudaMemcpy(d_Vptr_array, h_Vptr_array,
							   TLabLength * sizeof(cuDoubleComplex*),
							   cudaMemcpyHostToDevice));

    // Perform LU decomposition
    status = cublasZgetrfBatched(handle, matLength, d_Fptr_array, matLength, d_pivotArray,
								 d_trfInfo, batchSize);

	// Calculate the T matrix
    status = cublasZgetrsBatched(handle, CUBLAS_OP_N, matLength, matLength, d_Fptr_array,
                                matLength, d_pivotArray, d_Vptr_array, matLength, &d_trsInfo,
								batchSize);

    // Copy data to host from device
    chkCudaErr(cudaMemcpy(d_Tarray, d_Varray, batchSize*matLength*matLength *
                            sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    // Free device variables
    chkCudaErr(cudaFree(d_Fptr_array));
    chkCudaErr(cudaFree(d_Vptr_array));
    chkCudaErr(cudaFree(d_trfInfo));
    chkCudaErr(cudaFree(d_pivotArray));
    chkCudaErr(cudaFree(d_Farray));
    chkCudaErr(cudaFree(d_Varray));

    // Destroy cuBLAS handle
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("> ERROR: cuBLAS uninitialization failed...\n");
    }
}
