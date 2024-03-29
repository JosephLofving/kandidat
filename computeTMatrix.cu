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

void computeTMatrixCUBLAS(cuDoubleComplex* T_d,
         			cuDoubleComplex* F_d,
		 			int matLength, int TLabLength,
                    cublasStatus_t status,
                    cublasHandle_t handle) {

    using microseconds = std::chrono::microseconds;

    // Host variables
    cuDoubleComplex** Fptr_array_h;
    cuDoubleComplex** Tptr_array_h;

    Fptr_array_h = (cuDoubleComplex**)malloc(TLabLength * sizeof(cuDoubleComplex*));
    Tptr_array_h = (cuDoubleComplex**)malloc(TLabLength * sizeof(cuDoubleComplex*));



    // Device variables
    cuDoubleComplex** Fptr_array_d;
    cuDoubleComplex** Tptr_array_d;

    int* pivotArray_d;
    int* trfInfo_d;
    int  trsInfo_d;

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("> ERROR: cuBLAS initialization failed\n");
    }



    // Allocate memory for device variables
    chkCudaErr(cudaMalloc((void**)&pivotArray_d, matLength * TLabLength * sizeof(int)));
    chkCudaErr(cudaMalloc((void**)&trfInfo_d, TLabLength * sizeof(int)));
    chkCudaErr(cudaMalloc((void**)&Fptr_array_d, TLabLength * sizeof(cuDoubleComplex*)));
    chkCudaErr(cudaMalloc((void**)&Tptr_array_d, TLabLength * sizeof(cuDoubleComplex*)));




    // Create pointer array for matrices
    for (int i = 0; i < TLabLength; i++) {
        Fptr_array_h[i] = F_d + (i * matLength * matLength);
        Tptr_array_h[i] = T_d + (i * matLength);
    }

    // Copy pointer array to device memory
    chkCudaErr(cudaMemcpy(Fptr_array_d, Fptr_array_h,
                               TLabLength * sizeof(cuDoubleComplex*),
							   cudaMemcpyHostToDevice));
    chkCudaErr(cudaMemcpy(Tptr_array_d, Tptr_array_h,
							   TLabLength * sizeof(cuDoubleComplex*),
							   cudaMemcpyHostToDevice));

    // Perform LU decomposition
    status = cublasZgetrfBatched(handle, matLength, Fptr_array_d, matLength, pivotArray_d,
								 trfInfo_d, TLabLength);

	// Calculate the T matrix
    status = cublasZgetrsBatched(handle, CUBLAS_OP_N, matLength, 1, Fptr_array_d,
                                matLength, pivotArray_d, Tptr_array_d, matLength, &trsInfo_d,
								TLabLength);

    // Copy data to host from device
    // chkCudaErr(cudaMemcpy(T_d, V_d, TLabLength*matLength*matLength *
    //                         sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));

    // Free device variables
    chkCudaErr(cudaFree(Fptr_array_d));
    chkCudaErr(cudaFree(Tptr_array_d));
    chkCudaErr(cudaFree(trfInfo_d));
    chkCudaErr(cudaFree(pivotArray_d));
    chkCudaErr(cudaFree(F_d));
    // chkCudaErr(cudaFree(V_d));

    // Destroy cuBLAS handle
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("> ERROR: cuBLAS uninitialization failed...\n");
    }

}
