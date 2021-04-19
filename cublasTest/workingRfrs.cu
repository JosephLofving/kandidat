// p ((@global double*)h_Aptr_array[0])[0]
// p ((@global double*)A)[1]

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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

// configurable parameters
// dimension of matrix
#define N 3
#define BATCH_SIZE 2

// use double precision data type
#define DOUBLE_PRECISION /* comment this to use single precision */
#ifdef DOUBLE_PRECISION
#define DATA_TYPE double
#else
#define DATA_TYPE float
#endif /* DOUBLE_PRCISION */

// wrapper around cublas<t>getrfBatched()
cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int n,
                                   DATA_TYPE* const A[], int lda, int* P,
                                   int* info, int batchSize) {
#ifdef DOUBLE_PRECISION
  return cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
  return cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
#endif
}

// wrapper around cublas<t>getrsBatched()
cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n,
                                   int nrhs, DATA_TYPE* const A[],
                                   int lda, const int* P, DATA_TYPE* B[],
                                   int ldb, int* info, int batchSize) {
#ifdef DOUBLE_PRECISION
  return cublasDgetrsBatched(handle, trans, n, nrhs, A, lda, P, B, ldb,
                             info, batchSize);
#else
  return cublasSgetrsBatched(handle, trans, n, nrhs, A, lda, P, B, ldb,
                             info, batchSize);
#endif
}

// wrapper around malloc
// clears the allocated memory to 0
// terminates the program if malloc fails
void* xmalloc(size_t size) {
  void* ptr = malloc(size);
  if (ptr == NULL) {
    printf("> ERROR: malloc for size %zu failed..\n", size);
    exit(EXIT_FAILURE);
  }
  memset(ptr, 0, size);
  return ptr;
}

void initSetAMatrix(DATA_TYPE* mat, DATA_TYPE factor) {
  DATA_TYPE toSet[N*N] = {2, -1, 1, 1, 1, 2, 1, -1, 3}; // A matrix that has a solution for the given B
  for (int i = 0; i < N*N; i++) {
    mat[i] = toSet[i]*factor; // Scale each element by the factor and set it
  }
}

void initSetBMatrix(DATA_TYPE* mat, DATA_TYPE factor) {
  DATA_TYPE toSet[N] = {2, 3, -10}; // B matrix that has a solution for the given A
  for (int i = 0; i < N; i++) {
    mat[i] = toSet[i]*factor; // Scale each element by the factor and set it
  }
}

// print column-major matrix
void printMatrix(DATA_TYPE* mat, int width, int height) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%6.3f ", mat[(j * height) + i]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char** argv) {
  // cuBLAS variables
  cublasStatus_t status;
  cublasHandle_t handle;

  // host variables
  size_t AmatSize = N * N * sizeof(DATA_TYPE);
  size_t BmatSize = N * sizeof(DATA_TYPE);

  DATA_TYPE* h_AarrayInput;
  DATA_TYPE* h_Aptr_array[BATCH_SIZE];

  DATA_TYPE* h_BarrayInput;
  DATA_TYPE* h_BarrayOutput;
  DATA_TYPE* h_Bptr_array[BATCH_SIZE];

  // device variables
  DATA_TYPE* d_Aarray;
  DATA_TYPE** d_Aptr_array;

  DATA_TYPE* d_Barray;
  DATA_TYPE** d_Bptr_array;

  int* d_pivotArray;
  int* d_AinfoArray;
  int d_Binfo;

  // seed the rand() function with time
  // srand(12345);

  // find cuda device
  printf("> Initializing...\n");
  // int dev = findCudaDevice(argc, (const char**)argv);
  // if (dev == -1) {
  //   return (EXIT_FAILURE);
  // }

  // initialize cuBLAS
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cuBLAS initialization failed\n");
    return (EXIT_FAILURE);
  }

#ifdef DOUBLE_PRECISION
  printf("> Using DOUBLE precision...\n");
#else
  printf("> Using SINGLE precision...\n");
#endif

  // allocate memory for host variables
  h_AarrayInput = (DATA_TYPE*)xmalloc(BATCH_SIZE * AmatSize);

  h_BarrayInput = (DATA_TYPE*)xmalloc(BATCH_SIZE * BmatSize);
  h_BarrayOutput = (DATA_TYPE*)xmalloc(BATCH_SIZE * BmatSize);

  // allocate memory for device variables
  checkCudaErrors(cudaMalloc((void**)&d_Aarray, BATCH_SIZE * AmatSize));
  checkCudaErrors(cudaMalloc((void**)&d_Barray, BATCH_SIZE * BmatSize));
  checkCudaErrors(
      cudaMalloc((void**)&d_pivotArray, N * BATCH_SIZE * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_AinfoArray, BATCH_SIZE * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_Binfo, sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void**)&d_Aptr_array, BATCH_SIZE * sizeof(DATA_TYPE*)));
  checkCudaErrors(
      cudaMalloc((void**)&d_Bptr_array, BATCH_SIZE * sizeof(DATA_TYPE*)));

  // fill matrix with random data
  printf("> Generating A matrices...\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    initSetAMatrix(h_AarrayInput + (i * N * N), (DATA_TYPE)(i+1)); // Create matrices scaled by factors 1, 2, ...
  }

  printf("> First A matrix:\n");
  printMatrix(h_AarrayInput, N, N);

  printf("> Generating B matrices...\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    initSetBMatrix(h_BarrayInput + (i * N), (DATA_TYPE)(i+1)); // Create matrices scaled by factors 1, 2, ...
  }

  printf("> First B matrix:\n");
  printMatrix(h_BarrayInput, 1, N);

  // copy data to device from host
  printf("> Copying data from host memory to GPU memory...\n");
  checkCudaErrors(cudaMemcpy(d_Aarray, h_AarrayInput, BATCH_SIZE * AmatSize,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Barray, h_BarrayInput, BATCH_SIZE * BmatSize,
                             cudaMemcpyHostToDevice));

  // create pointer array for matrices
  for (int i = 0; i < BATCH_SIZE; i++) h_Aptr_array[i] = d_Aarray + (i * N * N);
  for (int i = 0; i < BATCH_SIZE; i++) h_Bptr_array[i] = d_Barray + (i * N);

  // copy pointer array to device memory
  checkCudaErrors(cudaMemcpy(d_Aptr_array, h_Aptr_array,
                             BATCH_SIZE * sizeof(DATA_TYPE*),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Bptr_array, h_Bptr_array,
                             BATCH_SIZE * sizeof(DATA_TYPE*),
                             cudaMemcpyHostToDevice));

  // perform LU decomposition
  printf("> Performing LU decomposition...\n");

  status = cublasXgetrfBatched(handle, N, d_Aptr_array, N, d_pivotArray,
                               d_AinfoArray, BATCH_SIZE);

  printf("> Calculating X matrix...\n");

  status = cublasXgetrsBatched(handle, CUBLAS_OP_N, N, 1, d_Aptr_array, N,
                               d_pivotArray, d_Bptr_array, N, &d_Binfo,
                               BATCH_SIZE);

  // copy data to host from device
  printf("> Copying data from GPU memory to host memory...\n");
  checkCudaErrors(cudaMemcpy(h_BarrayOutput, d_Barray, BATCH_SIZE * BmatSize,
                             cudaMemcpyDeviceToHost));

  printf("> First X matrix:\n");
  printMatrix(h_BarrayOutput, 1, N);

  // free device variables
  checkCudaErrors(cudaFree(d_Aptr_array));
  checkCudaErrors(cudaFree(d_Bptr_array));
  checkCudaErrors(cudaFree(d_AinfoArray));
  checkCudaErrors(cudaFree(&d_Binfo));
  checkCudaErrors(cudaFree(d_pivotArray));
  checkCudaErrors(cudaFree(d_Aarray));
  checkCudaErrors(cudaFree(d_Barray));

  // free host variables
  if (h_BarrayOutput) free(h_BarrayOutput);
  if (h_AarrayInput) free(h_AarrayInput);
  if (h_BarrayInput) free(h_AarrayInput);

  // destroy cuBLAS handle
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cuBLAS uninitialization failed...\n");
    return (EXIT_FAILURE);
  }

  return (EXIT_SUCCESS);
}
