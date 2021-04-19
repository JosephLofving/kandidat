#ifndef TMATR_H
#define TMATR_H

#include <stdio.h>  // Onödig?
#include <stdlib.h> // Onödig?
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)

// #define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
static const char *_cudaGetErrorEnum(cudaError_t error)

void computeTMatrixCUBLAS(cuDoubleComplex* h_Tarray, cuDoubleComplex* h_Farray, cuDoubleComplex* h_Varray, int N, int batchSize)

#endif