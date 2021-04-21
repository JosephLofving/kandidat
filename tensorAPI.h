#ifndef TENSORAPI_H
#define TENSORAPI_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>


void setElement(cuDoubleComplex* tensor, int row, int column, int slice, int matLength, cuDoubleComplex value);
cuDoubleComplex getElement(cuDoubleComplex* tensor, int row, int column, int slice, int matLength);

#endif