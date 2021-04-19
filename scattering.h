#ifndef SCATTERING_H
#define SCATTERING_H

#include "constants.h"
#include "lapackAPI.h"
//#include "computeTMatrix.h"
#include "quantumStates.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include <iomanip>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cuComplex.h>

template <typename T>
void check(T result, char const* const func, const char* const file, int const line);

// #define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
static const char* _cudaGetErrorEnum(cudaError_t error);

__device__
void computeTMatrixCUBLAS(cuDoubleComplex* h_Tarray, cuDoubleComplex* h_Farray, cuDoubleComplex* h_Varray, int N, int batchSize);

//#endif


__device__
void setupG0Vector(cuDoubleComplex* D,
	double* k,
	double* w,
	double k0,
	int quadratureN,
	double mu,
	bool coupled);

__device__
void setupVGKernel(cuDoubleComplex* VG,
	cuDoubleComplex* V,
	cuDoubleComplex* G0,
	cuDoubleComplex* F,
	double* k,
	double* w,
	double k0,
	int quadratureN,
	int matSize,
	double mu,
	bool coupled);


__global__
void computeTMatrix(cuDoubleComplex** T,
	cuDoubleComplex** V,
	cuDoubleComplex** G0,
	cuDoubleComplex** VG,
	cuDoubleComplex** F,
	cuDoubleComplex** phases,
	double* k,
	double* w,
	double* k0,
	int quadratureN,
	int matSize,
	int TLabLength,
	double mu,
	bool coupled);

__device__
void blattToStapp(cuDoubleComplex* phases, cuDoubleComplex deltaMinusBB, cuDoubleComplex deltaPlusBB, cuDoubleComplex twoEpsilonJBB);

__device__
void computePhaseShifts(cuDoubleComplex* phases,
	cuDoubleComplex* T,
	double k0,
	int quadratureN,
	double mu,
	bool coupled);


#endif