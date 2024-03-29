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
#include <cublas_v2.h>


__global__
void setupDVector(cuDoubleComplex* D,
	double* k,
	double* w,
	double* k0,
	double* sum,
	int quadratureN,
	int matLength,
	int TLabLength,
	double mu,
	bool coupled);

__global__
void setupVDKernel(cuDoubleComplex* T,
	cuDoubleComplex* VD,
	cuDoubleComplex* V,
	cuDoubleComplex* D,
	cuDoubleComplex* F,
	double* k,
	double* w,
	double* k0,
	int quadratureN,
	int matLength,
	int TLabLength,
	double mu,
	bool coupled);

__device__
void blattToStapp(cuDoubleComplex* phases,
	cuDoubleComplex deltaMinusBB,
	cuDoubleComplex deltaPlusBB,
	cuDoubleComplex twoEpsilonJBB,
	int TLabLength);

__global__
void computePhaseShifts(cuDoubleComplex* phases,
	cuDoubleComplex* T,
	double* k0,
	int quadratureN,
	double mu,
	bool coupled,
	int TLabLength,
	int matLength);


#endif