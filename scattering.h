#ifndef SCATTERING_H
#define SCATTERING_H

#include "constants.h"
#include "lapackAPI.h"
#include "quantumStates.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include <iomanip>
#include <cuda_runtime.h>
#include <cuComplex.h>


// constexpr std::complex<double> I(0.0, 1.0);


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
void computeTMatrix(cuDoubleComplex* T,
	cuDoubleComplex* V,
	cuDoubleComplex* G0,
	cuDoubleComplex* VG,
	cuDoubleComplex* F,
	double* k,
	double* w,
	double* k0,
	int quadratureN,
	int matSize,
	double mu,
	bool coupled);


//std::vector<std::complex<double>> blattToStapp(std::complex<double> deltaMinusBB, std::complex<double> deltaPlusBB, std::complex<double> twoEpsilonJBB);
//void computePhaseShifts(cuDoubleComplex* phases, double mu, bool coupled, double* k0, cuDoubleComplex* T, int Nkvadr);


#endif