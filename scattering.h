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


constexpr std::complex<double> I(0.0, 1.0);

double getReducedMass(std::vector<QuantumState> channel);
bool isCoupled(std::vector<QuantumState> channel);
cuDoubleComplex* setupG0Vector(double mu, double* k, double* w, double k0, int Nkvadr);
void setupVGKernel(cuDoubleComplex* VG, double mu, bool coupled, cuDoubleComplex** V, double* k, double* w, double k0, int Nkvadr, int G0Size);
cuDoubleComplex* computeTMatrix(cuDoubleComplex** V_matrix, double* k, double* w, double* k0, int Nkvadr, int G0Size, double mu, bool coupled);
std::vector<std::complex<double>> blattToStapp(std::complex<double> deltaMinusBB, std::complex<double> deltaPlusBB, std::complex<double> twoEpsilonJBB);
void computePhaseShifts(cuDoubleComplex* phases, double mu, bool coupled, double* k0, cuDoubleComplex* T, int Nkvadr);


#endif