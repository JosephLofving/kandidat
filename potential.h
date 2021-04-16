#ifndef POTENTIAL_H
#define POTENTIAL_H

#include "chiral_LO.h"
#include "constants.h"
#include "lapackAPI.h"
#include "quantumStates.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>

void getk0(double* k0, int tzChannel, double* TLab, int TLabLength);
cuDoubleComplex* potential(std::vector<QuantumState> channel, double* k, double Tlab, double k0, int NKvadratur);

#endif