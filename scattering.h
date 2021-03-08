#ifndef SCATTERING_H
#define SCATTERING_H

#include "constants.h"
#include "lapackAPI.h"
#include "quantumStates.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>


constexpr std::complex<double> I(0.0, 1.0);

double get_reduced_mass(std::vector<QuantumState> channel);
bool isCoupled(std::vector<QuantumState> channel);
std::vector<std::complex<double>> setup_G0_vector(std::vector<QuantumState> NN_channe, std::vector<double> k, std::vector<double> w, double k0);
LapackMat setup_VG_kernel(std::vector<QuantumState> channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0);
std::vector<std::complex<double>> blattToStapp(std::complex<double> deltaMinusBB, std::complex<double> deltaPlusBB, std::complex<double> twoEpsilonJBB);
LapackMat computeTMatrix(std::vector<QuantumState> channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0);
std::vector<std::complex<double>> compute_phase_shifts(std::vector<QuantumState> channel,std::string key, double k0, LapackMat T);


#endif