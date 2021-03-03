#ifndef SCATTERING_H
#define SCATTERING_H

#include <complex>		// for complex numbers with std::complex<double>,
#include <tuple>
#include <cmath>
#include <vector>
#include <iostream>

#include "constants.h"	// namespace constants
#include "quantumStates.h"
#include "lapackAPI.h"

double get_reduced_mass(std::vector<QuantumState> channel);

std::vector<std::complex<double>> setup_G0_vector(std::vector<double> k, std::vector<double> w, double k0);
LapackMat setup_VG_kernel(std::vector<QuantumState> channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0);
std::vector<std::complex<double>> blattToStapp(double deltaMinusBB, double deltaPlusBB, double twoEpsilonJBB);
LapackMat computeTMatrix(std::vector<QuantumState> NN_channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0);
std::vector<std::complex<double>> compute_phase_shifts(std::vector<QuantumState> NN_channel,std::string key, double k0, std::vector<double> T);


#endif