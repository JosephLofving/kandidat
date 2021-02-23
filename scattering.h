#ifndef SCATTERING_H
#define SCATTERING_H

#include <complex>		// for complex numbers with std::complex<double>,
#include <tuple>
#include <cmath>
#include <vector>
#include <iostream>

std::vector<std::complex<double>> setup_G0_vector(std::vector<double> k, std::vector<double> w, double k0);
// setup_VG_kernel...
std::vector<double> blattToStapp(double deltaMinusBB, double deltaPlusBB, double twoEpsilonJBB);
// compute_T_matrix...
std::vector<double> compute_phase_shifts(std::vector<std::vector<double>> NN_channel, double k0, std::vector<double> T);



#endif