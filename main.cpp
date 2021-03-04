#include <iostream>
#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <stdio.h>
#include <fenv.h>


#include <iomanip> // can use std::cout << std::setprecision(16) to get 16 decimals


int main() {
	//feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	std::vector<QuantumState> base = setup_Base(0,2,0,2);
    std::map<std::string, std::vector<QuantumState> > channels = setup_NN_channels(base);
	printChannels(channels);

	int N{ 100 };
	double scale{ 100 };

	Two_vectors k_and_w{ gauss_legendre_inf_mesh(N, scale) };
	std::vector<double> k{ k_and_w.v1 };
	std::vector<double> w{ k_and_w.v2 };

	std::string key = "j:0 s:0 tz:0 pi:1"; //could change key format
	std::vector<QuantumState> channel = channels[key]; 
	printStates(channel);

	double Tlab = 100.0;

	LapackMat V_matrix = potential(channel, k, Tlab);

	double k0 = get_k0(channel, Tlab);

	LapackMat T = computeTMatrix(channel, key, V_matrix, k, w, k0);

	std::vector<std::complex<double>> phase = compute_phase_shifts(channel, key, k0, T);

	std::cout << phase[0] << std::endl;

	return 0;
}