#include <iostream>
//#include "mesh.h" //mesh.cpp kanske funkar ocks�?
#include "quantumStates.h"
#include "chiral_LO.h"
#include "mesh.h"
#include "lapackAPI.h"


int main() {
	std::vector<QuantumState> base = setup_Base(0,2,0,0);
    std::multimap<std::string, QuantumState> channels = setup_NN_channels(base);
	int Np = 100;
	Two_vectors p_and_w = gauss_legendre_inf_mesh(Np);
	std::vector<double> p = p_and_w.v1;
	std::vector<double> w = p_and_w.v2;
	double Tlab = 100.0;
	channel chan = channels[0];
	Two_vectors V_and_k0 = potential(chan, p, Tlab);
	std::vector<double> V = V_and_k0.v1;
	std::vector<double> k0 = V_and_k0.v2;
	lapackMat T = compute_Tmatrix(chan, V, k0, p, w);
	std::vector<double> phase = compute_phase_shifts(chan, k0, T);
	for (std::vector<double>::const_iterator i = phase.begin(); i != phase.end(); ++i) //print(phase)
		std::cout << *i << ' ';

	
}