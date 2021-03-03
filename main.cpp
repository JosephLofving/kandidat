#include <iostream>
#include "mesh.h"
#include "scattering.h"
#include "potential.h"


int main() {
	std::vector<QuantumState> base = setup_Base(0,2,0,0);
    std::map<std::string, std::vector<QuantumState> > channels = setup_NN_channels(base);
	printChannels(channels);

	int N{ 100 };
	double scale{ 100 };

	Two_vectors k_and_w{ gauss_legendre_inf_mesh(N, scale) };
	std::vector<double> k{ k_and_w.v1 };


	std::string key = "j:0 s:0 tz:0 pi:1"; //could change key format
	std::vector<QuantumState> channel = channels[key]; 
	printStates(channel);

	double Tlab = 100.0;

	LapackMat V_and_k0 = potential(channel, k, Tlab);

	V_and_k0.print();



	//std::vector<double> V = V_and_k0.v1;
	//std::vector<double> k0 = V_and_k0.v2;
	//LapackMat T = compute_Tmatrix(channel, V, k0, p, w);
	//std::vector<double> phase = compute_phase_shifts(channel,key, k0, T);
	//for (std::vector<double>::const_iterator i = phase.begin(); i != phase.end(); ++i) //print(phase)
		//std::cout << *i << ' ';

	return 0;
}