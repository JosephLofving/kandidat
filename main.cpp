#include <iostream>
//#include "mesh.h"
//#include "scattering.h"
#include "potential.h"


int main() {
	std::vector<QuantumState> base = setup_Base(0,2,0,0);
    std::map<std::string, std::vector<QuantumState> > channels = setup_NN_channels(base);
	int Np = 100;
	//Two_vectors p_and_w = gauss_legendre_inf_mesh(Np);
	//std::vector<double> p = p_and_w.v1;
	//std::vector<double> w = p_and_w.v2;
	std::vector<double> p;
	p.push_back(1.2);
	p.push_back(1.4);
	p.push_back(1.5);
	p.push_back(1.6);
	p.push_back(1.7);
	//double Tlab = 100.0;
	std::string key = "j:0 s:0 tz:0 pi:0"; //could change key format
	std::vector<QuantumState> channel = channels[key]; 
	LapackMat V_and_k0 = potential(channel, p);



	//std::vector<double> V = V_and_k0.v1;
	//std::vector<double> k0 = V_and_k0.v2;
	//LapackMat T = compute_Tmatrix(channel, V, k0, p, w);
	//std::vector<double> phase = compute_phase_shifts(channel,key, k0, T);
	//for (std::vector<double>::const_iterator i = phase.begin(); i != phase.end(); ++i) //print(phase)
		//std::cout << *i << ' ';

	return 0;
}