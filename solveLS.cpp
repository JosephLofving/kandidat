#include <iostream>
//#include "mesh.h" //mesh.cpp kanske funkar ocks�?
#include "quantumStates.h"
#include "chiral_LO.h"


int main() {
	std::vector<QuantumState> base = setup_Base(0,2,0,0);
    std::multimap<std::string, QuantumState> channels = setup_NN_channels(base);

	int Np = 100;
	//tuple a = gauss_kegendre_inf_mesh(Np);
	double Tlab = 100.0;

	//tuple b = 
	
}