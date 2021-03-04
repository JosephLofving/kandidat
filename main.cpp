#include <iostream>
#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <stdio.h>
#include <fenv.h>


int main() {
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	std::vector<QuantumState> base = setup_Base(0,2,0,2);
    std::map<std::string, std::vector<QuantumState> > channels = setup_NN_channels(base);
	printChannels(channels);

	int N{ 4 };
	double scale{ 100 };

	Two_vectors k_and_w{ gauss_legendre_inf_mesh(N, scale) };
	std::vector<double> k{ k_and_w.v1 };
	std::vector<double> w{ k_and_w.v2 };

	std::string key = "j:0 s:0 tz:0 pi:1"; //could change key format
	std::vector<QuantumState> channel = channels[key]; 
	printStates(channel);

	double Tlab = 100.0;

	// LapackMat V_matrix = potential(channel, k, Tlab);

	std::vector<std::complex<double>> V_matr_contents = {-7.10257467e-06, -6.56342984e-06, -4.69313397e-06, -1.28344336e-36, -4.27431929e-06, -6.56342984e-06, -6.21496649e-06, -4.67102965e-06, -1.28233728e-36, -4.26723150e-06, -4.69313397e-06, -4.67102965e-06, -4.36659316e-06, -1.26846110e-36, -4.13719348e-06, -1.28344336e-36, -1.28233728e-36, -1.26846110e-36, -4.95908872e-67, -1.25177115e-36, -4.27431929e-06, -4.26723150e-06, -4.13719348e-06, -1.25177115e-36, -3.99453309e-06};
	LapackMat V_matrix = LapackMat(5, 5, V_matr_contents);

	// std::cout<<V_matrix.getElement(10,10);

	std::cout << "V_matrix" << std::endl;
	V_matrix.print();


	double k0 = get_k0(channel, Tlab);
	LapackMat T = computeTMatrix(channel, key, V_matrix, k, w, k0);
	std::vector<std::complex<double>> phase = compute_phase_shifts(channel, key, k0, T);
	for (std::vector<std::complex<double>>::const_iterator i = phase.begin(); i != phase.end(); ++i) { //print(phase)
			std::cout << *i << ' ';
		}
	std::cout << std::endl;
	std::cout << "\nThe code ran successfully :)\n";

	T.print();

	return 0;
}