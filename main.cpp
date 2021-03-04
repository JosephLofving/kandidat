#include <iostream>
#include "mesh.h"
#include "scattering.h"
#include "potential.h"

int main() {
	std::vector<QuantumState> base = setup_Base(0,2,0,2);
    std::map<std::string, std::vector<QuantumState> > channels = setup_NN_channels(base);
	printChannels(channels);

	int N{ 3 };
	double scale{ 100 };

	Two_vectors k_and_w{ gauss_legendre_inf_mesh(N, scale) };
	std::vector<double> k{ k_and_w.v1 };
	std::vector<double> w{ k_and_w.v2 };

	std::string key = "j:0 s:0 tz:0 pi:1"; //could change key format
	std::vector<QuantumState> channel = channels[key]; 
	printStates(channel);

	double Tlab = 100.0;

	LapackMat V_matrix = potential(channel, k, Tlab);

	//std::vector<std::complex<double>> V_matr_contents = {-7.12889305e-06, -6.86668257e-06, -5.79313917, -3.86252135e-06, -0.00000000e+00, -4.27445492e-06, -6.86668257e-06, -6.65532343e-06, -5.72545278e-06, -3.86154301e-06, -0.00000000e+00, -4.27148123e-06, -5.79313917e-06, -5.72545278e-06, -5.32659750e-06, -3.85347901e-06, -0.00000000e+00, -4.24747437e-06, -3.86252135e-06, -3.86154301e-06, -3.85347901e-06, -3.54673322e-06, -0.00000000e+00, -3.73075208e-06, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00, -4.27445492e-06, -4.27148123e-06, -4.24747437e-06, -3.73075208e-06, -0.00000000e+00, -3.99453309e-06};
	//LapackMat V_matrix = LapackMat(6, 6, V_matr_contents);

	// std::cout<<V_matrix.getElement(10,10);

	//V_matrix.print();





	double k0 = get_k0(channel, Tlab);
	LapackMat T = computeTMatrix(channel, key, V_matrix, k, w, k0);
	std::vector<std::complex<double>> phase = compute_phase_shifts(channel, key, k0, T);
	for (std::vector<std::complex<double>>::const_iterator i = phase.begin(); i != phase.end(); ++i) { //print(phase)
			std::cout << *i << ' ';
		}
	std::cout << std::endl;
	std::cout << "\nThe code ran successfully :)\n";

	return 0;
}