#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>

int main() {
	std::ofstream myfile;
    myfile.open ("data.csv");

	myfile << "imaginärdel av fasskift";
	myfile << ",";
	myfile << "Tlab [Mev]";
	myfile << "\n";

	std::vector<QuantumState> base = setupBase(0,2,0,2);
    std::map<std::string, std::vector<QuantumState> > channels = setupNNChannels(base);
	printChannels(channels);

	int N{ 100 };
	double scale{ 100 };

	TwoVectors k_and_w{ gaussLegendreInfMesh(N, scale) };
	std::vector<double> k{ k_and_w.v1 };
	std::vector<double> w{ k_and_w.v2 };

	std::string key = "j:0 s:0 tz:0 pi:1"; //could change key format
	std::vector<QuantumState> channel = channels[key]; 
	if (channel.size()==0) {
		std::cout << "Invalid key";
		abort();
	}
	printStates(channel);

	double Tlab = 0.0;

	for (int i = 1; i <= 350*5; i++)
	{
		Tlab = i/5.0;
	
	

		LapackMat V_matrix = potential(channel, k, Tlab);

		double k0 = get_k0(channel, Tlab);

		LapackMat T = computeTMatrix(channel, key, V_matrix, k, w, k0);
		//T.print();

		std::vector<std::complex<double>> phase = compute_phase_shifts(channel, key, k0, T);

		double realPart = phase[0].imag();
		myfile << realPart;
		myfile << ",";
		myfile << Tlab;
		myfile << "\n";
	}
	myfile.close();
	return 0;
}