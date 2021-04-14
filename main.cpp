#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>
#include <iomanip>

int main() {
	std::ofstream myfile;
    myfile.open ("data.csv");

	myfile << "Real av fasskift";
	myfile << ",";
	myfile << "N";
	myfile << "\n";

	std::vector<QuantumState> base = setupBase(0, 2, 0, 2);
    std::map<std::string, std::vector<QuantumState> > channels = setupNNChannels(base);
	printChannels(channels);

	int N = 100;
	double scale = 100;


	std::string key = "j:0 s:0 tz:0 pi:1"; //could change key format
	std::vector<QuantumState> channel = channels[key];
	if (channel.size()==0) {
		std::cout << "Invalid key";
		abort();
	}
	printStates(channel);

	double Tlab = 100.0; //Rörelseenergin hos 

	for (int j = 1; j <= 1750; j++) {
		Tlab = 1.0 * j;
		for (int i = 3; i <= 100; i++) {
			N = i;
			TwoVectors k_and_w{ gaussLegendreInfMesh(N, scale) };
			std::vector<double> k{ k_and_w.v1 };
			std::vector<double> w{ k_and_w.v2 };


			LapackMat V_matrix = potential(channel, k, Tlab);

			double k0 = getk0(channel, Tlab);

			LapackMat T = computeTMatrix(channel, key, V_matrix, k, w, k0);
			//T.print();

			std::vector<std::complex<double>> phase = computePhaseShifts(channel, key, k0, T);

			double realPart = phase[0].real();
			myfile << std::fixed << std::setprecision(20) << std::endl;
			myfile << Tlab;
			myfile << ",";
			myfile << realPart;
			myfile << ",";
			myfile << N;
		}
		std::cout << Tlab << "\n";
	}
	myfile.close();
	return 0;
}