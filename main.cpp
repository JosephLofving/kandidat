#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>
#include <iomanip>

int main() {

	std::vector<QuantumState> base = setupBase(0, 2, 0, 2);
    std::map<std::string, std::vector<QuantumState> > channels = setupNNChannels(base);
	printChannels(channels);

	int N = 100;
	double scale = 100;

	TwoVectors k_and_w = gaussLegendreInfMesh(N, scale);

	std::vector<double> k = k_and_w.v1;
	std::vector<double> w = k_and_w.v2;

	std::string key = "j:0 s:0 tz:0 pi:1"; //could change key format
	std::vector<QuantumState> channel = channels[key];
	if (channel.size()==0) {
		std::cout << "Invalid key";
		abort();
	}
	printStates(channel);

	double Tlab = 100.0; //R�relseenergin hos 

	LapackMat V_matrix = potential(channel, k, Tlab);

	double k0 = getk0(channel, Tlab);

	LapackMat T = computeTMatrix(channel, key, V_matrix, k, w, k0);
	//T.print();



	return 0;
}