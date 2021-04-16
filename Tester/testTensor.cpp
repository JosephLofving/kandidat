#include "mesh.h"
#include "tensorPotential.h"
#include <fstream>
#include <iomanip>

int main() {
	
	std::vector<QuantumState> base = setupBase(0,2,0,2);
	std::map<std::string, std::vector<QuantumState>> channels = setupNNChannels(base);
	printChannels(channels);

	std::string key = "j:0 s:0 tz:0 pi:1"; 
	std::vector<QuantumState> channel = channels[key];
	if (channel.size()==0) {
		std::cout << "Invalid key";
		abort();
	}
	printStates(channel);

    int N{ 10 };
	double scale{ 100 };
    TwoVectors k_and_w{ gaussLegendreInfMesh(N, scale) };
	std::vector<double> k{ k_and_w.v1 };
	std::vector<double> w{ k_and_w.v2 };

    std::vector<double> Tlab(25,0);
    for(int i=0; i<25;i++){
        Tlab[i]=i+1;
    }


	// allocate unified memory for VTensor
	//cudaMallocManaged(&VTensor, N * sizeof(Tensor));

	// initialize VTensor on host
	Tensor VTensor = potential(channel, k, Tlab);


    for(int i =0; i<Tlab.size(); i++){
        VTensor.print(i);
    }



	//cudaFree(VTensor);

	return 0;
}
