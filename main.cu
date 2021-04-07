#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>
#include <iomanip>

__global__
void setupVG(double *k, double *w, std::complex<double> *V, double k0, std::complex<double> *G0, std::complex<double> *VG, int matrixHeight)
{
	for (int row = 0; row < matrixHeight; row++) {
		for (int column = 0; column < matrixHeight; column++) {
			VG[row+column*matrixHeight] = V[row+column*matrixHeight]*G0[column];
		}
	}
}

int main() {
	/*
	std::ofstream myfile;
    myfile.open ("data.csv");

	myfile << "Real av fasskift";
	myfile << ",";
	myfile << "N";
	myfile << "\n";
	*/

	std::vector<QuantumState> base = setupBase(0, 2, 0, 2);
    std::map<std::string, std::vector<QuantumState> > channels = setupNNChannels(base);
	printChannels(channels);

	int N = 100;
	double scale = 100;


	std::string key = "j:0 s:0 tz:0 pi:1";
	std::vector<QuantumState> channel = channels[key];
	if (channel.size()==0) {
		std::cout << "Invalid key";
		abort();
	}
	printStates(channel);

	double Tlab = 100.0;

//------------------------------------------------------------------
//-------------------------- FOR GPU --------------------------------
//------------------------------------------------------------------

	std::vector<double> k;
	std::vector<double> w;

	

	double k0 = getk0(channel, Tlab);

	// Allocate Unified Memory, i.e. let the following objects be accessible
	// from both GPU and CPU
	

	// Initialize k, w, V_matrix on CPU
	TwoVectors k_and_w{ gaussLegendreInfMesh(N, scale) };
	k = k_and_w.v1;
	w = k_and_w.v2;

	int N = k.size();

	std::complex<double> *k_dev;
	std::complex<double> *w_dev;

	for(int kElement = 0; kElement < k.size(); kElement++){
		k_dev[kElement] = k[kElement];
	}

	for(int wElement = 0; wElement < w.size(); wElement++){
		w_dev[wElement] = w[wElement];
	}

	LapackMat V_matrix = potential(channel, k, Tlab);

	std::complex<double> *V_dev;

	for(int VElement = 0; VElement < V_matrix.content().size(); VElement++){
		V_dev[VElement] = V_matrix.content()[VElement];
	}

	std::vector<std::complex<double>> G0 = setupG0Vector(channel, k, w, k0);

	std::complex<double> *G0_dev;

	for(int G0Element = 0; G0Element < G0.size(); G0Element++){
		G0_dev[G0Element] = G0[G0Element];
	}

	std::complex<double> *VG_dev;
	
	cudaMallocManaged(&VG_dev,N*N*sizeof(std::complex<double>));
	cudaMallocManaged(&V_dev,N*N*sizeof(std::complex<double>));
	cudaMallocManaged(&G0_dev,N*sizeof(std::complex<double>));
	cudaMallocManaged(&k_dev,N*sizeof(std::complex<double>));
	cudaMallocManaged(&w_dev,N*sizeof(std::complex<double>));

	setupVG<<<1, 1>>>(k_dev,w_dev,V_dev,k0,G0_dev,VG_dev,N);

	// Compute the phase shifts for many different T matrices
	//std::vector<std::complex<double>> phase = computePhaseShifts<<<1,1>>>(channel, key, k0, T);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();




	// Free memory from Unified Memory
	cudaFree(VG_dev;
	cudaFree(V_dev);
	cudaFree(G0_dev);
	cudaFree(k_dev);
	cudaFree(w_dev);

	
	
	return 0;
}