#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>
#include <iomanip>

#include <cuComplex.h>

// Borde vara V^T * G0 iställer
__global__
void setupVG(double *k, double *w, cuDoubleComplex *V, double k0, cuDoubleComplex *G0, cuDoubleComplex *VG, int matrixHeight)
{
	for (int row = 0; row < matrixHeight; row++) {
		for (int column = 0; column < matrixHeight; column++) {
			VG[row+column*matrixHeight] = cuCmul(V[row+column*matrixHeight],G0[column]);
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

	int NKvadratur = 100;
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
	TwoVectors k_and_w{ gaussLegendreInfMesh(NKvadratur, scale) };
	k = k_and_w.v1;
	w = k_and_w.v2;

	int N = k.size();

	double *k_dev = new double[N];
	double *w_dev = new double[N];
	cuDoubleComplex *V_dev = new cuDoubleComplex[N*N];
	cuDoubleComplex *G0_dev = new cuDoubleComplex[N];
	cuDoubleComplex *VG_dev = new cuDoubleComplex[N*N];

	for(int kElement = 0; kElement < k.size(); kElement++){
		k_dev[kElement] = k[kElement];
	}

	for(int wElement = 0; wElement < w.size(); wElement++){
		w_dev[wElement] = w[wElement];
	}

	LapackMat V_matrix = potential(channel, k, Tlab);



	for(int VElement = 0; VElement < V_matrix.contents.size(); VElement++){
		V_dev[VElement] = make_cuDoubleComplex(V_matrix.contents[VElement].real(), V_matrix.contents[VElement].imag());
	}

	std::vector<std::complex<double>> G0 = setupG0Vector(channel, k, w, k0);



	for(int G0Element = 0; G0Element < G0.size(); G0Element++){
		G0_dev[G0Element] = make_cuDoubleComplex(G0[G0Element].real(),G0[G0Element].imag());
	}



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

	std::cout << cuCreal(VG_dev[0]);


	// Free memory from Unified Memory
	cudaFree(VG_dev);
	cudaFree(V_dev);
	cudaFree(G0_dev);
	cudaFree(k_dev);
	cudaFree(w_dev);

	return 0;
}