#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>
#include <iomanip>

#include <cuComplex.h>

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

	double k0 = getk0(channel, Tlab);

	TwoVectors k_and_w = gaussLegendreInfMesh(NKvadratur, scale);

	std::vector<double> kVect = k_and_w.v1;
	double* k = &kVect[0];
	std::vector<double> wVect = k_and_w.v2;
	double* w = &wVect[0];

std::vector<std::complex<double>> G0_std = setupG0Vector(channel, kVect, wVect, k0);

	int N = kVect.size();

	LapackMat V_matrix = potential(channel, kVect, Tlab);
	cuDoubleComplex *V = new cuDoubleComplex[N*N];

	for (int i = 0; i < V_matrix.height*V_matrix.width; i++) {
		V[i] = make_cuDoubleComplex(V_matrix.contents[i].real(), V_matrix.contents[i].imag());
	}

	cuDoubleComplex G0[N];

	for(int i = 0; i < G0_std.size(); i++){
		G0[i] = make_cuDoubleComplex(G0_std[i].real(), G0_std[i].imag());
	}

	double *k_dev;// = (double*) malloc(N*sizeof(double));
	double *w_dev;// = (double*) malloc(N*sizeof(double));
	cuDoubleComplex *V_dev;
	cuDoubleComplex *G0_dev;
	cuDoubleComplex *VG_dev;

	cudaMalloc((void**)&k_dev, N*sizeof(double));
	cudaMalloc((void**)&w_dev, N*sizeof(double));
	cudaMalloc((void**)&V_dev, N*N*sizeof(cuDoubleComplex));
	cudaMalloc((void**)&G0_dev, N*sizeof(cuDoubleComplex));
	cudaMalloc((void**)&VG_dev, N*N*sizeof(cuDoubleComplex));

	cudaMemcpy(&k_dev, &k, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&w_dev, &w, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(&V_dev, &V, N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(&G0_dev, &G0, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	setupVG<<<1, 1>>>(k_dev, w_dev, V_dev, k0, G0_dev, VG_dev, N);

	cudaDeviceSynchronize();

	cuDoubleComplex VG[N*N];
	cudaMemcpy(&VG, &VG_dev, N*N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	std::cout << cuCreal(VG[0]);

	cudaFree(VG_dev);
	cudaFree(V_dev);
	cudaFree(G0_dev);
	cudaFree(k_dev);
	cudaFree(w_dev);

	return 0;
}