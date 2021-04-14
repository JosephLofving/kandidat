#include <iostream>
#include <iomanip>
#include "mesh.h"
#include <vector>
#include "scattering.h"
#include "potential.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void setupVG(cuDoubleComplex *V, cuDoubleComplex *G0, cuDoubleComplex *VG, int matWidth) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < matWidth && col < matWidth) {
		VG[row + col*matWidth] = cuCmul(V[row + col*matWidth], G0[col]);
	}
}

__global__
void setupVGNonParallell(cuDoubleComplex *V, cuDoubleComplex *G0, cuDoubleComplex *VG, int matrixHeight)
{
	for (int row = 0; row < matrixHeight; row++) {
		for (int column = 0; column < matrixHeight; column++) {
			VG[row+column*matrixHeight] = cuCmul(V[row+column*matrixHeight],G0[column]);
		}
	}
}

int main() {
	const int Nkvadr = 100;
	double scale = 100.0;
	double Tlab = 100.0;

	std::vector<QuantumState> base = setupBase(0, 2, 0, 2);
    std::map<std::string, std::vector<QuantumState> > channels = setupNNChannels(base);
	std::string key = "j:0 s:0 tz:0 pi:1";
	std::vector<QuantumState> channel = channels[key];

	double k0 = getk0(channel, Tlab);

	TwoVectors k_and_w = gaussLegendreInfMesh(Nkvadr, scale);

	std::vector<double> kVect = k_and_w.v1;
	std::vector<double> wVect = k_and_w.v2;

	double* k = &kVect[0];
	double* w = &wVect[0];

	std::vector<std::complex<double>> G0_std = setupG0Vector(channel, kVect, wVect, k0);
	int N = G0_std.size();

	LapackMat V_matrix = potential(channel, kVect, Tlab);
	cuDoubleComplex* V_host = new cuDoubleComplex[V_matrix.width*V_matrix.height];

	for (int i = 0; i < N*N; i++) {
		V_host[i] = make_cuDoubleComplex(V_matrix.contents[i].real(), V_matrix.contents[i].imag());
	}


	cuDoubleComplex G0[(Nkvadr+1)];
	for(int i = 0; i < G0_std.size(); i++){
		G0[i] = make_cuDoubleComplex(G0_std[i].real(), G0_std[i].imag());
	}


	cuDoubleComplex* G0_dev;
	cuDoubleComplex* V_dev;
	cuDoubleComplex* VG_dev;

	cudaMalloc((void**)&V_dev, N*N*sizeof(double));
	cudaMalloc((void**)&VG_dev, N*N*sizeof(double));
	cudaMalloc((void**)&G0_dev, N*sizeof(double));

	cudaMemcpy(G0_dev, G0, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(V_dev, V_host, N*N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(VG_dev, V_host, N*N*sizeof(double), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (N*N > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x  = ceil(double(N)/double(threadsPerBlock.x));
		blocksPerGrid.y  = ceil(double(N)/double(threadsPerBlock.y));
	}

	//setupVG <<<threadsPerBlock, blocksPerGrid>>> (V_dev, G0_dev, VG_dev, N);
	setupVGNonParallell <<<1,1>>> (V_dev,G0_dev,VG_dev,N);
	cudaDeviceSynchronize();

	cuDoubleComplex* VG_host= new cuDoubleComplex[V_matrix.width*V_matrix.height];
	VG_host[5]= make_cuDoubleComplex(1,1);
	cudaMemcpy(VG_host, VG_dev, N*N*sizeof(double), cudaMemcpyDeviceToHost);

	//gpuErrchk( cudaPeekAtLastError() );

	// std::cout << V_host[0] << std::endl;
	// std::cout << V_host[100] << std::endl;

	cudaFree(G0_dev);
	cudaFree(V_dev);
	cudaFree(VG_dev);

	for (int i = 0; i < V_matrix.width*V_matrix.height; i += 5) {
		std::cout << cuCreal(VG_host[i]) << std::endl;
	}

	return 0;
}
