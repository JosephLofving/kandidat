#include <iostream>
#include <iomanip>
#include "mesh.h"
#include <vector>
#include "scattering.h"
#include "potential.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>
#include <fstream>

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

	// printf("Block: %d,%d \tThread: %d,%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

	// if (row < matWidth && col < matWidth && cuCreal(VG[row + col*matWidth]) != cuCreal(cuCmul(V[row + col*matWidth], G0[col]))) {
	// 	printf("Row: %d  \tcol: %d\t\tGPU: %.2e\tCPU: %.2e\n", row, col, cuCreal(cuCmul(V[row + col*matWidth], G0[col])), cuCreal(VG[row + col*matWidth]));
	// }

	if (row < matWidth && col < matWidth) {
		printf("Block: %d,%d \tThread: %d,%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
		VG[row + col*matWidth] = cuCmul(V[row + col*matWidth], G0[col]);
	}
}

__global__
void setupVGNonParallell(cuDoubleComplex *V, cuDoubleComplex *G0, cuDoubleComplex *VG, int matrixHeight)
{
	for (int row = 0; row < matrixHeight; row++) {
		for (int col = 0; col < matrixHeight; col++) {
			VG[row+col*matrixHeight] = cuCmul(V[row+col*matrixHeight],G0[col]);
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
	LapackMat VG_CPU = setupVGKernel(channel, key, V_matrix, kVect, wVect, k0);


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

	cuDoubleComplex* VG_host = new cuDoubleComplex[V_matrix.width*V_matrix.height];

	for (int i = 0; i < N*N; i++) {
		VG_host[i] = make_cuDoubleComplex(1.0, 1.0);//make_cuDoubleComplex(VG_CPU.contents[i].real(), VG_CPU.contents[i].imag());
	}

	cudaMalloc((void**)&V_dev, N*N*sizeof(cuDoubleComplex));
	cudaMalloc((void**)&VG_dev, N*N*sizeof(cuDoubleComplex));
	cudaMalloc((void**)&G0_dev, N*sizeof(cuDoubleComplex));

	cudaMemcpy(G0_dev, G0, N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(V_dev, V_host, N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(VG_dev, VG_host, N*N*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);
	if (N*N > 512) {
		threadsPerBlock.x = 32;//512;
		threadsPerBlock.y = 32;//512;
		blocksPerGrid.x  = 4;//ceil(double(N)/double(threadsPerBlock.x));
		blocksPerGrid.y  = 4;//ceil(double(N)/double(threadsPerBlock.y));
	}

	printf("%d, %d\n", blocksPerGrid.x, threadsPerBlock.x);

	setupVG <<<blocksPerGrid, threadsPerBlock>>> (V_dev, G0_dev, VG_dev, N);
	//setupVG <<<threadsPerBlock, blocksPerGrid>>> (V_dev, G0_dev, VG_dev, N);
	// setupVGNonParallell <<<1,1>>> (V_dev,G0_dev,VG_dev,N);
	cudaDeviceSynchronize();

	// cuDoubleComplex* VG_host= new cuDoubleComplex[V_matrix.width*V_matrix.height];
	// VG_host[5]= make_cuDoubleComplex(1,1);
	cudaMemcpy(VG_host, VG_dev, N*N*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	//gpuErrchk( cudaPeekAtLastError() );

	// std::cout << V_host[0] << std::endl;
	// std::cout << V_host[100] << std::endl;

	cudaFree(G0_dev);
	cudaFree(V_dev);
	cudaFree(VG_dev);

	for (int i = 0; i < N*N; i++) {
		// printf("%d\n", i);
		// if (cuCreal(VG_host[i]) - VG_CPU.contents[i].real() != 0) {
			// printf("Index: %d \t GPU: %f \t CPU: %f\n", i, cuCreal(VG_host[i]), VG_CPU.contents[i].real());
		// }
		printf("GPU: %.2e \t CPU: %.2e\n", cuCreal(VG_host[i]), VG_CPU.contents[i].real());
	}

	return 0;
}
