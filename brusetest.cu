#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>
#include <iomanip>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__
void setupVG(double* k)
{	
	//int matrixHeight = 100;
	k[0] *= 100;
}


int main() {
	int N = 100;
	TwoVectors k_and_w = gaussLegendreInfMesh(100, 100);

	std::vector<double> kVect = k_and_w.v1;
	double* k = &kVect[0];
	double* k_dev;
	cudaMalloc((void**)&k_dev, N * sizeof(double));
	cudaMemcpy(&k_dev, &k, N * sizeof(double), cudaMemcpyHostToDevice);

	std::cout << k_dev[0];
	setupVG << <1, 1 >> > (k_dev);

	cudaDeviceSynchronize();

	std::cout << k_dev[0];
	cudaFree(k_dev);
	std::cout << "hej";

	return 0;
}