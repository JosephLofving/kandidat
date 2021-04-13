#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__
void setupVG(double* a)
{
	*a *= 100;
}


int main() {
	int N = 100;

	double kVect = 78;
	double* k = &kVect;
	double* k_dev;
	cudaMalloc((void**)&k_dev, N * sizeof(double));
	cudaMemcpy(k_dev, k, N * sizeof(double), cudaMemcpyHostToDevice);

	std::cout << k << std::endl;
	setupVG << <1, 1 >> > (k_dev);
	cudaMemcpy(k, k_dev, N * sizeof(double), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	std::cout << k << std::endl;
	cudaFree(k_dev);
	std::cout << "hej" << std::endl;

	return 0;
}