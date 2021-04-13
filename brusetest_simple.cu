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
	double kVect = 78;
	double* k = &kVect;
	double* k_dev;
	cudaMalloc((void**)&k_dev, sizeof(double));
	cudaMemcpy(k_dev, k, sizeof(double), cudaMemcpyHostToDevice);

	std::cout << *k << std::endl;
	setupVG << <1, 1 >> > (k_dev);
	cudaMemcpy(k, k_dev, sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << *k << std::endl;
	cudaFree(k_dev);

	return 0;
}