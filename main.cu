#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include "computeTMatrix.h"
#include "solveLS.h"
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <chrono>

void datainsamling() {

	/*
	std::ofstream myfile;
	myfile.open("data.csv");

	myfile << "Real av fasskift";
	myfile << ",";
	myfile << "N";
	myfile << "\n";
	*/


	for (int i = 0; i < 5; i++) {
		using milli = std::chrono::milliseconds;
		auto start = std::chrono::high_resolution_clock::now();

		solveLS();
		
		auto finish = std::chrono::high_resolution_clock::now();
		std::cout << "myFunction() took "
			<< std::chrono::duration_cast<milli>(finish - start).count()
			<< " milliseconds\n";
	}

}

int main() {

	datainsamling();

	return 0;
}