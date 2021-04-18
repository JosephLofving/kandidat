#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>

__global__
void getk0(double* k0, int tzChannel, double* TLab, int TLabLength) {
	double k0Squared = 0;
	for (int i = 0; i < TLabLength; i++) {
		/* Proton-proton scattering */
		if (tzChannel == -1) {
			k0Squared = constants::protonMass * TLab[i] / 2;
		}
		/* Proton-neutron scattering */
		else if (tzChannel == 0) {
			k0Squared = pow(constants::neutronMass, 2) * TLab[i] * (TLab[i]
							+ 2 * constants::protonMass) / ((pow(constants::protonMass
							+ constants::neutronMass, 2) + 2 * TLab[i] * constants::neutronMass));
		}
		/* Neutron-neutron scattering */
		else if (tzChannel == 1) {
			k0Squared = constants::neutronMass * TLab[i] / 2;
		}

		k0[i] = sqrtf(k0Squared); // Does not handle case where tz is NOT -1, 0 or 1 (should be handled earlier?)
	}
}


/**
	Gets the reduced mass by checking the isospin channel, which determines the type of NN scattering
	@param channel:	Scattering channel
	@return			Reduced mass
*/
double getReducedMass(std::vector<QuantumState> channel) {
	double mu = 0;
	int tzChannel = channel[0].state["tz"];
	if (tzChannel == -1)	 // Proton-proton scattering
		mu = constants::protonMass / 2;
	else if (tzChannel == 0) // Proton-neutron scattering
		mu = constants::nucleonReducedMass;
	else if (tzChannel == 1) // Neutron-neutron scattering
		mu = constants::neutronMass / 2;

	return mu;
}


int main() {
	std::vector<QuantumState> base = setupBase(0, 2, 0, 2);
	std::map<std::string, std::vector<QuantumState>> channels = setupNNChannels(base);

	std::string key = "j:0 s:0 tz:0 pi:1";
	std::vector<QuantumState> channel = channels[key];
	if (channel.size() == 0) {
		std::cout << "Invalid key";
		abort();
	}

	/* Create host array for TLab [MeV] */
	double TLabMin = 0;
	double TLabMax = 290;
	double TLabIncr = 0.01;
	int TLabLength = (int)(TLabMax - TLabMin) / TLabIncr; // static_cast<int>?
	double* TLab = new double[TLabLength];

	for (int i = 0; i < TLabLength; i++) {
		TLab[i] = i * TLabIncr;
	}

	/* Set up Gauss-Legendre quadrature */
	int quadratureN = 100;
	double scale = 100;
	kAndWPtrs kAndW = gaussLegendreInfMesh(quadratureN, scale);
	double* k = kAndW.k;
	double* w = kAndW.w;

	/* Determine matrix and vector sizes for uncoupled cases */
	int matSize = quadratureN + 1;
	int phasesSize = 1; // only one phase shift in uncoupled case

	/* Determine matrix and vector sizes for coupled cases */
	bool coupled = isCoupled(channel);
	if (coupled) {
		matSize *= 2;	// a trick to facilitate calculations (see scattering.cu)
		phasesSize = 3; // two phase shifts and one mixing angle in coupled case
	}

	/* Get the k0 host array with parallelization on GPU */
	double* k0_h = new double[TLabLength];
	int tzChannel = channel[0].state["tz"];
	double* k0_d;

	cudaMalloc((void**)&k0_d, quadratureN * sizeof(double));
	cudaMemcpy(k0_d, k0_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);
	getk0 << <1, 1 >> > (k0_d, tzChannel, TLab, TLabLength);
	cudaMemcpy(k0_h, k0_d, matSize * matSize * sizeof(double), cudaMemcpyDeviceToHost);

	/* Create the potential matrix on the CPU */
	cuDoubleComplex** VMatrix = new cuDoubleComplex* [matSize * matSize * TLabLength];
	for (int i = 0; i < TLabLength; i++) {
		VMatrix[i] = potential(channel, k, TLab[i], k0_h[i], quadratureN);
	}

	double mu = getReducedMass(channel);

	/* Allocate host memory and declare device variable */
	cuDoubleComplex* VG_h = new cuDoubleComplex[matSize * matSize];
	cuDoubleComplex* VG_d;

	/* Allocate device memory and copy host variable to device variable*/
	cudaMalloc((void**)&VG_d, matSize * matSize * sizeof(cuDoubleComplex));
	cudaMemcpy(VG_d, VG_h, matSize * matSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	/* Call (device) kernels from the (host) function computeTMatrix and declare host variable for phase shifts */
	cuDoubleComplex* T = computeTMatrix(VMatrix, k, w, k0_d, quadratureN, matSize, mu, coupled); // CPU function that calls kernels, see scattering.cu
	cuDoubleComplex* phases = new cuDoubleComplex[phasesSize];

	/* Declare device variables */
	cuDoubleComplex* T_d;
	cuDoubleComplex* phases_d;

	/* Allocate memory on the device */
	cudaMalloc((void**)&T_d, matSize * matSize * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&phases_d, phasesSize * sizeof(cuDoubleComplex));

	/* Copy host variables to variables allocated on device */
	cudaMemcpy(T_d, T, matSize * matSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(phases_d, phases, phasesSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	/* Parallellized calculation of phase shifts */
	computePhaseShifts <<<1,1>>> (phases, mu, coupled, k0, T, quadratureN);

	/* Copy device variables to variables allocated on host */
	cudaMemcpy(T, T_d, matSize * matSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(phases, phases_d, phasesSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	
	//------------------------------------------
	// perhaps some printing of T or phases here
	//------------------------------------------

	/* Free all the allocated memory */ 
	delete[] TLab;
	delete[] k0_h;
	delete[] VMatrix;
	delete[] phases;
	delete[] VG_h;
	cudaFree(k0_d);
	cudaFree(T_d);
	cudaFree(phases_d);
	cudaFree(VG_d);

	return 0;
}