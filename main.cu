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
	
	for (int i = 0; i < TLabLength; ++i) {

		if (tzChannel == -1) {	  // Proton-proton scattering
			k0Squared = constants::protonMass * TLab[i] / 2;
		}
		else if (tzChannel == 0) { // Proton-neutron scattering
			k0Squared = pow(constants::neutronMass, 2) * TLab[i] * (TLab[i] +
				2 * constants::protonMass) / ((pow(constants::protonMass +
					constants::neutronMass, 2) + 2 * TLab[i] * constants::neutronMass));
		}
		else if (tzChannel == 1) { // Neutron-neutron scattering
			k0Squared = constants::neutronMass * TLab[i] / 2;
		}

		k0[i] = sqrtf(k0Squared); // Does not handle case where tz is NOT -1, 0 or 1.
	}
}


int main() {

	std::vector<QuantumState> base = setupBase(0, 2, 0, 2);
	std::map<std::string, std::vector<QuantumState> > channels = setupNNChannels(base);

	int NKvadratur = 100;
	double scale = 100;

	std::string key = "j:0 s:0 tz:0 pi:1";
	std::vector<QuantumState> channel = channels[key];
	if (channel.size() == 0) {
		std::cout << "Invalid key";
		abort();
	}

	double* TLab;
	double TLabMin = 0;
	double TLabMax = 290;
	double TLabStep = 0.01;
	int TLabLength = (int)(TLabMax - TLabMin) / TLabStep;
	
	for (int i = 0; i < TLabLength; ++i) {
			TLab[i] = i * TLabStep;
	}

	int tzChannel = channel[0].state["tz"];
	double* k0 = new double[TLabLength];
	getk0(k0, tzChannel, TLab, TLabLength);

	kAndWPtrs kAndW = gaussLegendreInfMesh(NKvadratur, scale);

	double* k = kAndW.k;
	double* w = kAndW.w;

	int G0Size = NKvadratur + 1;
	int phasesSize = 1;
	bool coupled = (isCoupled(channel));
	if (coupled) {
		G0Size *= 2;
		phasesSize = 3;
	}

	cuDoubleComplex** V_matrix = new cuDoubleComplex*[G0Size * G0Size * TLabLength];
	for (int i = 0; i < TLabLength; ++i) {
		V_matrix[i] = potential(channel, k, TLab[i], k0[i], NKvadratur);
	}

	double mu = getReducedMass(channel);

	cuDoubleComplex* T = computeTMatrix(V_matrix, k, w, k0, NKvadratur, G0Size, mu, coupled);


	cuDoubleComplex* phases = new cuDoubleComplex[phasesSize];
	computePhaseShifts(phases, mu, coupled, k0, T, NKvadratur);


	cudaMalloc((void**)&V_dev, N * N * sizeof(double));
	cudaMemcpy(G0_dev, G0, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(V_host, VG_dev, N * N * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(G0_dev);

	return 0;
}