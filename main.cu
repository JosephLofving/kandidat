#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>



/* Kvar att göra som jag kan komma på just nu (innan vi är redo att testa t.ex. setupVGKernel)
 * - Fixa block/threads osv i scatteringfilerna samt i kernel calls (glöm inte getk0)
 * - Fixa computePhaseShifts och BlattToStapp så att de går att köra parallellt (de är typ orörda nu)
 */







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


/**
	Checks if the state is coupled or not.
	@param channel: Scattering channel
	@return			True if coupled, false if not
*/
bool isCoupled(std::vector<QuantumState> channel) {
	/* If there is only one channel the state is uncoupled, otherwise there are four channels and the state is coupled. */
	return !(channel.size() == 1);
}


__global__
void getk0(double* k0, double* TLab, int TLabLength, int tzChannel) {
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






int main() {
	std::vector<QuantumState> base = setupBase(0, 2, 0, 2);
	std::map<std::string, std::vector<QuantumState>> channels = setupNNChannels(base);

	std::string key = "j:0 s:0 tz:0 pi:1";
	std::vector<QuantumState> channel = channels[key];
	if (channel.size() == 0) {
		std::cout << "Invalid key";
		abort();
	}
	int tzChannel = channel[0].state["tz"];


	/* Set up Gauss-Legendre quadrature */
	int quadratureN = 100;
	double scale = 100;
	kAndWPtrs kAndW = gaussLegendreInfMesh(quadratureN, scale);
	double* k_h = kAndW.k;
	double* w_h = kAndW.w;

	/* Determine matrix and vector sizes */
	int matSize;
	int phasesSize;
	bool coupled = isCoupled(channel);
	if (coupled) {
		matSize = 2 * (quadratureN + 1);	// a trick to facilitate calculations (see scattering.cu)
		phasesSize = 3; // two phase shifts and one mixing angle in coupled case
	}
	else {
		matSize = quadratureN + 1;
		phasesSize = 1; // only one phase shift in uncoupled case
	}

	/* Prepare generation of TLab [Mev] */
	double TLabMin = 100;
	double TLabMax = 100;
	double TLabIncr = 1;
	int TLabLength = static_cast<int>( (TLabMax - TLabMin) / TLabIncr + 1);


	/* Allocate host memory */
	double* TLab_h = new double[TLabLength];
	double* k0_h = new double[TLabLength];
	cuDoubleComplex* V_h = new cuDoubleComplex[matSize * matSize * TLabLength];
	cuDoubleComplex* T_h = new cuDoubleComplex[matSize * matSize];
	cuDoubleComplex* G0_h = new cuDoubleComplex[matSize];
	cuDoubleComplex* VG_h = new cuDoubleComplex[matSize * matSize];
	cuDoubleComplex* F_h = new cuDoubleComplex[matSize * matSize];
	cuDoubleComplex* phases_h = new cuDoubleComplex[phasesSize];
	

	/* Allocate device memory */
	double* TLab_d;
	double* k0_d;
	double* k_d;
	double* w_d;
	cuDoubleComplex* V_d;
	cuDoubleComplex* T_d;
	cuDoubleComplex* G0_d;
	cuDoubleComplex* VG_d;
	cuDoubleComplex* F_d;
	cuDoubleComplex* phases_d;
	cudaMalloc((void**)&TLab_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&k0_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&k_d, quadratureN * sizeof(double));
	cudaMalloc((void**)&w_d, quadratureN * sizeof(double));
	cudaMalloc((void**)&V_d, matSize * matSize * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&G0_d, matSize * matSize * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&VG_d, matSize * matSize * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&F_d, matSize * matSize * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&T_d, matSize * matSize * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&phases_d, phasesSize * sizeof(cuDoubleComplex));


	/* Copy host variables to device variables */
	cudaMemcpy(TLab_d, TLab_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(k0_d, k0_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(k_d, k_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(G0_d, G0_h, matSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(VG_d, VG_h, matSize * matSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(F_d, F_h, matSize * matSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(T_d, T_h, matSize * matSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(phases_d, phases_h, phasesSize * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);


	/* Generate different experimental kinetic energies [MeV]*/
	for (int i = 0; i < TLabLength; i++) {
		TLab_h[i] = i * TLabIncr;
	}

	getk0<<<1, 1 >>>(k0_d, TLab_d, TLabLength, tzChannel);

	cudaMemcpy(k0_h, k0_d, TLabLength * sizeof(double), cudaMemcpyDeviceToHost);

	/* Create the potential matrix on the CPU */
	for (int i = 0; i < TLabLength; i++) {
		V_h[i] = potential(channel, k_h, TLab_h[i], k0_h[i], quadratureN)[i];
	}

	double mu = getReducedMass(channel);

	dim3 threadsPerBlock(matSize, matSize);
	dim3 blocksPerGrid(1, 1);
	if (matSize * matSize > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = ceil(double(matSize) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(matSize) / double(threadsPerBlock.y));
	}

	/* Call kernels on GPU */
	computeTMatrix <<<threadsPerBlock, blocksPerGrid>>> (T_d, V_d, G0_d, VG_d, F_d, k_d, w_d, k0_d, quadratureN, matSize, mu, coupled);
	//computePhaseShifts <<<threadsPerBlock, blocksPerGrid>>> (phases_h, mu, coupled, k0_d, T_d, quadratureN);
	
	cudaDeviceSynchronize();

	/* Copy (relevant) device variables to host variables */
	cudaMemcpy(T_h, T_d, matSize * matSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(phases_h, phases_d, phasesSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	cudaMemcpy(VG_h, VG_d, matSize * matSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < matSize * matSize; i += 5) {
		std::cout << cuCreal(VG_h[i]) << std::endl;
	}
	//-------------------------------------------
	// perhaps some printing of T or phases here
	//-------------------------------------------

	/* Free all the allocated memory */ 
	delete[] TLab_h;
	delete[] k0_h;
	delete[] V_h;
	delete[] G0_h;
	delete[] VG_h;
	delete[] F_h;
	delete[] T_h;
	delete[] phases_h;
	cudaFree(TLab_d);
	cudaFree(k0_d);
	cudaFree(V_d);
	cudaFree(G0_d);
	cudaFree(VG_d);
	cudaFree(F_d);
	cudaFree(T_d);
	cudaFree(phases_d);

	return 0;
}