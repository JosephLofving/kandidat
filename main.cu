#include "mesh.h"
#include "scattering.h"
#include "potential.h"
#include "computeTMatrix.h"
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>


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
void setupG0VectorSum(
	double* sum,
	double* k0,
	int quadratureN,
	int TLabLength,
	double* k,
	double* w) {


	for (int energyIndex = 0; energyIndex < TLabLength; ++energyIndex) {
		sum[energyIndex] = 0;
		for (int column = 0; column < quadratureN; ++column) {
			sum[energyIndex] += w[column] / (k0[energyIndex] * k0[energyIndex] - k[column] * k[column]);
		}
	}
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
	const int quadratureN = 5;
	const double scale = 100;


	/* Determine matrix and vector sizes */
	int matLength;
	int phasesSize;
	bool coupled = isCoupled(channel);
	if (coupled) {
		matLength = 2 * (quadratureN + 1);	// a trick to facilitate calculations (see scattering.cu)
		phasesSize = 3; // two phase shifts and one mixing angle in coupled case
	}
	else {
		matLength = quadratureN + 1;
		phasesSize = 1; // only one phase shift in uncoupled case
	}

	/* Prepare generation of TLab [Mev] */
	const double TLabMin = 100;
	const double TLabMax = 100;
	constexpr int TLabLength = 3;
	const double TLabIncr = (TLabMax - TLabMin + 1) / TLabLength;

	/* Allocate host memory */
	double* TLab_h = new double[TLabLength];
	double* sum_h = new double[TLabLength];
	double* k0_h = new double[TLabLength];
	double* k_h = new double[quadratureN];
	double* w_h = new double[quadratureN];
	cuDoubleComplex* V_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	cuDoubleComplex* T_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	cuDoubleComplex* G0_h = new cuDoubleComplex[matLength * TLabLength];
	cuDoubleComplex* VG_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	cuDoubleComplex* F_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	cuDoubleComplex* phases_h = new cuDoubleComplex[phasesSize * TLabLength];

	/* Generate different experimental kinetic energies [MeV]*/
	for (int i = 0; i < TLabLength; i++) {
		TLab_h[i] = i * TLabIncr + TLabMin;
	}

	/* Allocate device memory */
	double* TLab_d;
	double* k0_d;
	double* k_d;
	double* w_d;
	double* sum_d;
	cuDoubleComplex* V_d;
	cuDoubleComplex* T_d;
	cuDoubleComplex* G0_d;
	cuDoubleComplex* VG_d;
	cuDoubleComplex* F_d;
	cuDoubleComplex* phases_d;



	gaussLegendreInfMesh(k_h, w_h, quadratureN, scale);


	//printf("kk[0] = %.4e\n", k_h[0]);
	//printf("kk[1] = %.4e\n", k_h[1]);
	//printf("kk[2] = %.4e\n", k_h[2]);
	//printf("kk[3] = %.4e\n", k_h[3]);
	//printf("kk[4] = %.4e\n", k_h[4]);
	//
	//printf("ww[0] = %.4e\n", w_h[0]);
	//printf("ww[1] = %.4e\n", w_h[1]);
	//printf("ww[2] = %.4e\n", w_h[2]);
	//printf("ww[3] = %.4e\n", w_h[3]);
	//printf("ww[4] = %.4e\n", w_h[4]);



	cudaMalloc((void**)&TLab_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&k0_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&sum_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&k_d, quadratureN * sizeof(double));
	cudaMalloc((void**)&w_d, quadratureN * sizeof(double));
	cudaMalloc((void**)&G0_d, matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&V_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&VG_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&F_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&T_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&phases_d, phasesSize * TLabLength * sizeof(cuDoubleComplex));



	/* Copy host variables to device variables */
	cudaMemcpy(TLab_d, TLab_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(sum_d, sum_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(k0_d, k0_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(k_d, k_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(G0_d, G0_h, matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(VG_d, VG_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(F_d, F_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(T_d, T_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(phases_d, phases_h, phasesSize * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(matLength, matLength,TLabLength); //Blocksize
	dim3 blocksPerGrid(1,1,1);//Gridsize
	if (matLength * matLength > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = ceil(double(matLength) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(matLength) / double(threadsPerBlock.y));
	}

	getk0 <<<1,1>>>(k0_d, TLab_d, TLabLength, tzChannel);

	cudaMemcpy(k0_h, k0_d, TLabLength * sizeof(double), cudaMemcpyDeviceToHost);

	potential(V_h, channel, k_h, TLab_h, k0_h, quadratureN, TLabLength, coupled, matLength);

	/* Create the potential matrix on the CPU */

	cudaMemcpy(V_d, V_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);


	double mu = getReducedMass(channel);

	/* Call kernels on GPU */
	setupG0VectorSum <<<1,1>>> (sum_d, k0_d, quadratureN, TLabLength, k_d, w_d);
	setupG0Vector <<<threadsPerBlock, blocksPerGrid >>> (G0_d, k_d, w_d, k0_d, sum_d, quadratureN, matLength, TLabLength, mu, coupled);
	/* Setup the VG kernel and, at the same time, the F matrix */
	setupVGKernel <<<threadsPerBlock, blocksPerGrid >>> (VG_d, V_d, G0_d, F_d, k_d, w_d, k0_d, quadratureN, matLength, TLabLength, mu, coupled);

	/* Copying the matricies back to the CPU for CuBLAS */
	// cudaDeviceSynchronize();
	// cudaMemcpy(T_h, T_d, matLength* matLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	// cudaMemcpy(F_h, F_d, matLength* matLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	// cudaMemcpy(V_h, V_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	/* Solve the equation FT = V with cuBLAS */
	computeTMatrixCUBLAS(T_d, F_d, V_d, matLength, TLabLength);

	// cudaMemcpy(T_d, T_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	/* Computes the phase shifts for the given T-matrix*/
	computePhaseShifts <<<threadsPerBlock, blocksPerGrid >>> (phases_d, T_d, k0_d, quadratureN, mu, coupled, TLabLength, matLength);

	cudaDeviceSynchronize();

	/* Copy (relevant) device variables to host variables */
	// cudaMemcpy(T_h, T_d, matLength * matLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(phases_h, phases_d, phasesSize * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	// cudaMemcpy(k_h, k_d, quadratureN * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(w_h, w_d, quadratureN * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(k0_h, k0_d, TLabLength * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(V_h, V_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	// cudaMemcpy(F_h, F_d, matLength * matLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	// cudaMemcpy(VG_h, VG_d, matLength * matLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);


	for (int i = 0; i < TLabLength; i++) {
		for (int j = 0; j < phasesSize; ++j) {
			printf("\nTLab = %f", TLab_h[i]);
			printf("\nReal(phases[%i]) = %.10e", j, cuCreal(phases_h[j + i * phasesSize]));
			printf("\nImag(phases[%i]) = %.10e", j, cuCimag(phases_h[j + i * phasesSize]));
			printf("\n");
		}
	}

	/* Free all the allocated memory */
	delete[] TLab_h;
	delete[] k0_h;
	delete[] V_h;
	delete[] G0_h;
	delete[] VG_h;
	delete[] F_h;
	delete[] T_h;
	delete[] phases_h;
	//delete[] k_h;
	delete[] w_h;
	cudaFree(TLab_d);
	cudaFree(k0_d);
	cudaFree(V_d);
	cudaFree(G0_d);
	cudaFree(VG_d);
	cudaFree(F_d);
	cudaFree(T_d);
	cudaFree(phases_d);
	cudaFree(k_d);
	cudaFree(w_d);

	return 0;
}