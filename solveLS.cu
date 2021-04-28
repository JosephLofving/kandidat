#include "solveLS.h"



/**
	Gets the reduced mass by checking the isospin channel, which determines the type of NN scattering
	@param channel:	Scattering channel
	@return			Reduced mass
*/
double getReducedMass(std::vector<QuantumState> channel) {
	double mu = 0;
	int tzChannel = channel[0].state["tz"];
	/* Proton-proton scattering */
	if (tzChannel == -1)
		mu = constants::protonMass / 2;
	/* Proton-neutron scattering */
	else if (tzChannel == 0)
		mu = constants::nucleonReducedMass;
	/* Neutron-neutron scattering */
	else if (tzChannel == 1)
		mu = constants::neutronMass / 2;

	return mu;
}



/**
	Checks if the quantum state is coupled or not.
	@param channel: Scattering channel
	@return			True if coupled, false if not
*/
bool isCoupled(std::vector<QuantumState> channel) {
	/* If there is only one channel the state is uncoupled, otherwise there are four channels and the state is coupled. */
	return !(channel.size() == 1);
}



/* TODO: Explain what this does*/
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



/**
	Gets the on-shell point k0 for different types of NN scattering
	@param k0:	On-shell point
	@param TLab: Kinetic energy for the projectile in the lab system
	@param TLabLength: Size of the TLab array
	@param tzChannel: Current tz channel
	@return			On-shell point k0
*/
__global__
void getk0(double* k0, double* TLab, int TLabLength, int tzChannel) {
	double k0Squared = 0;
	for (int i = 0; i < TLabLength; i++) {
		/* Proton-proton scattering */
		if (tzChannel == -1) {
			k0Squared = constants::protonMass * TLab[i] / 2;
		}
		/* Proton-neutron scattering (with proton as projectile, neutron as target) */
		else if (tzChannel == 0) {
			k0Squared = pow(constants::neutronMass, 2) * TLab[i] * (TLab[i]
				+ 2 * constants::protonMass) / ((pow(constants::protonMass
					+ constants::neutronMass, 2) + 2 * TLab[i] * constants::neutronMass));
		}
		/* Neutron-neutron scattering */
		else if (tzChannel == 1) {
			k0Squared = constants::neutronMass * TLab[i] / 2;
		}
		k0[i] = sqrtf(k0Squared);
	}
}



int solveLS() {
	/* Set up the quantum states by choosing ranges for the j and tz quantum numbers*/
	int jMin = 0;
	int jMax = 2;
	int tzMin = 0;
	int tzMax = 2;
	std::vector<QuantumState> basis = setupBasis(jMin, jMax, tzMin, tzMax);
	std::map<std::string, std::vector<QuantumState>> channels = setupNNChannels(basis);

	// TODO: Explain
	std::string key = "j:0 s:0 tz:0 pi:1"; // TODO: Looks like "magic numbers" for outside reader, explain this
	std::vector<QuantumState> channel = channels[key];
	if (channel.size() == 0) {
		std::cout << "Invalid key";
		abort();
	}
	int tzChannel = channel[0].state["tz"];

	/* Number of quadrature points, needed for array sizes and later the quadrature setup */
	constexpr int quadratureN = 5;

	/* All matrices and vectors have the same length/height; the number of quadrature points
	 * plus one (because of the on-shell point). Furthermore, in the uncoupled case there is 
	 * is only one phase shift whereas in the uncoupled case there are two phase shifts and 
	 * one mixing angle. */
	int matLength;
	int phasesSize;
	bool coupled = isCoupled(channel);
	if (!coupled) {
		matLength = quadratureN + 1;
		phasesSize = 1;
	}
	else {
		/* Let matLength be two times as big to facilitate calculations later */
		matLength = 2 * (quadratureN + 1);
		phasesSize = 3;
	}

	/* Prepare to generate TLab [Mev] */
	constexpr double TLabMin = 1;	// Minimum energy
	constexpr double TLabMax = 300; // Threshold energy for pion creation
	constexpr int TLabLength = 100; // Number of energies to generate
	constexpr double TLabIncr = (TLabMax - TLabMin + 1) / TLabLength;

	/* Allocate memory on the host */
	cuDoubleComplex* F_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	cuDoubleComplex* G0_h = new cuDoubleComplex[matLength * TLabLength];
	double* k_h = new double[quadratureN];
	double* k0_h = new double[TLabLength];
	cuDoubleComplex* phases_h = new cuDoubleComplex[phasesSize * TLabLength];
	double* sum_h = new double[TLabLength];
	cuDoubleComplex* T_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	double* TLab_h = new double[TLabLength];
	cuDoubleComplex* V_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	cuDoubleComplex* VG_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	double* w_h = new double[quadratureN];

	/* Generate different experimental kinetic energies [MeV] */
	for (int i = 0; i < TLabLength; i++) {
		TLab_h[i] = i * TLabIncr + TLabMin;
		TLab_h[i] = TLabMin + i * TLabIncr;
		//printf("Tlab[%i] = %.4e", i, TLab_h[i]);
	}

	/* Set up the quadrature points k with weights w */
	constexpr double scale = 100; // TODO: Explain how this is chosen
	gaussLegendreInfMesh(k_h, w_h, quadratureN, scale);

	/* Declare device variables to be able to allocate them on the device */
	cuDoubleComplex* F_d;
	cuDoubleComplex* G0_d;
	double* k_d;
	double* k0_d;
	cuDoubleComplex* phases_d;
	double* sum_d;
	cuDoubleComplex* T_d;
	double* TLab_d;
	cuDoubleComplex* V_d;
	cuDoubleComplex* VG_d;
	double* w_d;

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

	/* Allocate memory on the device */
	cudaMalloc((void**)&F_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&G0_d, matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&k_d, quadratureN * sizeof(double));
	cudaMalloc((void**)&k0_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&phases_d, phasesSize * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&sum_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&T_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&TLab_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&V_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&VG_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&w_d, quadratureN * sizeof(double));

	/* Copy host variables to device variables */
	cudaMemcpy(F_d, F_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(G0_d, G0_h, matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(k_d, k_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(k0_d, k0_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(phases_d, phases_h, phasesSize * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(sum_d, sum_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(T_d, T_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(TLab_d, TLab_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(VG_d, VG_h, matLength* matLength* TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);

	// TODO: Explain this
	dim3 threadsPerBlock(matLength, matLength,TLabLength); // Block size
	dim3 blocksPerGrid(1,1,1); // Grid size
	if (matLength * matLength > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = ceil(double(matLength) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(matLength) / double(threadsPerBlock.y));
	}
	if(TLabLength>512){
		threadsPerBlock.z = 512;
		blocksPerGrid.z = ceil(double(TLabLength) / double(threadsPerBlock.z));
	}

	/* Get the on-shell points for different TLab with parallellization */
	getk0 <<<1,1>>>(k0_d, TLab_d, TLabLength, tzChannel);

	/* Use k0 to generate different potentials on the CPU. The CPU generated potentials are
	 * then sent to the GPU as an array. */
	cudaMemcpy(k0_h, k0_d, TLabLength * sizeof(double), cudaMemcpyDeviceToHost);
	potential(V_h, channel, k_h, TLab_h, k0_h, quadratureN, TLabLength, coupled, matLength);
	cudaMemcpy(V_d, V_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);


	double mu = getReducedMass(channel);

	/* Call kernels on GPU */
	setupG0VectorSum <<<1,1>>> (sum_d, k0_d, quadratureN, TLabLength, k_d, w_d);
	setupG0Vector <<<threadsPerBlock, blocksPerGrid >>> (G0_d, k_d, w_d, k0_d, sum_d, quadratureN, matLength, TLabLength, mu, coupled);
	/* Setup the VG kernel and, at the same time, the F matrix */
	setupVGKernel <<<threadsPerBlock, blocksPerGrid >>> (VG_d, V_d, G0_d, F_d, k_d, w_d, k0_d, quadratureN, matLength, TLabLength, mu, coupled);

	/* Solve the equation FT = V with cuBLAS */
	computeTMatrixCUBLAS(T_d, F_d, V_d, matLength, TLabLength);

	// cudaMemcpy(T_d, T_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	/* TODO: Explain this */
	dim3 threadsPerBlockPhaseShift(TLabLength); //Blocksize
	dim3 blocksPerGridPhaseShift(1); //Gridsize

	/* Computes the phase shifts for the given T-matrix*/
	computePhaseShifts <<<threadsPerBlock.z, blocksPerGrid.z>>> (phases_d, T_d, k0_d, quadratureN, mu, coupled, TLabLength, matLength);

	/* Make sure all kernels are done before accessing device variables from host */
	cudaDeviceSynchronize();

	/* Copy (relevant) device variables to host variables */
	// cudaMemcpy(T_h, T_d, matLength * matLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(phases_h, phases_d, phasesSize * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	// cudaMemcpy(k_h, k_d, quadratureN * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(w_h, w_d, quadratureN * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(k0_h, k0_d, TLabLength * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(V_h, V_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	// cudaMemcpy(F_h, F_d, matLength * matLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	// cudaMemcpy(VG_h, VG_d, matLength * matLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);


	//for (int i = 0; i < TLabLength; i++) {
	//	if (coupled) {
	//		for (int j = 0; j < phasesSize; ++j) {
	//			printf("\nTLab = %f", TLab_h[i]);
	//			printf("\nReal(phases[%i]) = %.10e", j, cuCreal(phases_h[j + i * phasesSize]));
	//			printf("\nImag(phases[%i]) = %.10e", j, cuCimag(phases_h[j + i * phasesSize]));
	//			printf("\n");
	//		}
	//	}
	//	else {
	//		printf("\nTLab = %f", TLab_h[i]);
	//		printf("\nReal(phase) = %.10e", cuCreal(phases_h[i]));
	//		printf("\nImag(phase) = %.10e", cuCimag(phases_h[i]));
	//		printf("\n");
	//	}
	//}


	/*
		for (int i = 0; i < TLabLength; i = i+20) {
		if (coupled) {
			for (int j = 0; j < phasesSize; ++j) {
				printf("\nTLab = %f", TLab_h[i]);
				printf("\nReal(phases[%i]) = %.10e", j, cuCreal(phases_h[j + i * phasesSize]));
				printf("\nImag(phases[%i]) = %.10e", j, cuCimag(phases_h[j + i * phasesSize]));
				printf("\n");
			}
		}
		else {
			printf("\nTLab = %f", TLab_h[i]);
			printf("\nReal(phase) = %.10e", cuCreal(phases_h[i]));
			printf("\nImag(phase) = %.10e", cuCimag(phases_h[i]));
			printf("\n");
		}
	}
	*/

	printf("\none run\n");





	/* Free allocated host memory */
	delete[] F_h;
	delete[] G0_h;
	delete[] k_h;
	delete[] k0_h;
	delete[] phases_h;
	delete[] sum_h;
	delete[] T_h;
	delete[] TLab_h;
	delete[] V_h;
	delete[] VG_h;
	delete[] w_h;

	/* Free allocated device memory */
	cudaFree(F_d);
	cudaFree(G0_d);
	cudaFree(k0_d);
	cudaFree(k_d);
	cudaFree(phases_d);
	cudaFree(sum_d);
	cudaFree(T_d);
	cudaFree(TLab_d);
	cudaFree(V_d);
	cudaFree(VG_d);
	cudaFree(w_d);

	return 0;
}