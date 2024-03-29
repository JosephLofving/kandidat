#include "solveLS.h"



/**
	Gets the reduced mass by checking the isospin channel, which determines the type of NN scattering
	@param channel:	Scattering channel
	@return			Reduced mass
*/
double getReducedMass(std::vector<QuantumState> channel) {
	double mu = 0;
	int TzChannel = channel[0].state["Tz"];
	/* Proton-proton scattering */
	if (TzChannel == -1)
		mu = constants::protonMass / 2;
	/* Proton-neutron scattering */
	else if (TzChannel == 0)
		mu = constants::nucleonReducedMass;
	/* Neutron-neutron scattering */
	else if (TzChannel == 1)
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
void setupDVectorSum(
	double* sum,
	double* k0,
	int quadratureN,
	int TLabLength,
	double* k,
	double* w) {

	int slice = blockIdx.x * blockDim.x + threadIdx.x;
	if (slice < TLabLength) {
		sum[slice] = 0;
		for (int column = 0; column < quadratureN; ++column) {
			sum[slice] += w[column] / (k0[slice] * k0[slice] - k[column] * k[column]);
		}
	}
}



/**
	Gets the on-shell point k0 for different types of NN scattering
	@param k0:	On-shell point
	@param TLab: Kinetic energy for the projectile in the lab system
	@param TLabLength: Size of the TLab array
	@param TzChannel: Current Tz channel
	@return			On-shell point k0
*/
__global__
void getk0(double* k0, double* TLab, int TLabLength, int TzChannel) {
	int slice = blockIdx.x * blockDim.x + threadIdx.x;
		//Hardcode for Tz=0
		if (slice < TLabLength) {
			k0[slice] = sqrt(pow(constants::neutronMass, 2) * TLab[slice] * (TLab[slice]
				+ 2 * constants::protonMass) / ((pow(constants::protonMass
					+ constants::neutronMass, 2) + 2 * TLab[slice] * constants::neutronMass)));
		}
	}



int main() {

	using microseconds = std::chrono::microseconds;
	auto start = std::chrono::high_resolution_clock::now();

	/*We define our parameters here*/

	constexpr double TLabMin = 1;	// Minimum energy
	constexpr double TLabMax = 300; // Threshold energy for pion creation
	constexpr int TLabLength = 4096; // Number of energies to generate

	constexpr int quadratureN = 40;

	/*End of defining parameters*/


	/*We initialize CUDA and cuBLAS here*/

	// Initialize CUDA
	auto startcudafree0 = std::chrono::high_resolution_clock::now();
	cudaFree(0);
	auto stopcudafree0 = std::chrono::high_resolution_clock::now();


	// cuBLAS variables
	cublasStatus_t status;
	cublasHandle_t handle;

	// Initialize cuBLAS
	auto cublasCreate_start = std::chrono::high_resolution_clock::now();
	status = cublasCreate(&handle);
	auto cublasCreate_stop = std::chrono::high_resolution_clock::now();

	/*End of initializing CUDA and cuBlas*/



	/* Set up the quantum states by choosing ranges for the J and Tz quantum numbers*/
	int JMin = 0;
	int JMax = 2;
	int TzMin = 0;
	int TzMax = 2;
	std::vector<QuantumState> basis = setupBasis(JMin, JMax, TzMin, TzMax);
	std::map<std::string, std::vector<QuantumState>> channels = setupNNChannels(basis);

	// TODO: Explain
	std::string key = "J:0 S:0 Tz:0 pi:1"; // TODO: Looks like "magic numbers" for outside reader, explain this
	std::vector<QuantumState> channel = channels[key];
	if (channel.size() == 0) {
		std::cout << "Invalid key";
		abort();
	}
	int TzChannel = channel[0].state["Tz"];

	/* Number of quadrature points, needed for array sizes and later the quadrature setup */

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
	constexpr double TLabIncr = (TLabMax - TLabMin + 1) / TLabLength;

	auto startAllocateHost = std::chrono::high_resolution_clock::now();
	/* Allocate memory on the host */
	double* k_h = new double[quadratureN];
	double* k0_h = new double[TLabLength];
	cuDoubleComplex* phases_h = new cuDoubleComplex[phasesSize * TLabLength];
	double* TLab_h = new double[TLabLength];
	cuDoubleComplex* V_h = new cuDoubleComplex[matLength * matLength * TLabLength];
	double* w_h = new double[quadratureN];
	auto stopAllocateHost = std::chrono::high_resolution_clock::now();

	/* Generate different experimental kinetic energies [MeV] */
	for (int i = 0; i < TLabLength; i++) {
		TLab_h[i] = i * TLabIncr + TLabMin;
		TLab_h[i] = TLabMin + i * TLabIncr;
		//printf("Tlab[%i] = %.4e", i, TLab_h[i]);
	}

	/* Set up the quadrature points k with weights w */
	constexpr double scale = 100; // TODO: Explain how this is chosen
	auto startKvadratur = std::chrono::high_resolution_clock::now();
	gaussLegendreInfMesh(k_h, w_h, quadratureN, scale);
	auto stopKvadratur = std::chrono::high_resolution_clock::now();

	/* Declare device variables to be able to allocate them on the device */
	cuDoubleComplex* F_d;
	cuDoubleComplex* D_d;
	double* k_d;
	double* k0_d;
	cuDoubleComplex* phases_d;
	double* sum_d;
	cuDoubleComplex* T_d;
	double* TLab_d;
	cuDoubleComplex* V_d;
	cuDoubleComplex* VD_d;
	double* w_d;


	auto startAllocateDevice = std::chrono::high_resolution_clock::now();

	/* Allocate memory on the device */
	cudaMalloc((void**)&F_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&D_d, matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&k_d, quadratureN * sizeof(double));
	cudaMalloc((void**)&k0_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&phases_d, phasesSize * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&sum_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&T_d, matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&TLab_d, TLabLength * sizeof(double));
	cudaMalloc((void**)&V_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&VD_d, matLength * matLength * TLabLength * sizeof(cuDoubleComplex));
	cudaMalloc((void**)&w_d, quadratureN * sizeof(double));

	auto stopAllocateDevice = std::chrono::high_resolution_clock::now();

	auto startCopyHostToDevice = std::chrono::high_resolution_clock::now();
	/* Copy host variables to device variables */
	cudaMemcpy(k_d, k_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(TLab_d, TLab_h, TLabLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(w_d, w_h, quadratureN * sizeof(double), cudaMemcpyHostToDevice);
	auto stopCopyHostToDevice = std::chrono::high_resolution_clock::now();

	// TODO: Explain this
	dim3 threadsPerBlock(matLength, matLength, TLabLength); // Block size
	dim3 blocksPerGrid(1,1,1); // Grid size

	threadsPerBlock.x = 4;
	threadsPerBlock.y = 4;
	threadsPerBlock.z = 64;
	blocksPerGrid.x = ceil(double(matLength) / double(threadsPerBlock.x));
	blocksPerGrid.y = ceil(double(matLength) / double(threadsPerBlock.y));
	blocksPerGrid.z = ceil(double(TLabLength) / double(threadsPerBlock.z));


	int blockSize = 256;
	int numBlocks = (TLabLength + blockSize - 1) / blockSize;

	auto startGetk0 = std::chrono::high_resolution_clock::now();
	/* Get the on-shell points for different TLab with parallellization */
	getk0 <<<numBlocks, blockSize>>>(k0_d, TLab_d, TLabLength, TzChannel);
	auto stopGetk0 = std::chrono::high_resolution_clock::now();

	/* Use k0 to generate different potentials on the CPU. The CPU generated potentials are
	 * then sent to the GPU as an array. */
	cudaMemcpy(k0_h, k0_d, TLabLength * sizeof(double), cudaMemcpyDeviceToHost);

	auto startPotential = std::chrono::high_resolution_clock::now();
	potential(V_h, channel, k_h, TLab_h, k0_h, quadratureN, TLabLength, coupled, matLength);
	auto stopPotential = std::chrono::high_resolution_clock::now();

	cudaMemcpy(V_d, V_h, matLength * matLength * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);


	double mu = getReducedMass(channel);

	/* Call kernels on GPU */
	auto startDsum = std::chrono::high_resolution_clock::now();
	setupDVectorSum <<<numBlocks, blockSize >>> (sum_d, k0_d, quadratureN, TLabLength, k_d, w_d);
	auto stopDsum = std::chrono::high_resolution_clock::now();

	auto startSetupD = std::chrono::high_resolution_clock::now();
	setupDVector <<<blocksPerGrid, threadsPerBlock>>> (D_d, k_d, w_d, k0_d, sum_d, quadratureN, matLength, TLabLength, mu, coupled);
	auto stopSetupD = std::chrono::high_resolution_clock::now();

	auto startSetupVDKernel = std::chrono::high_resolution_clock::now();
	/* Setup the VD kernel and, at the same time, the F matrix */
	setupVDKernel <<<blocksPerGrid, threadsPerBlock>>> (T_d, VD_d, V_d, D_d, F_d, k_d, w_d, k0_d, quadratureN, matLength, TLabLength, mu, coupled);
	auto stopSetupVDKernel = std::chrono::high_resolution_clock::now();

	auto startcomputeTMatrixCUBLAS = std::chrono::high_resolution_clock::now();
	/* Solve the equation FT = V with cuBLAS */


	computeTMatrixCUBLAS(T_d, F_d, matLength, TLabLength, status, handle);
	auto stopcomputeTMatrixCUBLAS = std::chrono::high_resolution_clock::now();
	/* TODO: Explain this */

	/* Computes the phase shifts for the given T-matrix*/


	auto startcomputePhaseShifts = std::chrono::high_resolution_clock::now();
	computePhaseShifts <<<numBlocks, blockSize>>> (phases_d, T_d, k0_d, quadratureN, mu, coupled,
		TLabLength, matLength);
	auto stopcomputePhaseShifts = std::chrono::high_resolution_clock::now();
	/* Make sure all kernels are done before accessing device variables from host */

	cudaDeviceSynchronize();

	/* Copy (relevant) device variables to host variables */
	cudaMemcpy(phases_h, phases_d, phasesSize * TLabLength * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);



	// for (int i = 0; i < TLabLength; i++) {
	// 	if (coupled) {
	// 		for (int j = 0; j < phasesSize; ++j) {
	// 			printf("\nTLab = %f", TLab_h[i]);
	// 			printf("\nReal(phases[%i]) = %.10e", j, cuCreal(phases_h[j + i * phasesSize]));
	// 			printf("\nImag(phases[%i]) = %.10e", j, cuCimag(phases_h[j + i * phasesSize]));
	// 			printf("\n");
	// 		}
	// 	}
	// 	else {
	// 		printf("\nTLab = %f", TLab_h[i]);
	// 		printf("\nReal(phase) = %.10e", cuCreal(phases_h[i]));
	// 		printf("\nImag(phase) = %.10e", cuCimag(phases_h[i]));
	// 		printf("\n");
	// 	}
	// }


	/**
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

	auto freemem_start = std::chrono::high_resolution_clock::now();

	/* Free allocated host memory */
	delete[] k_h;
	delete[] k0_h;
	delete[] phases_h;
	delete[] TLab_h;
	delete[] V_h;
	delete[] w_h;

	/* Free allocated device memory */
	// cudaFree(F_d);
	cudaFree(D_d);
	cudaFree(k0_d);
	cudaFree(k_d);
	cudaFree(phases_d);
	cudaFree(sum_d);
	cudaFree(T_d);
	cudaFree(TLab_d);
	cudaFree(V_d);
	cudaFree(VD_d);
	cudaFree(w_d);

	auto freemem_end = std::chrono::high_resolution_clock::now();
	auto finish = std::chrono::high_resolution_clock::now();

	/*
	std::cout << "total: \t\t\t" << std::chrono::duration_cast<microseconds>(finish - start).count()<<"\n";

	std::cout << "cudafree(0): \t\t" << std::chrono::duration_cast<microseconds>(stopcudafree0 - startcudafree0).count() << "\n";

	std::cout << "cublasCreate:           " << std::chrono::duration_cast<microseconds>(cublasCreate_stop - cublasCreate_start).count() << "\n";

	std::cout << "allocate host: \t\t" << std::chrono::duration_cast<microseconds>(stopAllocateHost - startAllocateHost).count()<<"\n";

	std::cout << "kvadratur: \t\t" << std::chrono::duration_cast<microseconds>(stopKvadratur - startKvadratur).count()<<"\n";

	std::cout << "allocated device: \t" << std::chrono::duration_cast<microseconds>(stopAllocateDevice - startAllocateDevice).count()<<"\n";

	std::cout << "copy host to device: \t" << std::chrono::duration_cast<microseconds>(stopCopyHostToDevice - startCopyHostToDevice).count()<<"\n";

	std::cout << "getk0: \t\t\t" << std::chrono::duration_cast<microseconds>(stopGetk0 - startGetk0).count()<<"\n";

	std::cout << "potential: \t\t" << std::chrono::duration_cast<microseconds>(stopPotential - startPotential).count()<<"\n";

	std::cout << "Dsum: \t\t\t" << std::chrono::duration_cast<microseconds>(stopDsum - startDsum).count()<<"\n";

	std::cout << "setupD: \t\t" << std::chrono::duration_cast<microseconds>(stopSetupD - startSetupD).count()<<"\n";

	std::cout << "SetupVDKernel: \t\t" << std::chrono::duration_cast<microseconds>(stopSetupVDKernel - startSetupVDKernel).count()<<"\n";

	std::cout << "computeTMatrixCUBLAS: \t" << std::chrono::duration_cast<microseconds>(stopcomputeTMatrixCUBLAS - startcomputeTMatrixCUBLAS).count()<<"\n";

	std::cout << "computePhaseShifts: \t" << std::chrono::duration_cast<microseconds>(stopcomputePhaseShifts - startcomputePhaseShifts).count()<<"\n";

	std::cout << "Free memory: \t" << std::chrono::duration_cast<microseconds>(freemem_end - freemem_start).count() << "\n";
	*/

	std::cout << std::chrono::duration_cast<microseconds>(finish - start).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopcudafree0 - startcudafree0).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(cublasCreate_stop - cublasCreate_start).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopAllocateHost - startAllocateHost).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopKvadratur - startKvadratur).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopAllocateDevice - startAllocateDevice).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopCopyHostToDevice - startCopyHostToDevice).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopGetk0 - startGetk0).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopPotential - startPotential).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopDsum - startDsum).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopSetupD - startSetupD).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopSetupVDKernel - startSetupVDKernel).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopcomputeTMatrixCUBLAS - startcomputeTMatrixCUBLAS).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(stopcomputePhaseShifts - startcomputePhaseShifts).count() << ", ";

	std::cout << std::chrono::duration_cast<microseconds>(freemem_end - freemem_start).count() << "\n";

	return 0;
}