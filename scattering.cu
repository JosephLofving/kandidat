#include "scattering.h"


/**
	Sets up a complex vector needed to solve the T matrix equation.
	@param k:	Quadrature points
	@param w:	Weights for quadrature points
	@param k0:	On-shell-point
	@return		G0 vector
*/

//const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);


template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}

__device__
void computeTMatrixCUBLAS(cuDoubleComplex* h_Tarray,
	cuDoubleComplex* h_Farray,
	cuDoubleComplex* h_Varray,
	int N, int mSize) {

	const int batchSize{1};
	if (true) {
		const int batchSize = mSize;
	}

	// cuBLAS variables
	cublasStatus_t status;
	cublasHandle_t handle;

	// Host variables
	size_t matSize = N * N * sizeof(cuDoubleComplex); //denna ska la dock vara mSize * mSize, ty matriserna som löses har den storleken.

	// cuDoubleComplex* h_Farray;
	cuDoubleComplex* h_Fptr_array[batchSize];

	// cuDoubleComplex* h_Varray;
	cuDoubleComplex* h_Vptr_array[batchSize];

	// Device variables
	cuDoubleComplex* d_Farray;
	cuDoubleComplex** d_Fptr_array;

	cuDoubleComplex* d_Varray;
	cuDoubleComplex** d_Vptr_array;

	int* d_pivotArray;
	int* d_trfInfo;
	int d_trsInfo;

	// Initialize cuBLAS
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("> ERROR: cuBLAS initialization failed\n");
		//return (EXIT_FAILURE); /return type är void, inte exit_failure
	}

	// Allocate memory for host variables
	// h_Farray = (cuDoubleComplex*)xmalloc(batchSize * matSize);
	// h_Varray = (cuDoubleComplex*)xmalloc(batchSize * matSize);

	// Allocate memory for device variables

	checkCudaErrors(cudaMalloc((void**)&d_Farray, batchSize * matSize));
	checkCudaErrors(cudaMalloc((void**)&d_Varray, batchSize * matSize));
	checkCudaErrors(
		cudaMalloc((void**)&d_pivotArray, N * batchSize * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_trfInfo, batchSize * sizeof(int)));
	checkCudaErrors(
		cudaMalloc((void**)&d_Fptr_array, batchSize * sizeof(cuDoubleComplex*)));
	checkCudaErrors(
		cudaMalloc((void**)&d_Vptr_array, batchSize * sizeof(cuDoubleComplex*)));

	// for (int i = 0; i < batchSize; i++) {
	//     initSetAMatrix(h_Farray + (i * N*N), (double)(i+1)); // Create matrices scaled by factors 1, 2, ...
	// }

	// printMatrix(h_Farray, N, N);

	// for (int i = 0; i < batchSize; i++) {
	//     initSetBMatrix(h_Varray + (i * N*N), (double)(i+1)); // Create matrices scaled by factors 1, 2, ...
	// }

	// printMatrix(h_Varray, N, N);

	// Copy data to device from host
	checkCudaErrors(cudaMemcpy(d_Farray, h_Farray, batchSize * matSize,
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Varray, h_Varray, batchSize * matSize,
		cudaMemcpyHostToDevice));

	// Create pointer array for matrices
	for (int i = 0; i < batchSize; i++) h_Fptr_array[i] = d_Farray + (i * N * N);
	for (int i = 0; i < batchSize; i++) h_Vptr_array[i] = d_Varray + (i * N * N);

	// Copy pointer array to device memory
	checkCudaErrors(cudaMemcpy(d_Fptr_array, h_Fptr_array,
		batchSize * sizeof(cuDoubleComplex*),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Vptr_array, h_Vptr_array,
		batchSize * sizeof(cuDoubleComplex*),
		cudaMemcpyHostToDevice));

	// Perform LU decomposition
	status = cublasZgetrfBatched(handle, N, d_Fptr_array, N, d_pivotArray,
		d_trfInfo, batchSize);

	// Calculate the T matrix
	status = cublasZgetrsBatched(handle, CUBLAS_OP_N, N, N, d_Fptr_array, N,
		d_pivotArray, d_Vptr_array, N, &d_trsInfo,
		batchSize);

	// Copy data to host from device
	checkCudaErrors(cudaMemcpy(h_Tarray, d_Varray, batchSize * matSize,
		cudaMemcpyDeviceToHost));

	// printMatrix(h_Varray, N, N);

	// Free device variables
	checkCudaErrors(cudaFree(d_Fptr_array));
	checkCudaErrors(cudaFree(d_Vptr_array));
	checkCudaErrors(cudaFree(d_trfInfo));
	checkCudaErrors(cudaFree(d_pivotArray));
	checkCudaErrors(cudaFree(d_Farray));
	checkCudaErrors(cudaFree(d_Varray));

	// Free host variables
	// if (h_Farray) free(h_Farray);
	// if (h_Varray) free(h_Varray);

	// Destroy cuBLAS handle
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("> ERROR: cuBLAS uninitialization failed...\n");
	}
}








__device__
cuDoubleComplex operator+(cuDoubleComplex A, cuDoubleComplex B) {
	cuDoubleComplex result = make_cuDoubleComplex(cuCreal(A)+cuCreal(B), cuCimag(A)+cuCimag(B));
	return result;
}

__device__
cuDoubleComplex operator-(cuDoubleComplex A, cuDoubleComplex B) {
	cuDoubleComplex result = make_cuDoubleComplex(cuCreal(A) - cuCreal(B), cuCimag(A) - cuCimag(B));
	return result;
}

__device__
cuDoubleComplex operator-(double a, cuDoubleComplex A) {
	cuDoubleComplex result = cuCsub(make_cuDoubleComplex(a, 0), A);
	return result;
}

__device__
cuDoubleComplex operator-(cuDoubleComplex A, double a) {
	cuDoubleComplex result = cuCsub(A, make_cuDoubleComplex(a, 0));
	return result;
}

__device__
cuDoubleComplex operator*(double scalar, cuDoubleComplex A) {
	cuDoubleComplex result = make_cuDoubleComplex(scalar * cuCreal(A), scalar * cuCimag(A));
	return result;
}

__device__
cuDoubleComplex operator*(cuDoubleComplex A, double scalar) {
	return scalar * A;
}

__device__
cuDoubleComplex operator*(cuDoubleComplex A, cuDoubleComplex B) {
	cuDoubleComplex realProd = cuCreal(A) * B;
	cuDoubleComplex imagProd = cuCimag(A) * B;
	cuDoubleComplex result = cuCadd(realProd, imagProd);
	return result;
}

__device__
cuDoubleComplex operator/(cuDoubleComplex A, cuDoubleComplex B) {
	return cuCdiv(A, B);
}

__device__
cuDoubleComplex operator/(cuDoubleComplex A, double a) {
	return cuCdiv(A, make_cuDoubleComplex(a, 0));
}

__device__
cuDoubleComplex operator/(double a, cuDoubleComplex A) {
	return cuCdiv(make_cuDoubleComplex(a, 0), A);
}

__device__
cuDoubleComplex logCudaComplex(cuDoubleComplex argument) {
	double x = cuCreal(argument);
	double y = cuCimag(argument);
	double real = logf(sqrtf(x * x + y * y));
	double imag = atan2f(y, x);
	cuDoubleComplex result = make_cuDoubleComplex(real, imag);
	return result;
}

__device__
double signCuda(double argument) {
	if (argument > 0) return 1;
	else if (argument == 0) return 0;
	else return -1;
}

__device__
cuDoubleComplex sqrtCudaComplex(cuDoubleComplex argument) {
	double x = cuCreal(argument);
	double y = cuCimag(argument);
	double real = sqrtf((sqrtf(x * x + y * y) + x) / 2);
	double imag = signCuda(y) * sqrtf((sqrtf(x * x + y * y) - x) / 2);
	return make_cuDoubleComplex(real, imag);
}

__device__
cuDoubleComplex atanCudaComplex(cuDoubleComplex argument) {
	cuDoubleComplex numerator = cuCadd(make_cuDoubleComplex(1, 0), cuCmul(make_cuDoubleComplex(0, 1), argument));
	cuDoubleComplex denominator = cuCsub(make_cuDoubleComplex(1, 0), cuCmul(make_cuDoubleComplex(0, 1), argument));
	cuDoubleComplex logOfStuff = logCudaComplex(cuCdiv(numerator, denominator));
	cuDoubleComplex result = cuCmul(make_cuDoubleComplex(0, -0.5), logOfStuff);
	return result;
}

__device__
cuDoubleComplex asinCudaComplex(cuDoubleComplex argument) {
	const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
	return I * logCudaComplex(sqrtCudaComplex(1 - argument * argument) - I * argument);
}

__device__
cuDoubleComplex expCudaComplex(cuDoubleComplex argument) {
	double x = cuCreal(argument);
	double y = cuCimag(argument);
	cuDoubleComplex trig = make_cuDoubleComplex(cosf(y), sinf(y));
	cuDoubleComplex result = make_cuDoubleComplex(expf(x), 0) * trig;
	return result;
}

__device__
cuDoubleComplex sinCudaComplex(cuDoubleComplex argument) {
	const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
	return (expCudaComplex(I * argument) - expCudaComplex(-1.0 * I * argument)) / 2;
}

__device__
cuDoubleComplex tanCudaComplex(cuDoubleComplex argument) {
	const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
	cuDoubleComplex numerator = I * (expCudaComplex(-1.0 * I * argument) - expCudaComplex(I * argument));
	cuDoubleComplex denominator = expCudaComplex(-1.0 * I * argument) + expCudaComplex(I * argument);
	return numerator / denominator;
}




__device__
void setupG0Vector(cuDoubleComplex* G0,
	double* k,
	double* w,
	double k0,
	int quadratureN,
	double mu,
	bool coupled) {

	double twoMu = (2.0 * mu);
	double twoOverPi = (2.0 / constants::pi);
	double sum = 0;
	for (int i = 0; i < quadratureN; i++) {
		G0[i] = make_cuDoubleComplex(twoOverPi * twoMu * k[i] * k[i] * w[i] / (k0 * k0 - k[i] * k[i]), 0);
		sum += w[i] / (k0 * k0 - k[i] * k[i]);

		/* If coupled, append G0 to itself to facilitate calculations.
		 * This means the second half of G0 is a copy of the first. */
		if (coupled) {
			G0[quadratureN + 1 + i] = G0[i];
		}
	}

	/* Assign the last element of D */
	G0[quadratureN] = make_cuDoubleComplex(-twoOverPi * twoMu * k0 * k0 * sum, -twoMu * k0);
	if (coupled) {
		G0[2 * (quadratureN + 1) - 1] = G0[quadratureN];
	}
}

/**
	Multiplies the potential matrix elements with the G0 vector elements.

	@param channel: Scattering channel
	@param key:		Channel name
	@param V:		Potential matrix
	@param k:		Quadrature points
	@param w:		Weights for quadrature points
	@param k0:		The on-shell-point
	@return			VG kernel
*/
__device__
void setupVGKernel(cuDoubleComplex* VG,
	cuDoubleComplex* V,
	cuDoubleComplex* G0,
	cuDoubleComplex* F,
	double* k,
	double* w,
	double k0,
	int quadratureN,
	int matSize,
	double mu,
	bool coupled) {

	setupG0Vector(G0, k, w, k0, quadratureN, mu, coupled);
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < matSize && column < matSize) {
		VG[row + column * matSize] = cuCmul(V[row + column * matSize], G0[column]);
		//cuDoubleComplex test = cuCmul(V[row + column * matSize], G0[column]);
		cuDoubleComplex testG0 = G0[column];
		cuDoubleComplex testV = V[row + column * matSize];
		//printf("\nV = %f, %f", cuCreal(testV), cuCimag(testV));
		if (row == column) {
			F[row + row * matSize] = cuCadd(make_cuDoubleComplex(1, 0), cuCmul(make_cuDoubleComplex(-1, 0), VG[row + row * matSize])); // Diagonal element
		}
		else {
			F[row + column * matSize] = cuCmul(make_cuDoubleComplex(-1, 0), VG[row + column * matSize]);
		}

	}

}



	//for (int row = 0; row < matSize; row++) {
	//	for (int column = 0; column < matSize; column++) {
	//		/* Create VG by using VG[i,j] = V[i,j] * G[j] */
	//		VG[row + column * matSize] = cuCmul(V[row + column * matSize], G0[column]);

	//		/* At the same time, create F = delta_ij - VG_ij for computeTMatrix*/
	//		if (row != column) {
	//			F[row + column * matSize] = cuCmul(make_cuDoubleComplex(-1, 0), VG[row + column * matSize]);
	//		}
	//	}
	//	F[row + row * matSize] = cuCadd(make_cuDoubleComplex(1, 0), cuCmul(make_cuDoubleComplex(-1, 0), VG[row + row * matSize])); // Diagonal element
	//}
//}




/**
	Computes the T-matrix from the equation [F][T] = [V]

	@param channel: Scattering channel
	@param key:		Channel name
	@param V:		Potential matrix
	@param k:		Quadrature points
	@param w:		Weights for quadrature points
	@param k0:		On-shell-point
	@return			T matrix
*/
__global__
void computeTMatrix(cuDoubleComplex** T,
	cuDoubleComplex** V,
	cuDoubleComplex** G0,
	cuDoubleComplex** VG,
	cuDoubleComplex** F,
	cuDoubleComplex** phases,
	double* k,
	double* w,
	double* k0,
	int quadratureN,
	int matSize,
	int TLabLength,
	double mu,
	bool coupled) {

	/* Setup the VG kernel and, at the same time, the F matrix */
	for (int i = 0; i < TLabLength; i++) {
		setupVGKernel(VG[i], V[i], G0[i], F[i], k, w, k0[i], quadratureN, matSize, mu, coupled);

		/* Solve the equation FT = V with cuBLAS */
		computeTMatrixCUBLAS(T[i], F[i], V[i], quadratureN, matSize);

		/* Computes the phase shifts for the given T-matrix*/
		computePhaseShifts(phases[i], T[i], k0[i], quadratureN, mu, coupled);
	}

}



/* TODO: Explain theory for this. */
__device__
void blattToStapp(cuDoubleComplex* phases, cuDoubleComplex deltaMinusBB, cuDoubleComplex deltaPlusBB, cuDoubleComplex twoEpsilonJBB) {
	cuDoubleComplex twoEpsilonJ = asinCudaComplex(sinCudaComplex(twoEpsilonJBB) * sinCudaComplex(deltaMinusBB - deltaPlusBB));

	phases[0] = 0.5 * (deltaPlusBB + deltaMinusBB + asinCudaComplex(tanCudaComplex(twoEpsilonJ) / tanCudaComplex(twoEpsilonJBB))) * constants::rad2deg;
	phases[1] = 0.5 * (deltaPlusBB + deltaMinusBB - asinCudaComplex(tanCudaComplex(twoEpsilonJ) / tanCudaComplex(twoEpsilonJBB))) * constants::rad2deg;
	phases[2] = 0.5 * twoEpsilonJ * constants::rad2deg;
}


/**
	Computes the phase shift for a given channel and T matrix.

	@param channel: Scattering channel
	@param key:		Channel name
	@param k0:		On-shell-point
	@param T:		T matrix
	@return			Complex phase shifts
*/
__device__
void computePhaseShifts(cuDoubleComplex* phases,
					    cuDoubleComplex* T,
						double k0,
						int quadratureN,
						double mu,
						bool coupled) {

	double rhoT =  2 * mu * k0; // Equation (2.27) in the theory
	const cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);

	// TODO: Explain theory for the phase shift for the coupled state
	if (coupled) {
		int N = quadratureN;
		cuDoubleComplex T11 = T[(N) + (N * N)]; //row + column * size
		cuDoubleComplex T12 = T[(2 * N + 1) + (N * N)];
		cuDoubleComplex T22 = T[(2 * N + 1) + (N * (2 * N + 1))];

		//Blatt - Biedenharn(BB) convention
		cuDoubleComplex twoEpsilonJBB = atanCudaComplex(cuCdiv(cuCmul(make_cuDoubleComplex(2.0, 0), T12), cuCsub(T11, T22)));
		cuDoubleComplex deltaPlusBB{ -0.5 * I * logCudaComplex(1.0 - I * rhoT * (T11 + T22) + I * rhoT * (2.0 * T12) / sinCudaComplex(twoEpsilonJBB)) };
		cuDoubleComplex deltaMinusBB{ -0.5 * I * logCudaComplex(1.0 - I * rhoT * (T11 + T22) - I * rhoT * (2.0 * T12) / sinCudaComplex(twoEpsilonJBB)) };
		blattToStapp(phases, deltaMinusBB, deltaPlusBB, twoEpsilonJBB);

	}
	/* The uncoupled case completely follows equation (2.26). */
	else {
		double T0 = cuCreal(T[(quadratureN) + (quadratureN * quadratureN)]); //Farligt, detta element kanske inte �r helt reellt. Dock var koden d�lig f�rut is�fall.
		cuDoubleComplex argument = make_cuDoubleComplex(1, -2.0 * rhoT * T0);
		cuDoubleComplex swappedLog = make_cuDoubleComplex(cuCimag(logCudaComplex(argument)), cuCreal(logCudaComplex(argument)));
		cuDoubleComplex delta = cuCmul(make_cuDoubleComplex(-0.5 * constants::rad2deg, 0), swappedLog);
		phases[0] = delta;
	}
}