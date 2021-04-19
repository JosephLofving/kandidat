#include "scattering.h"


/**
	Sets up a complex vector needed to solve the T matrix equation.
	@param k:	Quadrature points
	@param w:	Weights for quadrature points
	@param k0:	On-shell-point
	@return		G0 vector
*/
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
void computeTMatrix(cuDoubleComplex* T,
	cuDoubleComplex* V,
	cuDoubleComplex* G0,
	cuDoubleComplex* VG,
	cuDoubleComplex* F,
	double* k,
	double* w,
	double* k0,
	int quadratureN,
	int matSize,
	double mu,
	bool coupled) {

	/* Setup the VG kernel and, at the same time, the F matrix */
	setupVGKernel(VG, V, G0, F, k, w, k0[0], quadratureN, matSize, mu, coupled);

	/* Solve the equation FT = V with cuBLAS */
	//T = solveMatrixEq(F, V); // old lapack function
	// cuBLAS function here, hopefully takes in a parameter cuDoubleComplex pointer T and changes it
}



/* TODO: Explain theory for this. */
//__device__
//std::vector<std::complex<double>> blattToStapp(std::complex<double> deltaMinusBB, std::complex<double> deltaPlusBB, std::complex<double> twoEpsilonJBB) {
//	std::complex<double> twoEpsilonJ = std::asin(std::sin(twoEpsilonJBB) * std::sin(deltaMinusBB - deltaPlusBB));
//
//	std::complex<double> deltaMinus = 0.5 * (deltaPlusBB + deltaMinusBB + std::asin(tan(twoEpsilonJ) / std::tan(twoEpsilonJBB))) * constants::rad2deg;
//	std::complex<double> deltaPlus = 0.5 * (deltaPlusBB + deltaMinusBB - std::asin(tan(twoEpsilonJ) / std::tan(twoEpsilonJBB))) * constants::rad2deg;
//	std::complex<double> epsilon = 0.5 * twoEpsilonJ * constants::rad2deg;
//
//	return { deltaMinus, deltaPlus, epsilon };
//}


/**
	Computes the phase shift for a given channel and T matrix.

	@param channel: Scattering channel
	@param key:		Channel name
	@param k0:		On-shell-point
	@param T:		T matrix
	@return			Complex phase shifts
*/
//__global__
//void computePhaseShifts(cuDoubleComplex* phases, 
//					    cuDoubleComplex* T, 
//						double* k0, 
//						int quadratureN, 
//						double mu, 
//						bool coupled) {
//	
//	double rhoT =  2 * mu * k0; // Equation (2.27) in the theory
//
//	// TODO: Explain theory for the phase shift for the coupled state
//	if (coupled) {
//		/*int N = quadratureN;
//		cuDoubleComplex T11 = T[(N) + (N * N)]; //row + column * size
//		cuDoubleComplex T12 = T[(2 * N + 1) + (N * N)];
//		cuDoubleComplex T22 = T[(2 * N + 1) + (N * (2 * N + 1))];
//
//		//Blatt - Biedenharn(BB) convention
//		std::complex<double> twoEpsilonJBB{ std::atan(2.0 * T12 / (T11 - T22)) };
//		std::complex<double> deltaPlusBB{ -0.5 * I * std::log(1.0 - I * rhoT * (T11 + T22) + I * rhoT * (2.0 * T12) / std::sin(twoEpsilonJBB)) };
//		std::complex<double> deltaMinusBB{ -0.5 * I * std::log(1.0 - I * rhoT * (T11 + T22) - I * rhoT * (2.0 * T12) / std::sin(twoEpsilonJBB)) };
//
//		std::vector<std::complex<double>> phasesAppend{ blattToStapp(deltaMinusBB, deltaPlusBB, twoEpsilonJBB) };
//
//		phases.push_back(phasesAppend[0]);
//		phases.push_back(phasesAppend[1]);
//		phases.push_back(phasesAppend[2]); 
//
//		*/
//		//avkommenterade för de ger error vid icke kopplad kompilering. Avkommentera och fixa.
//	}
//	/* The uncoupled case completely follows equation (2.26). */
//	else {
//		double T0 = cuCreal(T[(quadratureN) + (quadratureN * quadratureN)]); //Farligt, detta element kanske inte är helt reellt. Dock var koden dålig förut isåfall.
//		cuDoubleComplex argument = make_cuDoubleComplex(1, -2.0 * rhoT * T0);
//		cuDoubleComplex delta = (-0.5 * I) * logf(argument) * constants::rad2deg;
//
//		phases.push_back(delta);
//	}
//}