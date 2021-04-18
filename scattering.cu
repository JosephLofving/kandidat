#include "scattering.h"





/** 
	Checks if the state is coupled or not. 
	@param channel: Scattering channel
	@return			True if coupled, false if not
*/
bool isCoupled(std::vector<QuantumState> channel) {
	return !(channel.size() == 1); // If there is only one channel the state is uncoupled, otherwise there are four channels and the state is coupled.
}


/** 
	Sets up a complex vector needed to solve the T matrix equation. 
	@param k:	Quadrature points
	@param w:	Weights for quadrature points
	@param k0:	On-shell-point
	@return		G0 vector
*/
__device__
cuDoubleComplex* setupG0Vector(double mu, double* k, double* w, double k0, int quadratureN) {
	cuDoubleComplex* D = new cuDoubleComplex[quadratureN + 1]; // should this not be deleted somewhere..?

	double twoMu = (2.0 * mu);
	double twoOverPi = (2.0 / constants::pi);
	double sum = 0;

	for (int i = 0; i < quadratureN; i++) {
		D[i] = make_cuDoubleComplex(-twoOverPi * twoMu * pow(k[i], 2) * w[i] / (pow(k0, 2) - pow(k[i], 2)), 0);
		sum += w[i] / (pow(k0, 2) - pow(k[i], 2));														
	}

	D[quadratureN] = make_cuDoubleComplex(twoOverPi * twoMu * pow(k0, 2) * sum, twoMu * k0);


	return D;
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
cuDoubleComplex* setupVGKernel(double mu, bool coupled, cuDoubleComplex** V, double* k, double* w, double k0, int quadratureN, int matSize) {
	
	cuDoubleComplex* G0 = setupG0Vector(mu, k, w, k0, quadratureN);

	/* If coupled, append G0 to itself to facilitate calculations. This means the second half of G0 is a copy of the first. */
	if (coupled) {
		cuDoubleComplex* G0Coupled = new cuDoubleComplex[2*(quadratureN + 1)];
		for (int i = 0; i < quadratureN + 1; ++i) {
			G0Coupled[i] = G0[i];
			G0Coupled[quadratureN + 1 + i] = G0[i];
		}
		cuDoubleComplex* G0 = G0Coupled;
	}

	/* Create VG by using VG[i,j] = V[i,j] * G[j] */
	cuDoubleComplex* VElement = new cuDoubleComplex[1];
	for (int row = 0; row < matSize; row++) {
		for (int column = 0; column < matSize; column++) {
			//VG[row + column * matSize] = make_cuDoubleComplex(cuCreal(cuCmul(V[row + column * matSize], G0[column])),
			//												 cuCimag(cuCmul(V[row + column * matSize], G0[column])));
			VG[row + column * matSize] = cuCmul(V[row + column * matSize], G0[column]);
		}
	}

	//for (int i = 0; i < matSize * matSize; i += 100) {
	//	std::cout << VG.contents[i].real() << std::endl;
	//}
}


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
void computeTMatrix(cuDoubleComplex* T, cuDoubleComplex** VMatrix, cuDoubleComplex* VG, double* k, double* w, double* k0, int quadratureN, int matSize, double mu, bool coupled)  {
	cuDoubleComplex* VG = setupVGKernel(mu, coupled, VMatrix, k, w, k0_d, quadratureN, matSize);

	/* F = delta_ij - VG_ij */
	cuDoubleComplex* F = new cuDoubleComplex* [matSize * matSize]; // need to allocate memory for F somehow... either allocate here or in main
	for (int i = 0; i < matSize; i++) {
		// let diagoal elements be 1 - VG_ij
		F[i + i * matSize] = cuCadd(make_cuDoubleComplex(1, 0), -VG[i + i * matSize]);
		for (int j = 0; j < matSize; j++) {
			// let all non-diagonal elements be -VG_ij
			if (i != j)
			F[i + j * matSize] = -VG[i + j * matSize];
		}
	}

	/* Solve the equation FT = V with cuBLAS */
	T = solveMatrixEq(F, VMatrix); // Josephs problem :)

	delete[] F;
}

// should be a __device__ kernel called from computePhaseShift
/* TODO: Explain theory for this. */
std::vector<std::complex<double>> blattToStapp(std::complex<double> deltaMinusBB, std::complex<double> deltaPlusBB, std::complex<double> twoEpsilonJBB) {
	std::complex<double> twoEpsilonJ = std::asin(std::sin(twoEpsilonJBB) * std::sin(deltaMinusBB - deltaPlusBB));

	std::complex<double> deltaMinus = 0.5 * (deltaPlusBB + deltaMinusBB + std::asin(tan(twoEpsilonJ) / std::tan(twoEpsilonJBB))) * constants::rad2deg;
	std::complex<double> deltaPlus = 0.5 * (deltaPlusBB + deltaMinusBB - std::asin(tan(twoEpsilonJ) / std::tan(twoEpsilonJBB))) * constants::rad2deg;
	std::complex<double> epsilon = 0.5 * twoEpsilonJ * constants::rad2deg;

	return { deltaMinus, deltaPlus, epsilon };
}


/**
	Computes the phase shift for a given channel and T matrix.

	@param channel: Scattering channel
	@param key:		Channel name
	@param k0:		On-shell-point
	@param T:		T matrix
	@return			Complex phase shifts
*/

__global__
void computePhaseShifts(cuDoubleComplex* phases, double mu, bool coupled, double* k0, cuDoubleComplex* T, int quadratureN) {
	
	double rhoT =  2 * mu * k0; // Equation (2.27) in the theory

	// TODO: Explain theory for the phase shift for the coupled state
	if (coupled) {
		/*int N = quadratureN;
		cuDoubleComplex T11 = T[(N) + (N * N)]; //row + column * size
		cuDoubleComplex T12 = T[(2 * N + 1) + (N * N)];
		cuDoubleComplex T22 = T[(2 * N + 1) + (N * (2 * N + 1))];

		//Blatt - Biedenharn(BB) convention
		std::complex<double> twoEpsilonJBB{ std::atan(2.0 * T12 / (T11 - T22)) };
		std::complex<double> deltaPlusBB{ -0.5 * I * std::log(1.0 - I * rhoT * (T11 + T22) + I * rhoT * (2.0 * T12) / std::sin(twoEpsilonJBB)) };
		std::complex<double> deltaMinusBB{ -0.5 * I * std::log(1.0 - I * rhoT * (T11 + T22) - I * rhoT * (2.0 * T12) / std::sin(twoEpsilonJBB)) };

		std::vector<std::complex<double>> phasesAppend{ blattToStapp(deltaMinusBB, deltaPlusBB, twoEpsilonJBB) };

		phases.push_back(phasesAppend[0]);
		phases.push_back(phasesAppend[1]);
		phases.push_back(phasesAppend[2]); 

		*/
		//avkommenterade f�r de ger error vid icke kopplad kompilering. Avkommentera och fixa.
	}
	/* The uncoupled case completely follows equation (2.26). */
	else {
		double T0 = cuCreal(T[(quadratureN) + (quadratureN * quadratureN)]); //Farligt, detta element kanske inte �r helt reelt. Dock var koden d�lig f�rut is�fall.
		cuDoubleComplex argument = make_cuDoubleComplex(1, -2.0 * rhoT * T0);
		cuDoubleComplex delta = (-0.5 * I) * logf(argument) * constants::rad2deg;

		phases.push_back(delta);
	}
}