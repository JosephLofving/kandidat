#include "scattering.h"


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
	return !(channel.size() == 1); // If there is only one channel the state is uncoupled, otherwise there are four channels and the state is coupled.
}


/** 
	Sets up a complex vector needed to solve the T matrix equation. 
	@param k:	Quadrature points
	@param w:	Weights for quadrature points
	@param k0:	On-shell-point
	@return		G0 vector
*/
std::vector<std::complex<double>> setupG0Vector(std::vector<QuantumState> channel, std::vector<double> k, std::vector<double> w, double k0) {
	int N = k.size();
	std::vector<std::complex<double>> D(N + 1);

	double mu = getReducedMass(channel);
	double twoMu = (2.0 * mu);
	double twoOverPi = (2.0 / constants::pi);
	double sum = 0; 
	for (int i = 0; i < N; i++) {
		D[i] = - twoOverPi * twoMu * pow(k[i], 2) * w[i] / (pow(k0, 2) - pow(k[i], 2)); // Define D[0,N-1] with vectors k and w
		sum += w[i] / (pow(k0, 2) - pow(k[i], 2));										// Used in D[N]
	}

	D[N] = twoOverPi * twoMu * pow(k0, 2) * sum + twoMu * I * k0;						// In the theory, this element is placed at index 0

	return D;
}

/**
	Multiplies the potential matrix with the G0 vector.

	@param channel: Scattering channel
    @param key:		Channel name
    @param V:		Potential matrix
	@param k:		Quadrature points
	@param w:		Weights for quadrature points
	@param k0:		The on-shell-point
	@return			VG kernel
*/
LapackMat setupVGKernel(std::vector<QuantumState> channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0) {
	//std::cout << "Setting up G0(k0) in channel " << key << std::endl;
	std::vector<std::complex<double>> G0 = setupG0Vector(channel, k, w, k0);

	/* If coupled, append G0 to itself to facilitate calculations. This means the second half of G0 is a copy of the first. */
	if (isCoupled(channel)) {
		G0.insert(std::end(G0), std::begin(G0), std::end(G0)); // TODO: Risk that this does not work properly, might want to test in uncoupled case
	}

	/* Create VG by using VG[i,j] = V[i,j] * G[j] */
	LapackMat VG = LapackMat(G0.size());
	for (int row = 0; row < G0.size(); row++) {
		for (int column = 0; column < G0.size(); column++) {
			VG.setElement(row, column, V.getElement(row, column) * G0[column]);
		}
	}

	for (int i = 0; i < VG.width * VG.height; i += 100) {
		std::cout << VG.contents[i].real() << std::endl;
	}

	return VG;
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
LapackMat computeTMatrix(std::vector<QuantumState> channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0)  {
	std::cout << "Solving for the complex T-matrix in channel " << key << std::endl;

	LapackMat VG = setupVGKernel(channel, key, V, k, w, k0);
	LapackMat identity = LapackMat(VG.width);
	LapackMat F = identity + VG;

	// Solves the equation FT = V.
	LapackMat T = solveMatrixEq(F, V);

	return T;
}


/* TODO: Explain theory for this. */
std::vector<std::complex<double>> blattToStapp(std::complex<double> deltaMinusBB, std::complex<double> deltaPlusBB, std::complex<double> twoEpsilonJBB)
{
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
std::vector<std::complex<double>> computePhaseShifts(std::vector<QuantumState> channel, std::string key, double k0, LapackMat T) {
	std::cout << "Computing phase shifts in channel " << key << std::endl;
	std::vector<std::complex<double>> phases;

	double mu = getReducedMass(channel);
	double rhoT =  2 * mu * k0; // Equation (2.27) in the theory
	int N = T.width;
	// TODO: Explain theory for the phase shift for the coupled state
	if (isCoupled(channel)) {
		N = static_cast<int>( (N - 2) / 2);
		std::complex<double> T11 = T.getElement(N,N);
		std::complex<double> T12 = T.getElement(2 * N + 1, N);
		std::complex<double> T22 = T.getElement(2 * N + 1, 2 * N + 1);

		/* Blatt - Biedenharn(BB) convention */
		std::complex<double> twoEpsilonJBB{ std::atan(2.0 * T12 / (T11 - T22)) };
		std::complex<double> deltaPlusBB{ -0.5 * I * std::log(1.0 - I * rhoT * (T11 + T22) + I * rhoT * (2.0 * T12) / std::sin(twoEpsilonJBB)) };
		std::complex<double> deltaMinusBB{ -0.5 * I * std::log(1.0 - I * rhoT * (T11 + T22) - I * rhoT * (2.0 * T12) / std::sin(twoEpsilonJBB)) };

		std::vector<std::complex<double>> phasesAppend{ blattToStapp(deltaMinusBB, deltaPlusBB, twoEpsilonJBB) };

		phases.push_back(phasesAppend[0]);
		phases.push_back(phasesAppend[1]);
		phases.push_back(phasesAppend[2]);
	}
	/* The uncoupled case completely follows equation (2.26). */
	else {
		N -= 1;
		std::complex<double> T0 = T.getElement(N, N);
		std::complex<double> argument = 1.0 - 2.0 * I * rhoT * T0;
		std::complex<double> delta = (-0.5 * I) * std::log(argument) * constants::rad2deg;

		phases.push_back(delta);
	}

	return phases;
}