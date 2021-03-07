#include "constants.h"	// namespace constants
#include "quantumStates.h"
#include "lapackAPI.h"

#include <complex>		
#include <tuple>
#include <cmath>
#include <vector>
#include <iostream>
#include <tuple>


/* About this file:
 * 1. Here k is used for quadrature points, whereas the Python code uses p.
 * 2. Here the G0 vector has size N + 1, whereas the Python code has a G0 vector
 *    of size 2 * (N + 1). In Python, the last N + 1 elements are copied from 
 *    the first N + 1. This was used to facilitate use for coupled states, but 
 *    in this project coupling should be taken into consideration when needed instead. 
 * 3. Here the on shell point is in the zeroth element of G0, whereas it is in
 *    element N + 1 in the Python code. */




// TODO:
// Coupling
// Constants pi/2
// constants mu
// return to own theory

// Plan for coupling: isCoupled function and then a local copy when needed


inline constexpr std::complex<double> I(0.0,1.0); 


/* To get the reduced mass, check the isospin channel to determine the type of scattering */
double get_reduced_mass(std::vector<QuantumState> channel) {
	int tz_channel{ channel[0].state["tz"] };
	if (tz_channel == -1)	  // Proton-proton scattering
		return constants::proton_mass / 2;
	else if (tz_channel == 0) // Proton-neutron scattering
		return constants::nucleon_reduced_mass;
	else if (tz_channel == 1) // Neutron-neutron scattering
		return constants::neutron_mass / 2;

	std::cout << "Incorrect tz_channel"; // TODO: This error handling should maybe be done sooner?
	abort();
}


/* If there is only one channel the state is uncoupled, otherwise there are four channels and the state is coupled. */
bool isCoupled(std::vector<QuantumState> NN_channel) {
	return !(NN_channel.size() == 1);
}


// OBS!!! SETUP_G0 HAS A NEW INPUT: NN_CHANNEL
/* k is quadrature points (was "p" in python), w is weights, k0 is on-shell-point. */
std::vector<std::complex<double>> setup_G0_vector(std::vector<QuantumState> NN_channel, std::vector<double> k, std::vector<double> w, double k0) { 
	int N = k.size();  // k has N elements k_j for j = 1,...,N
	std::vector<std::complex<double>> D(N + 1);

    /* Equation (2.22) is used to set elements in D. */
	double mu = get_reduced_mass(NN_channel);
	double two_over_pi = (2.0 / constants::pi);
	double sum{}; // for D[0]
	for (int i{ 0 }; i < N; i++) {
		D[i] = - two_over_pi * 2.0 * mu * pow(k[i], 2) * w[i] / (pow(k0, 2) - pow(k[i], 2)); // Define D[1,N] with k and w vectors
		sum += w[i] / (pow(k0, 2) - pow(k[i], 2));										   // Used in D[0]
	}
	
	D[N] = two_over_pi * 2.0 * mu * pow(k0, 2) * sum + 2.0 * mu * I * k0; // In the theory, this element is placed at index 0

	return D;
}



LapackMat setup_VG_kernel(std::vector<QuantumState> NN_channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0) {
	std::cout << "Setting up G0(k0) in channel " << key << std::endl;
	std::cout << "IN SCATTERING BRANCH";
	std::vector<std::complex<double>> G0{ setup_G0_vector(NN_channel, k, w, k0) };

	/* If coupled, append G0 to itself to facilitate calculations. This means the second half of G0 is a copy of the first. */
	if (isCoupled(NN_channel)) {
		G0.insert(std::end(G0), std::begin(G0), std::end(G0)); // TODO: Risk that this does not work properly, might want to test
	}

	/* Create VG by initializing identity matrix and then using VG[i,j] = V[i,j] * G[j] */
	LapackMat VG = LapackMat(G0.size());

	for (int row{ 0 }; row < G0.size(); row++) {
		for (int column{ 0 }; column < G0.size(); column++) {
			VG.setElement(row, column, V.getElement(row, column) * G0[column]);
		}
	}

	return VG;
}



/* Equation (2.24): [F][T] = [V]. */
LapackMat computeTMatrix(std::vector<QuantumState> NN_channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0)  {
	std::cout << "Solving for the complex T-matrix in channel " << key << std::endl;

	LapackMat VG = setup_VG_kernel(NN_channel, key, V, k, w, k0);
	LapackMat identity = LapackMat(VG.width);
	LapackMat F = identity + VG; // Definition in equation (2.25)

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


/* TODO: Explain theory for this. */
std::vector<std::complex<double>> compute_phase_shifts(std::vector<QuantumState> NN_channel,std::string key, double k0, LapackMat T) {
	std::cout << "Computing phase shifts in channel " << key << std::endl;

	std::vector<std::complex<double>> phases{};
	double mu{ get_reduced_mass(NN_channel) };
	double rho_T =  2 * mu * k0; // Equation (2.27) in the theory
	int N{ T.width };

	if (isCoupled(NN_channel)) {
		N = static_cast<int>( (N - 2) / 2);				// WHY?
		std::complex<double> T11 = T.getElement(N,N);
		std::complex<double> T12 = T.getElement(2 * N + 1, N);
		std::complex<double> T22 = T.getElement(2 * N + 1, 2 * N + 1);

		/* Blatt - Biedenharn(BB) convention */
		std::complex<double> twoEpsilonJ_BB{ std::atan(2.0 * T12 / (T11 - T22)) };
		std::complex<double> delta_plus_BB{ -0.5 * I * std::log(1.0 - I * rho_T * (T11 + T22) + I * rho_T * (2.0 * T12) / std::sin(twoEpsilonJ_BB)) };
		std::complex<double> delta_minus_BB{ -0.5 * I * std::log(1.0 - I * rho_T * (T11 + T22) - I * rho_T * (2.0 * T12) / std::sin(twoEpsilonJ_BB)) };

		std::vector<std::complex<double>> phases_append{ blattToStapp(delta_minus_BB, delta_plus_BB, twoEpsilonJ_BB) };

		phases.push_back(phases_append[0]);
		phases.push_back(phases_append[1]);
		phases.push_back(phases_append[2]);
	}
	/* This completely follows the theory in equation (2.26). */
	else {
		N -= 1;
		std::complex<double> T0 = T.getElement(N, N);
		std::complex<double> argument = 1.0 - 2.0 * I * rho_T * T0;
		std::complex<double> delta = (-0.5 * I) * std::log(argument) * constants::rad2deg;
     
		phases.push_back(delta);
	}

	return phases;
}