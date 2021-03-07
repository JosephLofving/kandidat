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




inline constexpr std::complex<double> I(0.0,1.0); // Temporary way to get imaginary unit in this file

//std::complex<double> I(0,1);

/* To get the reduced mass, check the isospin channel to determine the type of scattering */
double get_reduced_mass(std::vector<QuantumState> channel)
{
	int tz_channel{ channel[0].state["tz"] };
	if (tz_channel == -1)	  // Proton-proton scattering
		return constants::proton_mass / 2;
	else if (tz_channel == 0) // Proton-neutron scattering
		return constants::nucleon_reduced_mass;
	else if (tz_channel == 1) // Neutron-neutron scattering
		return constants::neutron_mass / 2;
	std::cout << "Incorrect tz_channel";
	abort();
}



/* k is quadrature points (was "p" in python), w is weights, k0 is on-shell-point. */
std::vector<std::complex<double>> setup_G0_vector(std::vector<double> k, std::vector<double> w, double k0)
{
	int N = k.size();							// k has N elements k_j for j = 1,...,N
	std::vector<std::complex<double>> D(2* (N + 1)); // Setup D vector, size N + 1

    /* Equation (2.22) is used to set elements in D. */
	double pi_over_two{ (constants::pi / 2.0) }; // Later we will also multiply by 2.0 * mu (in setup_VG_kernel)
	double sum{}; // for D[0]
	for (int ind{ 0 }; ind < N; ind++)
	{
		D[ind] = w[ind] * pow(k[ind], 2) / (pow(k0, 2) - pow(k[ind], 2));   // Define D[1,N] with k and w vectors
		D[ind + (N+1)] = D[ind];

		sum += w[ind] * pow(k0, 2) / (pow(k0, 2) - pow(k[ind], 2));          // Use in D[0]
	}

	D[N] = - sum - pi_over_two * I * k0;
	D[2 *( N + 1) - 1] = D[N];

	return D;
}

///* k is quadrature points (was "p" in python), w is weights, k0 is on-shell-point. */
//std::vector<std::complex<double>> setup_G0_vector(std::vector<double> k, std::vector<double> w, double k0)
//{
//	int N = k.size();							// k has N elements k_j for j = 1,...,N
//	std::vector<std::complex<double>> D(N + 1); // Setup D vector, size N + 1
//
//	/* Equation (2.22) is used to set elements in D. */
//	double pre_factor{ (2.0 / constants::pi) }; // Later we will also multiply by 2.0 * mu (in setup_VG_kernel)
//	double sum{}; // for D[0]
//	for (int ind{ 0 }; ind < N; ind++)
//	{
//		D[ind + 1] = -pre_factor * pow(k[ind], 2) * w[ind] / (pow(k0, 2) - pow(k[ind], 2));   // Define D[1,N] with k and w vectors
//		sum += w[ind] / (k0 * k0 - k[ind] * k[ind]);										  // Use in D[0]
//	}
//
//	D[0] = pre_factor * pow(k0, 2) * sum + I * k0;
//
//	return D;
//}


LapackMat setup_VG_kernel(std::vector<QuantumState> NN_channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0)
{
	std::cout << "Setting up G0(k0) in channel " << key << std::endl;

	int N = k.size();
	int number_of_blocks = NN_channel.size();								 // Is either 1 (uncoupled) or 4 (coupled)
	int N_channel = static_cast<int>(std::sqrt(number_of_blocks) * (N + 1)); // If coupled, use second half of G0
	std::vector<std::complex<double>> G0{ setup_G0_vector(k, w, k0) };		 // G0 has dimension N+1 here, as opposed to Python code

	/* Copy G0 up to (N_channel-1):th element */
	double mu{ get_reduced_mass(NN_channel) };
	std::vector<std::complex<double>> G0_part(N_channel);
	for (int index{ 0 }; index < N_channel; index++) { G0_part[index] = G0[index] * 2.0 * mu; } // * 2.0 * mu


	/* Create VG by initializing identity matrix and then using VG[i,j] = V[i,j] * G[j] */
	LapackMat VG = LapackMat(G0_part.size());

	for (int row{ 0 }; row < G0_part.size(); row++)
	{
		for (int column{ 0 }; column < G0_part.size(); column++)
		{
			VG.setElement(row, column, V.getElement(row, column) * G0_part[column]);
		}
	}

	return VG;
}



/* Use equation ??? */
/* Computes the T-matrix
   @param NN_channel The scattering channel
   @param key The channel key (?)
   @param V The potential matrix
   @param k The quadrature points from the Gauss-Legendre mesh
   @param w The weights of the quadrature points in k
   @param k0 The scattering momentum
   @return T-matrix
*/
LapackMat computeTMatrix(std::vector<QuantumState> NN_channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0)
{
	std::cout << "Solving for the complex T-matrix in channel " << key << ":";

	LapackMat VG = setup_VG_kernel(NN_channel, key, V, k, w, k0);

	LapackMat identity = LapackMat(VG.width);
	LapackMat two_over_pi_VG = (2.0 / constants::pi) * VG;
	LapackMat IVG = identity - two_over_pi_VG;

	LapackMat T = solveMatrixEq(IVG, V); // IVG*T = V

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
std::vector<std::complex<double>> compute_phase_shifts(std::vector<QuantumState> NN_channel,std::string key, double k0, LapackMat T)
{
	std::cout << "Computing phase shifts in channel " << key << std::endl;

	std::vector<std::complex<double>> phases{};
	int number_of_blocks = NN_channel.size(); // 1 if uncoupled, 4 if coupled
	double mu{ get_reduced_mass(NN_channel) };
	double factor{ 2 * mu * k0 };
	int N{ T.width };

	if (number_of_blocks > 1) // if coupled
	{
		N = static_cast<int>( (N - 2) / 2);				// WHY?
		std::complex<double> T11 = T.getElement(N,N);
		std::complex<double> T12 = T.getElement(2 * N + 1, N);
		std::complex<double> T22 = T.getElement(2 * N + 1, 2 * N + 1);

		/* Blatt - Biedenharn(BB) convention */
		std::complex<double> twoEpsilonJ_BB{ std::atan(2.0 * T12 / (T11 - T22)) };
		std::complex<double> delta_plus_BB{ -0.5 * I * std::log(1.0 - I * factor * (T11 + T22) + I * factor * (2.0 * T12) / std::sin(twoEpsilonJ_BB)) };
		std::complex<double> delta_minus_BB{ -0.5 * I * std::log(1.0 - I * factor * (T11 + T22) - I * factor * (2.0 * T12) / std::sin(twoEpsilonJ_BB)) };

		std::vector<std::complex<double>> phases_append{ blattToStapp(delta_minus_BB, delta_plus_BB, twoEpsilonJ_BB) };

		phases.push_back(phases_append[0]);
		phases.push_back(phases_append[1]);
		phases.push_back(phases_append[2]);

//std::cout << "\nDELTA_PLUS: " << delta_plus << "\n";
//std::cout << "\nDELTA_MINUS: " << delta_minus << "\n";
	}
	else
	{
		N -= 1;
		std::complex<double> T_element = T.getElement(N, N);
		std::complex<double> Z = 1.0 - factor * 2 * I * T_element;
		std::complex<double> delta{ (-0.5 * I) * std::log(Z) * constants::rad2deg };

		phases.push_back(delta);
//std::cout << "\nDELTA: " << delta << "\n";
	}

	return phases;
}