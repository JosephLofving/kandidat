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




inline constexpr std::complex<double> I{ (0.0,1.0) }; // Temporary way to get imaginary unit in this file


/* To get the reduced mass, check the isospin channel to determine the type of scattering */
double get_reduced_mass(std::vector<QuantumState> channel)
{
	int tz_channel{ channel[0].state["tz"] };
	if (tz_channel == -1)	  // Proton-proton scattering
		return constants::proton_mass / 2;
	else if (tz_channel == 0) // Proton-neutron scattering
		return constants::nucleon_reduced_mass;
	else if (tz_channel == 1) // Peutron-neutron scattering
		return constants::neutron_mass / 2;
	std::cout << "Incorrect tz_channel";
	abort();
}



/* k is quadrature points (was "p" in python), w is weights, k0 is on-shell-point. */
std::vector<std::complex<double>> setup_G0_vector(std::vector<double> k, std::vector<double> w, double k0)
{ 
	int N = k.size();							// k has N elements k_j for j = 1,...,N
	std::vector<std::complex<double>> D(N + 1); // Setup D vector, size N + 1

    /* Equation (2.22) is used to set elements in D. */
	double pre_factor{ (2.0 / constants::pi) }; // Later we will also multiply by 2.0 * mu (in setup_VG_kernel)
	double sum{}; // for D[0]
	for (int ind{ 0 }; ind < N; ind++)
	{
		D[ind + 1] = -pre_factor * pow(k[ind], 2) * w[ind] / (pow(k0, 2) - pow(k[ind], 2));   // Define D[1,N] with k and w vectors
		sum += w[ind] / (k0 * k0 - k[ind] * k[ind]);										  // Use in D[0]
	}
	
	D[0] = pre_factor * pow(k0, 2) * sum + I * k0;
	
	return D;
}



LapackMat setup_VG_kernel(std::vector<QuantumState> NN_channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0)
{
	std::cout << "Setting up G0(k0) in channel " << key;

	int N = k.size();
	int number_of_blocks = NN_channel.size();								 // TODO: What does this do?
	int N_channel = static_cast<int>(std::sqrt(number_of_blocks) * (N + 1)); // TODO: What does this do?
	std::vector<std::complex<double>> G0{ setup_G0_vector(k, w, k0) };		 // G0 has dimension N+1 here, as opposed to Python code

	/* Copy G0 up to (N_channel-1):th element */
	double mu{ get_reduced_mass(NN_channel) };
	std::vector<std::complex<double>> G0_part(N_channel);
	for (int index{ 0 }; index < N_channel; index++) { G0_part[index] = G0[index] * 2.0 * mu; } // TODO: May not work since G0_part not fixed size?

	/* Create VG by initializing identity matrix and then using VG[i,j] = V[i,j] * G[j] */
	LapackMat VG = LapackMat(G0_part.size());
	for (int row{ 0 }; row < G0_part.size(); row++)
	{
		for (int column{ 0 }; column < G0_part.size(); column++)
		{
			VG.setElement(row, column, V.getElement(row, column) * G0_part[column] );
		}
	}

	return VG;
}



/* Use equation ??? */
LapackMat computeTMatrix(std::vector<QuantumState> NN_channel, std::string key, LapackMat V, std::vector<double> k, std::vector<double> w, double k0) 
{
	LapackMat VG = setup_VG_kernel(NN_channel, key, V, k, w, k0);

	LapackMat identity = LapackMat(VG.width);
	LapackMat constants_matrix = (2.0 / constants::pi) * identity;
	LapackMat IVG = identity - constants_matrix;
	LapackMat LHS = IVG*VG;


	LapackMat T = solveMatrixEq(LHS, V); // IVG*T = V

	return T;
}



/* TODO: Explain theory for this. */
std::vector<std::complex<double>> blattToStapp(std::complex<double> deltaMinusBB, std::complex<double> deltaPlusBB, std::complex<double> twoEpsilonJBB) 
{
	std::complex<double> twoEpsilonJ = asin(sin(twoEpsilonJBB) * sin(deltaMinusBB - deltaPlusBB));

	std::complex<double> deltaMinus = 0.5 * (deltaPlusBB + deltaMinusBB + asin(tan(twoEpsilonJ) / tan(twoEpsilonJBB))) * constants::rad2deg;
	std::complex<double> deltaPlus = 0.5 * (deltaPlusBB + deltaMinusBB - asin(tan(twoEpsilonJ) / tan(twoEpsilonJBB))) * constants::rad2deg;
	std::complex<double> epsilon = 0.5 * twoEpsilonJ * constants::rad2deg;

	return { deltaMinus, deltaPlus, epsilon };
}


/* TODO: Explain theory for this. */
std::vector<std::complex<double>> compute_phase_shifts(std::vector<QuantumState> NN_channel,std::string key, double k0, LapackMat T)
{
	std::cout << "Computing phase shifts in channel " << key;

	std::vector<std::complex<double>> phases{};
	int number_of_blocks = NN_channel.size();			 // what blocks? block = quantumstate in channel

	double mu{ get_reduced_mass(NN_channel) };
	double factor{ 2 * mu * k0 }; // TODO: This might give wrong result depending on other constants used

	int N{ T.width };
	std::complex<double> complexOne(1);
	if (number_of_blocks > 1)
	{
		N = static_cast<int>( (N - 2) / 2); // WHY?
		std::complex<double> T11 = T.getElement(N,N);
		std::complex<double> T12 = T.getElement(2 * N + 1, N);
		std::complex<double> T22 = T.getElement(2 * N + 1, 2 * N + 1);

		/* Blatt - Biedenharn(BB) convention */
		std::complex<double> twoEpsilonJ{ std::atan(2.0 * T12 / (T11 - T22)) };
		std::complex<double> delta_plus{ -0.5 * I * std::log(complexOne - I * factor * (T11 + T22) + I * factor * (2.0 * T12) / std::sin(twoEpsilonJ)) };
		std::complex<double> delta_minus{ -0.5 * I * std::log(complexOne - I * factor * (T11 + T22) - I * factor * (2.0 * T12) / std::sin(twoEpsilonJ)) };

		std::vector<std::complex<double>> append_phases{ blattToStapp(delta_minus, delta_plus, twoEpsilonJ) };

		phases.insert(std::end(phases), std::begin(append_phases), std::end(append_phases));
	}
	else
	{
		N -= 1;
		std::complex<double> Telem = T.getElement(N, N);
		std::complex<double> Z = complexOne - factor * 2 * I * Telem;
		std::complex<double> delta{ (-0.5 * I) * std::log(Z) * constants::rad2deg };

		phases.insert(std::end(phases), &delta, &delta);
	}

	return phases;
}