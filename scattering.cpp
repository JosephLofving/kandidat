#include "constants.h" // namespace constants

#include <complex> // for complex numbers with std::complex<double>,
#include <numeric> // for std::accumulate
#include <iterator> // for std:size
#include <tuple>
#include <cmath>
#include <vector>
#include <iostream>





//-------------------------------------- HANNA'S ------------------------------------------------


/* k (vector) is quadrature points (was "p" in python), w (vector) is weights, k0 (double) is on-shell-point.
 * Passes vector of type complex double. */
std::vector<std::complex<double>> setup_G0_vector(std::vector<double> k, std::vector<double> w, double k0)
{ 
	int N{ k.size() };								  // since k has N elements k_j for j=1,...,N
	std::vector<std::complex<double>> D(2 * (N + 1)); // standard to let all elements be zero if not specified
												      // WHY 2*(N+1) rather than N+1?


    /* Now: Equation (2.22)
	 * OBS: Chose own theory before python script here, so it differs.
	 * Difference: In python first part of (2.22) is D[0,N] and second part of (2.22) is D[0],
	 * and we use 2*2*mu/pi as constant while python script does not 
	 * (probably because mu is not a constant? Defined in next function) */

	double mu{ 1 }; // reduced mass, from constants, set to 1 since it does not exist yet
	double pre_factor{ (2 / constants::pi) * 2 * mu };

	double sum{}; // for D[0]
	for (int ind{ 0 }; ind <= N; ind++)
	{
		D[ind+1]= -pre_factor * pow(k[ind],2) * w[ind] / (pow(k0,2) - pow(k[ind],2));   // define D[1,N] with k and w vectors
		D[(N + 1) + ind + 1] = D[ind + 1];												// copy first half of D to second half

		sum += w[ind] / (k0 * k0 - k[ind] * k[ind]);									// to use in D[0]
	}
	
	D[0] = pre_factor * k0 * k0 * sum;
	
	return D;
}

// setup_VG_kernel


// T should be matrix, maybe multidimensional vector? Let NN_channel be MD vector
std::vector<double> compute_phase_shifts(std::vector<std::vector<double>> NN_channel, double k0, std::vector<double> T)
{
	std::vector<double> phases{};
	int number_of_blocks{ NN_channel.size() };
	int channel_index[0]['channel_index'];
	std::cout << "Computing phase shifts in channel " << channel_index;

	// T.shape?

	double mu{};
	if (NN_channel[0]['tz'] == -1)
		mu = constants::mp / 2;
	if (NN_channel[0]['tz'] == 0)
		mu = constants::uN;
	if (NN_channel[0]['tz'] == -1)
		mu = constants::mN / 2;

}


//----------------------------------------JOSEPH'S------------------------------------------------------------------

std::tuple<double, double, double> blattToStapp(double deltaMinusBB, double deltaPlusBB, double twoEpsilonJBB) {
	double twoEpsilonJ = asin(sin(twoEpsilonJBB) * sin(deltaMinusBB - deltaPlusBB));
	double deltaMinus = 0.5 * (deltaPlusBB + deltaMinusBB + asin(tan(twoEpsilonJ) / tan(twoEpsilonJBB)))* constants::rad2deg;
	double deltaPlus = 0.5 * (deltaPlusBB + deltaMinusBB - asin(tan(twoEpsilonJ) / tan(twoEpsilonJBB)))* constants::rad2deg;
	double epsilon = 0.5 * twoEpsilonJ* constants::rad2deg;

	return { deltaMinus, deltaPlus, epsilon };
}
