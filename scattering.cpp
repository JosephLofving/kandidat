#include "constants.h" // namespace const

#include <complex> // for complex numbers with std::complex<double>, imaginary unit with 
#include <numeric> // for std::accumulate
#include <iterator> // for std:size
#include <tuple>
#include <cmath>





//-------------------------------------- HANNA'S ------------------------------------------------


/* k (array) is quadrature points (was "p" in python), w (array) is weights, k0 (double?) is on-shell-point
 * OBS: Passing arrays changes the array even outside function (something with pointers...)
 * This function needs to return a pointer... unsure at the moment
 */
double* setup_G0_vector(double k[], double k0[], double w[]) 
{ 
	int N{ std::size(k) };					// since k has N elements k_j for j=1,...,N
	std::complex<double> D[2 * (N + 1)]{ }; // standard to let all elements be zero if not specified
										    // WHY 2*(N+1) rather than N+1?


    /* Now: Equation (2.22)
	 * OBS: Chose own theory before python script here, so it differs.
	 * Difference: In python first part of (2.22) is D[0,N] and second part of (2.22) is D[0],
	 * and we use 2*2*mu/pi as constant while python script does not
	 */

	double mu{ 1 }; // reduced mass, from constants, set to 1 since it does not exist yet
	double pre_factor{ (2 / const::pi) * 2 * mu };

	D[1, N] = - pre_factor * k * k * w / (k0 * k0 - k * k);					// first part of (2.22)
	double d[N]{ w / (k0 * k0 - k * k };									// makes next line shorter
	D[0] = pre_factor * k0 * k0 * std::accumulate(d.begin(), d.end(), 0);	// second part of  (2.22)
	D[0] += 2 * mu * 1i * k0;
	D[N + 1, 2 * (N + 1)] = D[0, N + 1];									// make a copy of theoretical D and 
																			// store in second part of array D

	return D;
}

// setup_VG_kernel
//compute_phase_shifts






//----------------------------------------JOSEPH'S------------------------------------------------------------------

std::tuple<double, double, double> blattToStapp(double deltaMinusBB, double deltaPlusBB, double twoEpsilonJBB) {
	double twoEpsilonJ = asin(sin(twoEpsilonJBB) * sin(deltaMinusBB - deltaPlusBB));
	double deltaMinus = 0.5 * (deltaPlusBB + deltaMinusBB + asin(tan(twoEpsilonJ) / tan(twoEpsilonJBB)))* const::rad2deg;
	double deltaPlus = 0.5 * (deltaPlusBB + deltaMinusBB - asin(tan(twoEpsilonJ) / tan(twoEpsilonJBB)))* const::rad2deg;
	double epsilon = 0.5 * twoEpsilonJ* const::rad2deg;

	return { deltaMinus, deltaPlus, epsilon };
}
