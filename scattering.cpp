#include "constants.h" // namespace const

#include <complex> //for complex numbers with std::complex<double>
#include <iterator> // for std:size
#include <tuple>
#include <cmath>





//-------------------------------------- HANNA'S ------------------------------------------------


// What should this return? Return void in meantime
// k (array) is quadrature points (was "p" in python), w (array) is weights, k0 (double?) is on-shell-point
void setup_G0_vector(double k[], double w[], double k0[]) // OBS: Passing arrays changes the array even outside function (something with pointers...)
{ 
	int N{ std::size(k) }; // since k has N elements k_j for j=1,...,N
	std::complex<double> D[2*(N+1)]{ }; // standard to let all elements be zero if not specified
										// WHY 2*(N+1) rather than N+1?

	//Equation (2.22)
	//For the first N values let D have elements...?
	D[N]=







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
