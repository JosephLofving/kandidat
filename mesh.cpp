//#include <boost/math/quadrature/gauss.hpp> //vi skriver den sj�lva ist�let
#include "constants.h"
#include "lapackAPI.h"
#include <iostream>
#include <cmath>
#include <lapackAPI.h> //ger error av någon anledning...
#include <vector>
#include <algorithm>
#include <list>
#include <numeric> //beh�vs f�r std::iota
#include <random>

struct two_vectors {
	std::vector<double> t;
	std::vector<double> u;
};

two_vectors leggauss(int N) {
	std::vector<double> c(N);
	


};

lapackMat legcompanion(std::vector<double> c) {
	c[N] = 1;
	if (N == 2) {
		lapackMat mat = new lapackMat(1,1); //se kommentar nedan
		return mat;
	}
	lapackMat mat = new lapackMat(N, N); //den verkar inte kunna koppla till lapackAPI.h? Ska iaf skapa matris = zeros(N*N)
	std::vector<double> arange_vec(N);
	std::iota(std::begin(arange_vec), std::end(arange_vec), 0); //arange_vec = {0,1,...,N}
	std::vector<double> scl = 1./sqrt


}

/* This follows equation (2.18) and (2.19) with the same notation */
two_vectors gauss_legendre_line_mesh(int N, int a, int b) { // Change to struct?
	two_vectors X = leggauss(N);
	std::vector<double> p = X[0];
	std::vector<double> w_prime = X[1];

	/* Translate p and w_prime values for [-1,1] to a general 
	 * interval [a,b]  for quadrature points k and weights w */  

	double C{ 1 }; // what is the value of this parameter? Choose later?
	// for loops for these operations?
	std::vector<double> k = (1 + p) / (1 - p); // 0.5 * (p + 1) * (b - a) + a;
	std::vector<double> w = 2 * C * w_prime / pow(1 - k, 2);//  w_prime * 0.5 * (b - a);
	two_vectors k_and_w{ k, w };

	return k_and_w;
}

two_vectors gauss_legendre_inf_mesh(int N, double scale = 100.0) {
	two_vectors X = leggauss(N);
	std::vector<double> p = X.v1;		// first of the two vectors in X
	std::vector<double> w_prime = X.v2; // second of the two vectors in X
	
	double pi_over_four = constants::pi;

	std::vector<double> k = scale * tan(pi_over_four) * (p + 1.0);
	std::vector<double> w = scale * pi_over_four / cos(pi_over_four) * pow(p * 1.0, 2 * w_prime);
	two_vectors k_and_w{ k, w };

	return k_and_w;
}