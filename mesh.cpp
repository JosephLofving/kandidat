//#include <boost/math/quadrature/gauss.hpp> //vi skriver den sj�lva ist�let
#include "constants.h"
#include "lapackAPI.h"

#include <iostream>
#include <cmath>
<<<<<<< HEAD
#include <lapackAPI.h>
#include <vector>

struct tuple {
	std::vector<double> t;
	std::vector<double> u;
=======
#include <vector>



struct Two_vectors {
	std::vector<double> v1;
	std::vector<double> v2;
>>>>>>> 0bc0121ee6790b466965a30ed17bea8810ddac31
};

tuple leggauss(int N) {
	std::vector<double> c(N * N);
	c[N] = 1;
	if (N == 2) {
		tuple mat[1][1] = 0;
			return mat;
	}
	

}


/* This follows equation (2.18) and (2.19) with the same notation */
Two_vectors gauss_legendre_line_mesh(int N, int a, int b) { // Change to struct?
	Two_vectors X = leggauss(N);
	std::vector<double> p = X[0];
	std::vector<double> w_prime = X[1];

	/* Translate p and w_prime values for [-1,1] to a general 
	 * interval [a,b]  for quadrature points k and weights w */  

	double C{ 1 }; // what is the value of this parameter? Choose later?
	// for loops for these operations?
	std::vector<double> k = (1 + p) / (1 - p); // 0.5 * (p + 1) * (b - a) + a;
	std::vector<double> w = 2 * C * w_prime / pow(1 - k, 2);//  w_prime * 0.5 * (b - a);
	Two_vectors k_and_w{ k, w };

	return k_and_w;
}

Two_vectors gauss_legendre_inf_mesh(int N, double scale = 100.0) {
	Two_vectors X = leggauss(N);
	std::vector<double> x = X[0];
	std::vector<double> w = X[1];
	double pi_over_four = constants::pi;
	std::vector<double> t = scale * tan(pi_over_four) * (x + 1.0);
	std::vector<double> u = scale * pi_over_four / cos(pi_over_four) * pow(x * 1.0, 2 * w);
	Two_vectors T = new Two_vectors(t, u);
	return T;

}