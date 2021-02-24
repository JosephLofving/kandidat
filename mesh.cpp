//#include <boost/math/quadrature/gauss.hpp> //vi skriver den sj�lva ist�let
#include "constants.h"
#include "lapackAPI.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <list>
#include <numeric> //beh�vs f�r std::iota
#include <random>

struct two_vectors {
	std::vector<double> v1;
	std::vector<double> v2;
};

std::vector<double> elementwise_mult(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> vec;
	if (v1.size() != v2.size())
		std::cout << "Error: vectors must be same length\n";
	for (int i = 0; i < v1.size(); i++) {
		vec[i] = v1[i] * v2[i];
	}
	return vec;
}

two_vectors leggauss(int N) {
	std::vector<double> c(N);
	c[N - 1] = 1; //nu är alltså alla element 0, förutom det sista som är 1.
	lapackMat* m = legcompanion(c); //se legcompanion
	
	std::vector<double> x = {};
	std::vector<double> w = {};

	two_vectors x_and_w{ x, w };
	return x_and_w;
};

std::vector<double> legval(std::vector<double> x, std::vector<double> c) {

}

std::vector<double> legder(std::vector<double> c) {

}

//nu är den klar, bara att fixa så lapackMat är av rätt typ, den kan inte accessa sina funktioner
lapackMat* legcompanion(std::vector<double> c) {
	int N = c.size();
	if (N == 2) {
		lapackMat* mat = new lapackMat(1,1);
		return mat;
	}
	lapackMat* mat = new lapackMat(N, N); //Skapar matris mat = zeros(N*N)
	std::vector<double> scl(N);
	std::iota(std::begin(scl), std::end(scl), 0); //scl= {0,1,...,N-1}
	std::for_each(scl.begin(), scl.end(), [](double& v) {v = sqrt(2 * v + 1); }); //scl[i] = sqrt(2*scl[i]+1) forall i
	std::for_each(scl.begin(), scl.end(), [](double& v) { v = 1 / v; });
	//std::vector<double> top(N - 1, 0); //bös, behövs nog ej
	//std::vector<double> bot(N - 1, 0);
	std::vector<double> scll(N-1);
	std::iota(std::begin(scll), std::end(scll), 1); //scll= {1,...,N-1} (börjar på 1, men har denna gång endast N-1 element, till skillnad från scl).
	std::vector<double> scl_removedLast = scl; //scl fast med sista elementet borttaget
	std::vector<double> scl_removedFirst = scl; //scl fast med första elementet borttaget
	scl_removedLast.pop_back();
	scl_removedFirst.erase(scl_removedFirst.begin());
	std::vector<double> scl_prod = elementwise_mult(scl_removedLast, scl_removedFirst); //elementvis produkt mellan dessa
	std::vector<double> top = elementwise_mult(scl_prod, scll);
	for (int i = 0; i < N-1; i++) {
		mat->setElement(i, i + 1, top[i]);
		mat->setElement(i + 1, i, top[i]); //nää det är nåt fel med syntax fortfarande, den tror att mat är en pointer nu?
	}
	c.pop_back(); //nu är c bara nollor, samt 1 element kortare.
	double scl_lastValue = scl.back();
	std::for_each(scl.begin(), scl.end(), [&](double& v) { v = v / scl_lastValue; });
	double N_div = N / (2 * N - 1);
	std::vector<double> bigProd = elementwise_mult(scl, c);
	std::for_each(bigProd.begin(), bigProd.end(), [&](double& v) { v = v * N_div; });
	for (int i = 0; i < N; i++) {
		double prev_elem = mat->getElement(i, N);
		mat->setElement(i, N, prev_elem - bigProd[i]);
	}
	return mat;

}

/* This follows equation (2.18) and (2.19) with the same notation */
two_vectors gauss_legendre_line_mesh(int N, int a, int b) { // Change to struct?
	two_vectors X = leggauss(N);
	std::vector<double> p = X[0];
	std::vector<double> w_prime = X[1];

	/* Translate p and w_prime values for [-1,1] to a general 
	 * interval [a,b]  for quadrature points k and weights w */  

	// for loops for these operations?
	std::vector<double> k = 0.5 * (p + 1) * (b - a) + a;
	std::vector<double> w = w_prime * 0.5 * (b - a);
	two_vectors k_and_w{ k, w };

	return k_and_w;
}

two_vectors gauss_legendre_inf_mesh(int N, double scale = 100.0) {
	two_vectors X = leggauss(N);
	std::vector<double> p = X.v1;		// first of the two vectors in X
	std::vector<double> w_prime = X.v2; // second of the two vectors in X
	
	double pi_over_four = 0.25 * constants::pi;

	std::vector<double> k = (1 + p) / (1 - p); 
	std::vector<double> w = 2 * scale * w_prime / pow(1 - k, 2);
	two_vectors k_and_w{ k, w };

	return k_and_w;
}