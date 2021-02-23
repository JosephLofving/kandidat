#include <boost/math/quadrature/gauss.hpp>
#include "constants.h"
#include <iostream>
#include <cmath>

class arr {

};


struct tuple {
	arr t;
	arr u;
};

tuple leggauss(N) {
	arr c[N] = { 0 };
	c[N] = 1;
	if (N == 2) {
		tuple mat[1][1] = 0;
			return mat;
	}
	

}

tuple gauss_legendre_line_mesh(int N, int a, int b) {
	tuple X = leggauss(N);
	arr x = X[0];
	arr w = X[1];
	arr t = 0.5 * (x + 1) * (b - a) + a;
	arr u = w * 0.5 * (b - a);
	tuple T = new tuple(t, u);
	return T;
}

tuple gauss_legendre_inf_mesh(int N, double scale = 100.0) {
	tuple X = leggauss(N);
	arr x = X[0];
	arr w = X[1];
	double pi_over_four = constants::pi;
	arr t = scale * tan(pi_over_four) * (x + 1.0);
	arr u = scale * pi_over_four / cos(pi_over_four) * pow(x * 1.0, 2 * w);
	tuple T = new tuple(t, u);
	return T;

}