//#include <boost/math/quadrature/gauss.hpp> //vi skriver den sj�lva ist�let
#include "constants.h"
#include "lapackAPI.h"
#include "mesh.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <list>
#include <numeric> //beh�vs f�r std::iota
#include <random>


std::vector<double> elementwise_mult(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> vec;
	if (v1.size() != v2.size())
		std::cout << "Error: vectors must be same length\n";
	for (int i = 0; i < v1.size(); i++) {
		vec[i] = v1[i] * v2[i];
	}
	return vec;
}

//först opererar den abs() på alla vektorns element, och sen returnar den det största av dessa element.
double absmax(std::vector<double> vec) {
	std::vector<double> vec1;
	for (int i = 0; i < vec.size(); ++i) {
		vec1[i] = abs(vec[i]);
	}
	double max = *max_element(vec1.begin(), vec1.end());
	return max;
}

//Gauss-Legendre quadrature
//Computes the sample pointsand weights for Gauss - Legendre quadrature.
//These sample points and weights will correctly integrate polynomials of
//degree : math:`2*deg - 1` or less over the interval : math:`[ - 1, 1]` with
//the weight function : math:`f(x) = 1`.
Two_vectors leggauss(int N) {
	if (N < 1) {
		std::cout << "Index must be >=1\n";
	}
	//first approximation of roots.We use the fact that the companion
	//matrix is symmetric in this case in order to obtain better zeros.
	std::vector<double> c(N);
	c[N - 1] = 1; //nu är alltså alla element 0, förutom det sista som är 1.
	LapackMat* m = legcompanion(c); //se legcompanion
	std::vector<double> x = eigenValues(*m);

	//improve roots by one application of Newton
	std::vector<double> dy = legval(x, c);
	std::vector<double> df = legval(x, legder(c));
	for (int i = 0; i < x.size(); ++i) {
		x[i] -= dy[i] / df[i];
	}

	//compute the weights. We scale the factor to avoid possible numerical overflow.
	std::vector<double> c_removedFirst = c;
	c_removedFirst.erase(c_removedFirst.begin()); //c med första elementet borttaget
	std::vector<double> fm = legval(x, c_removedFirst);
	std::vector<double> w;
	
	double fm_absmax = absmax(fm);
	double df_absmax = absmax(df);
	for (int i = 0; i < fm.size(); ++i) {
		fm[i] /= fm_absmax;
		df[i] /= df_absmax;
		w[i] = 1 / (fm[i] / df[i]);
	}

	//for Legendre we can also symmetrize
	std::vector<double> w_reverse = w;
	std::reverse(w_reverse.begin(), w_reverse.end());
	std::vector<double> x_reverse = x;
	std::reverse(x_reverse.begin(), x_reverse.end());
	for (int i = 0; i < w.size(); ++i) {
		w[i] = (w[i] + w_reverse[i]) / 2;
		x[i] = (x[i] - x_reverse[i]) / 2;
	}

	//scale w to get the right value
	double w_sum;
	std::for_each(w.begin(), w.end(), [&](double v) {
		w_sum += v;
		});
	for (int i = 0; i < w.size(); ++i) {
		w[i] *= 2 / w_sum;
	}

	Two_vectors x_and_w{ x, w };
	return x_and_w;
};

std::vector<double> legval(std::vector<double> x, std::vector<double> c) {
	std::vector<double> c0, c1;
	if (c.size() == 1) {
		c0[0] = c[0];
		c1[0] = 0;
	}
	else if (c.size() == 2) {
		c0[0] = c[0];
		c1[0] = c[1];
	}
	else {
		int nd = c.size();
		c0[0] = c[nd - 2]; //näst sista elementet i c
		c1[0] = c[nd - 1]; //sista elementet i c
		for (int i = 3; i < nd + 1; i++) {
			std::vector<double> tmp = c0;
			nd -= 1;
			std::vector<double> c1_times_const;
			std::for_each(c1_times_const.begin(), c1_times_const.end(), [&](double v) {
				v *= (nd-1)/nd;
				});
			c0 = c1_times_const;
			std::for_each(c0.begin(), c0.end(), [&](double v) {
				v = c[nd-i] - v;
				}); //c0 = c[nd - i] - c1_times_const;
			std::vector<double> c1x = elementwise_mult(c1, x);
			c1 = tmp;
			c1.insert(c1.end(), c1x.begin(), c1x.end());
		}
	}
	std::vector<double> c1x_2 = elementwise_mult(c1, x);
	std::vector<double> c0_cat_c1x = c0;
	c0_cat_c1x.insert(c0_cat_c1x.end(), c1x_2.begin(), c1x_2.end());
	return c0_cat_c1x;
}

std::vector<double> legder(std::vector<double> c) {
	//c = np.array(c, ndmin = 1, copy = True)
	//	if c.dtype.char in '?bBhHiIlLqQpP':
	//c = c.astype(np.double)
	//	cnt = pu._deprecate_as_int(m, "the order of derivation")
	//	iaxis = pu._deprecate_as_int(axis, "the axis")
	//	if cnt < 0 :
	//		raise ValueError("The order of derivation must be non-negative")
	//		iaxis = normalize_axis_index(iaxis, c.ndim)


	// c = np.moveaxis(c, iaxis, 0)

	int n = c.size();
	for (int i{ 0 }; i < 1; i++)
	{
		n = n - 1;
		// der = np.empty((n, ) + c.shape[1:], dtype = c.dtype)
		std::vector<double> der;
		for (int j{ n }; j >= 2; j--)
		{
			der[j - 1] = (2 * j - 1) * c[j];
			c[j - 2] += c[j];
		}
		if (n > 1)
			der[1] = 3 * c[2];	
		der[0] = c[1];
		c = der;

	}
	// c = np.moveaxis(c, 0, iaxis);
	return c;
}

//hjälpreda till leggauss
LapackMat* legcompanion(std::vector<double> c) {
	int N = c.size();
	if (N == 2) {
		LapackMat* mat = new LapackMat(1,1);
		return mat;
	}
	LapackMat* mat = new LapackMat(N, N); //Skapar matris mat = zeros(N*N)
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


/* Get quadrature points and weights for an interval [a,b]*/
Two_vectors gauss_legendre_line_mesh(int N, int a, int b) {
	Two_vectors X = leggauss(N);
	std::vector<double> p = X.v1;
	std::vector<double> w_prime = X.v2;

	/* Translate p and w_prime values for [-1,1] to a general
	 * interval [a,b]  for quadrature points k and weights w */
	std::vector<double> k{};
	std::vector<double> w{};
	for (int j{ 0 }; j < p.size(); j++)
	{
		k[j] = 0.5 * (p[j] + 1) * (b - a) + a;
		w[j] = w_prime[j] * 0.5 * (b - a);
	}

	Two_vectors k_and_w{ k, w };

	return k_and_w;
}


/* This follows equation (2.18) and (2.19) with the same notation */
Two_vectors gauss_legendre_inf_mesh(int N, double scale) {
	Two_vectors X = leggauss(N);
	std::vector<double> p = X.v1;
	std::vector<double> w_prime = X.v2;


	/* Translate p and w_prime values for [-1,1] to a infinte
	 * interval [0,inf]  for quadrature points k and weights w */
	std::vector<double> k{};
	std::vector<double> w{};
	for (int j{ 0 }; j < p.size(); j++)
	{
		k[j] = (1 + p[j]) / (1 - p[j]);
		w[j] = 2 * scale * w_prime[j] / pow(1 - k[j], 2);
	}

	Two_vectors k_and_w{ k, w };

	return k_and_w;
}


