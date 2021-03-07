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


std::vector<double> elementwiseMult(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> vec(v1.size(), 0);

	for (int i = 0; i < v1.size(); i++) {
		vec[i] = v1[i] * v2[i];
	}
	return vec;
}

std::vector<double> elementwiseAdd(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> vec(v1.size(), 0);

	for (int i = 0; i < v1.size(); i++) {
		vec[i] = v1[i] + v2[i];
	}
	return vec;
}

//först opererar den abs() på alla vektorns element, och sen returnar den det största av dessa element.
double absmax(std::vector<double> vec) {
	std::vector<double> vec1(vec.size(), 0);
	for (int i = 0; i < vec.size(); ++i) {
		vec1[i] = std::abs(vec[i]);
	}

	double max = 0;
	for (int j{ 0 }; j < vec1.size(); j++) //max(vec1)
	{
		if (vec1[j] > max) {
			max = vec1[j];
		}
	}
	return max;
}

//Gauss-Legendre quadrature
//Computes the sample pointsand weights for Gauss - Legendre quadrature.
//These sample points and weights will correctly integrate polynomials of
//degree : math:`2*deg - 1` or less over the interval : math:`[ - 1, 1]` with
//the weight function : math:`f(x) = 1`.
TwoVectors leggauss(int N) {
	if (N < 3) {
		std::cout << "Index must be > 2\n";
	}
	//first approximation of roots.We use the fact that the companion
	//matrix is symmetric in this case in order to obtain better zeros.
	std::vector<double> c(N+1, 0);
	c[N] = 1; //nu är alltså alla element 0, förutom det sista som är 1.
	LapackMat* m = legcompanion(c); //se legcompanion
	std::vector<double> x = eigenValues(*m);

	//improve roots by one application of Newton
	std::vector<double> dy = legval(x, c);
	std::vector<double> df = legval(x, legder(c));

	for (int i = 0; i < x.size(); ++i) {
		x[i] -= dy[i] / df[i];
	}

	//compute the weights. We scale the factor to avoid possible numerical overflow.
	std::vector<double> cRemovedFirst = c;
	cRemovedFirst.erase(cRemovedFirst.begin()); //c med första elementet borttaget
	std::vector<double> fm = legval(x, cRemovedFirst);
	std::vector<double> w(fm.size());
	
	double fmAbsmax = absmax(fm);
	double dfAbsmax = absmax(df);
	for (int i = 0; i < fm.size(); ++i) {
		fm[i] /= fmAbsmax;
		df[i] /= dfAbsmax;
		w[i] = 1 / (fm[i] * df[i]);
	}

	//for Legendre we can also symmetrize
	std::vector<double> wReverse = w;
	std::reverse(wReverse.begin(), wReverse.end());
	std::vector<double> xReverse = x;
	std::reverse(xReverse.begin(), xReverse.end());
	for (int i = 0; i < w.size(); ++i) {
		w[i] = (w[i] + wReverse[i]) / 2;
		x[i] = (x[i] - xReverse[i]) / 2;
	}

	//scale w to get the right value
	double wSum;
	std::for_each(w.begin(), w.end(), [&](double v) {
		wSum += v;
		});
	for (int i = 0; i < w.size(); ++i) {
		w[i] *= 2 / wSum;
	}

	TwoVectors xAndW{ x, w };
	return xAndW;
};

std::vector<double> scale(double a, std::vector<double> v) {
	std::vector<double> w(v.size());
	for (int i = 0; i < v.size(); i++) {
		w[i] = a*v[i];
	}

	return w;
}

//c.size >= 3 ty.
std::vector<double> legval(std::vector<double> x, std::vector<double> c) {
	int ND = c.size();
	int nd = ND*1.0; // Double kanske är onödigt
	
	std::vector<double> c0(x.size());
	std::vector<double> c1(x.size());
	std::vector<double> identityVector(x.size(), 1.0);

	c0[0] = c[nd-2];
	c1[0] = c[nd-1];

	for (int i = 3; i < ND + 1; i++) {
		std::vector<double> tmp = c0;
		nd -= 1;
	
		/* c0 = c[-i] - c1*(nd - 1) / nd
		 * c1 = tmp + c1*x*(2*nd - 1) / nd */
		if (i == 3) { // c0 and c1 are effectively scalars for the first iteration
			c0 = elementwiseAdd(scale(c[ND-i], identityVector), scale((1.0-nd)/nd, scale(c1[0], identityVector)));
			c1 = elementwiseAdd(scale(tmp[0], identityVector), scale((c1[0]*(2.0*nd-1.0))/nd, x));
		} else {
			c0 = elementwiseAdd(scale(c[ND-i], identityVector), scale((1.0-nd)/nd, c1));
			c1 = elementwiseAdd(tmp, scale((2.0*nd-1.0)/nd, elementwiseMult(c1, x)));
		}
	}

	return elementwiseAdd(c0, elementwiseMult(c1, x)); // c0 + c1*x


	// std::vector<double> c0(x.size());
	// std::vector<double> c1(x.size());

	// int nd = c.size();
	// c0[0] = c[nd - 2]; //näst sista elementet i c
	// c1[0] = c[nd - 1]; //sista elementet i c
	// for (int i = 3; i < nd + 1; i++) {
	// 	std::vector<double> tmp = c0;
	// 	nd -= 1;
	// 	std::vector<double> c1_times_const(c0.size());
	// 	std::for_each(c1_times_const.begin(), c1_times_const.end(), [&](double v) {
	// 		v = c1[0] * (nd-1) / nd;
	// 		});
	// 	c0 = c1_times_const;
	// 	std::for_each(c0.begin(), c0.end(), [&](double v) {
	// 		v = c[nd-i] - v;
	// 		}); //c0 = c[nd - i] - c1_times_const;
	// 	std::vector<double> c1x = c1;
	// 	std::for_each(c1_times_const.begin(), c1_times_const.end(), [&](double v) {
	// 		v = c1[0] * (nd-1) / nd;
	// 		});
	// 	c1 = elementwiseAdd(tmp, c1x);
		
	// }
	// std::vector<double> c1x_2 = elementwiseMult(c1, x);
	// std::vector<double> vec = elementwiseAdd(c0, c1x_2);
	// return vec;
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
		std::vector<double> der(c.size());
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
	int N = c.size()-1;

	LapackMat* mat = new LapackMat(N, N); //Skapar matris mat = zeros(N*N)
	std::vector<double> scl(N);
	std::iota(std::begin(scl), std::end(scl), 0); //scl= {0,1,...,N-1}
	std::for_each(scl.begin(), scl.end(), [](double& v) {v = sqrt(2 * v + 1); }); //scl[i] = sqrt(2*scl[i]+1) forall i
	std::for_each(scl.begin(), scl.end(), [](double& v) { v = 1 / v; });

	std::vector<double> scll(N-1);
	std::iota(std::begin(scll), std::end(scll), 1); //scll= {1,...,N-1} (börjar på 1, men har denna gång endast N-1 element, till skillnad från scl).

	std::vector<double> sclRemovedLast = scl; //scl fast med sista elementet borttaget
	std::vector<double> sclRemovedFirst = scl; //scl fast med första elementet borttaget
	sclRemovedLast.pop_back();
	sclRemovedFirst.erase(sclRemovedFirst.begin());

	std::vector<double> sclProd = elementwiseMult(sclRemovedLast, sclRemovedFirst); //elementvis produkt mellan dessa
	std::vector<double> top = elementwiseMult(sclProd, scll);
	
	for (int i = 0; i < N-1; i++) {
		mat->setElement(i, i + 1, top[i]);
		mat->setElement(i + 1, i, top[i]);
	}

	return mat;

}


/* Get quadrature points and weights for an interval [a,b]*/
TwoVectors gaussLegendreLineMesh(int N, int a, int b) {
	TwoVectors X = leggauss(N);
	std::vector<double> p = X.v1;
	std::vector<double> wPrime = X.v2;

	/* Translate p and wPrime values for [-1,1] to a general
	 * interval [a,b]  for quadrature points k and weights w */
	std::vector<double> k(p.size(), 0);
	std::vector<double> w(p.size(), 0);
	for (int j{ 0 }; j < p.size(); j++)
	{
		k[j] = 0.5 * (p[j] + 1) * (b - a) + a;
		w[j] = wPrime[j] * 0.5 * (b - a);
	}

	TwoVectors kAndW{ k, w };

	return kAndW;
}


/* This follows equation (2.18) and (2.19) with the same notation */
TwoVectors gaussLegendreInfMesh(int N, double scale) {
	TwoVectors X = leggauss(N);
	std::vector<double> p = X.v1;
	std::vector<double> wPrime = X.v2;


	/* Translate p and wPrime values for [-1,1] to a infinte
	 * interval [0,inf]  for quadrature points k and weights w */
	std::vector<double> k(p.size());
	std::vector<double> w(wPrime.size());
	for (int j{ 0 }; j < p.size(); j++)
	{
		//k[j] = scale * (1 + p[j]) / (1 - p[j]);
		//w[j] = 2 * scale * wPrime[j] / pow(1 - k[j], 2);

		//test: copied from python
		k[j] = scale * std::tan(constants::pi * (p[j] + 1.0) / 4);
		w[j] = wPrime[j] * scale * (constants::pi / 4) / pow(std::cos(constants::pi * (p[j] + 1.0) / 4), 2);
	}

	TwoVectors kAndW{ k, w };

	return kAndW;
}


