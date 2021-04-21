/*
	This class creates the weights and quadrature points for the Gauss-Legendre quadrature.
	Use gaussLegendreInfMesh for improper integrals and gaussLegendreLineMesh for integrals on a finite interval.
*/

#include "mesh.h"


/*
	Multiplies each corresponding element from the two input vectors.
	@param v1, v2: two vectors of the same length
	@return the vector with multiplied elements
*/
std::vector<double> elementwiseMult(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> vec(v1.size(), 0);

	for (int i = 0; i < v1.size(); i++) {
		vec[i] = v1[i] * v2[i];
	}
	return vec;
}

/*
	Adds each corresponding element from the two input vectors.
	@param v1 (vector of length n)
	@param v2 (vector of length n)
	@return the vector with added elements
*/
std::vector<double> elementwiseAdd(std::vector<double> v1, std::vector<double> v2) {
	std::vector<double> vec(v1.size(), 0);

	for (int i = 0; i < v1.size(); i++) {
		vec[i] = v1[i] + v2[i];
	}
	return vec;
}

/*
	Reverses a vector
	@param v1, a vector
	@return The reversed vector
*/
std::vector<double> vecReverse(std::vector<double> v1) {
	std::reverse(v1.begin(), v1.end());
	return v1;
}

/*
	Sums all the elements of a vector
	@param v1, a vector
	@return The sum
*/
double vecSum(std::vector<double> v1) {
	double sum = 0;
	std::for_each(v1.begin(), v1.end(), [&](double v) {
		sum += v;
		});
	return sum;
}

/*
	Multiplies each vector element with a constant
	@param v1, a vector; a, a constant
	@return The vecScaled vector
*/
std::vector<double> vecScale(double a, std::vector<double> v1) {
	std::vector<double> w(v1.size());
	for (int i = 0; i < v1.size(); i++) {
		w[i] = a * v1[i];
	}

	return w;
}

/*
	Removes the first vector element, the vector is now 1 shorter
	@param v1, a vector
	@return A 1 element shorter vector
*/
std::vector<double> vecRemoveFirst(std::vector<double> v1) {
	v1.erase(v1.begin());
	return v1;
}

/*
	Removes the last vector element, the vector is now 1 shorter
	@param v1, a vector
	@return A 1 element shorter vector
*/
std::vector<double> vecRemoveLast(std::vector<double> v1) {
	v1.pop_back();
	return v1;
}

/*
	Creates a vector of size N with elements (a, a+1, a+2,..., N+a-2, N+a-1).
	@param a, the starting value; N, the vector size.
	@return A vector with N elements: (a, a+1, a+2,..., N+a-2, N+a-1).
*/
std::vector<double> iota(int a, int N) {
	std::vector<double> v(N);
	std::iota(std::begin(v), std::end(v), a);
	return v;
}

/*
	Finds the largest element in a vector according to its absolute size.
	@param v1: a vector
	@return The absolute-largest element
*/
double absmax(std::vector<double> v1) {
	std::vector<double> v(v1.size(), 0);
	for (int i = 0; i < v1.size(); ++i) {
		v[i] = std::abs(v1[i]);
	}

	double max = 0;
	for (int j{ 0 }; j < v.size(); j++)
	{
		if (v[j] > max) {
			max = v[j];
		}
	}
	return max;
}

/*
	Computes the sample points and weights for Gauss - Legendre quadrature.
	int_-1^1 f(x)dx = sum_x_n w(x_n)f(x_n)
	@param int N, the number of quadrature points
	@returm Two vectors, first one is the quadrature points, second is the weights.
*/
TwoVectors leggauss(int N) {
	if (N < 3) {
		std::cout << "Number of quadrature points must be > 2\n";
		abort();
	}

	/*
		We wish to obtain the N roots to a Legendre polynomial of degree N.
		These are our quadrature points x. x.size() = N
	*/
	//first approximation of roots. We use the fact that the companion
	//matrix is symmetric in this case in order to obtain better zeros.
	std::vector<double> c(N+1, 0);
	c[N] = 1; //c = (0,0,0,...,0,1), c.size() = N+1
	LapackMat* m = legcompanion(N); //see legcompanion
	std::vector<double> x = eigenValues(*m);

	//We improve the roots by using one application of Newton's method
	std::vector<double> function = legval(x, c);
	std::vector<double> derivative = legval(x, legder(c));
	for (int i = 0; i < x.size(); ++i) {
		x[i] -= function[i] / derivative[i];
	}

	//Now we compute the weights. We scale the factor to avoid possible numerical overflow.
	std::vector<double> function2 = legval(x, vecRemoveFirst(c));
	std::vector<double> w(N);
	//std::cout << "AAAAAAAAAAAAAAAAAAA:  " << w.size() << "\n";

	double constant = absmax(function2) * absmax(derivative);
	for (int i = 0; i < function2.size(); ++i) {
		w[i] = constant / (function2[i] * derivative[i]);
	}

	//We have to symmetrize and scale w to get the correct values
	std::vector<double> wReverse = vecReverse(w);
	std::vector<double> xReverse = vecReverse(x);
	for (int i = 0; i < w.size(); ++i) {
		w[i] = (w[i] + wReverse[i]) / 2;
		x[i] = (x[i] - xReverse[i]) / 2;
	}

	double wSum = vecSum(w);
	for (int i = 0; i < w.size(); ++i) {
		w[i] *= 2 / wSum;
	}

	TwoVectors xAndW{ x, w };
	return xAndW;
};


//c.size >= 3 ty.
/*
	Evaluates the Legendre series p(x) = c[0]L_0(x) + c[1]L_1(x) + ... c[N-1]L_N-1(x)
	at points x
	@param x: Points to evaluate series at; c: coefficients
	@return The evaluated values p(x)
*/
std::vector<double> legval(std::vector<double> x, std::vector<double> c) {
	int ND = c.size();
	int nd = ND*1.0; // Double kanske är onödigt

	std::vector<double> c0(x.size());
	std::vector<double> c1(x.size());
	std::vector<double> identityVector(x.size(), 1.0);

	c0[0] = c[nd-2];
	c1[0] = c[nd-1];

	for (int i = 3; i < ND + 1; ++i) {
		std::vector<double> tmp = c0;
		nd -= 1;

		/* c0 = c[-i] - c1*(nd - 1) / nd
		 * c1 = tmp + c1*x*(2*nd - 1) / nd */
		if (i == 3) { // c0 and c1 are effectively scalars for the first iteration
			c0 = vecScale(c[ND-i] + c1[0] * (1.0 - nd) / nd, identityVector);
			c1 = elementwiseAdd(vecScale(tmp[0], identityVector), vecScale((c1[0]*(2.0*nd-1.0))/nd, x));
		} else {
			c0 = elementwiseAdd(vecScale(c[ND-i], identityVector), vecScale((1.0-nd)/nd, c1));
			c1 = elementwiseAdd(tmp, vecScale((2.0*nd-1.0)/nd, elementwiseMult(c1, x)));
		}
	}

	return elementwiseAdd(c0, elementwiseMult(c1, x)); // c0 + c1*x
}

/*
	Differentiates a Legendre series with coefficients c.
	Note that the differentiation differens from normal power series differentiation.
	@param Coefficients c of a Legendre series
	@return Coefficients of the differentiated Legendre series.
*/
std::vector<double> legder(std::vector<double> c) {
	int N = c.size() - 1; //Samma N som i leggauss
	std::vector<double> der(c.size());

	for (int j = N; j >= 2; j--)
	{
		der[j - 1] = (2 * j - 1) * c[j];
		c[j - 2] += c[j];
	}

	der[1] = 3 * c[2];
	der[0] = c[1];

	return der;
}

//hjälpreda till leggauss
LapackMat* legcompanion(int N) {
	std::vector<double> scl = iota(0, N); //scl = (0,1,2,...,N-1)

	for (int i = 0; i < scl.size(); i++) {
		scl[i] = 1/sqrt(2 * scl[i] + 1);
	}

	std::vector<double> top = elementwiseMult(elementwiseMult(vecRemoveLast(scl), vecRemoveFirst(scl)), iota(1, N - 1)); //iota(1, N - 1) = (1,2,3,...,N-1)

	LapackMat* mat = new LapackMat(N, N); //Matrix mat = zeros(N*N)
	for (int i = 0; i < N-1; i++) {
		mat->setElement(i, i + 1, top[i]);
		mat->setElement(i + 1, i, top[i]);
	}

	return mat;
}


/* Get quadrature points and weights for an interval [a,b]*/
// kAndWPtrs gaussLegendreLineMesh(int N, int a, int b) {
// 	TwoVectors X = leggauss(N);
// 	std::vector<double> p = X.v1;
// 	std::vector<double> wPrime = X.v2;

// 	/* Translate p and wPrime values for [-1,1] to a general
// 	 * interval [a,b]  for quadrature points k and weights w */
// 	std::vector<double> k(p.size(), 0);
// 	std::vector<double> w(p.size(), 0);
// 	for (int j{ 0 }; j < p.size(); j++)
// 	{
// 		k[j] = 0.5 * (p[j] + 1) * (b - a) + a;
// 		w[j] = wPrime[j] * 0.5 * (b - a);
// 	}

// 	double* kk = &k[0];
// 	double* ww = &w[0];

// 	kAndWPtrs kAndW{ kk, ww };

// 	return kAndW;
// }


/* This follows equation (2.18) and (2.19) with the same notation */
void gaussLegendreInfMesh(double* k_h, double* w_h, int N, double vecScale) {
	TwoVectors X = leggauss(N);
	std::vector<double> p = X.v1;
	std::vector<double> wPrime = X.v2;


	/* Translate p and wPrime values for [-1,1] to a infinte
	 * interval [0,inf]  for quadrature points k and weights w */
	std::vector<double> k(p.size());
	std::vector<double> w(wPrime.size());
	for (int j{ 0 }; j < p.size(); j++)
	{
		k[j] = vecScale * std::tan(constants::pi * (p[j] + 1.0) / 4);
		w[j] = wPrime[j] * vecScale * (constants::pi / 4) / pow(std::cos(constants::pi * (p[j] + 1.0) / 4), 2);
	}
	for(int i =0; i< N; ++i){
		k_h[i] = k[i];
		w_h[i] = w[i];
	}
}


