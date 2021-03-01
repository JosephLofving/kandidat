﻿#include "constants.h"
#include "lapackAPI.h"
#include "mesh.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <list>
#include <numeric> //beh�vs f�r std::iota
#include <random>


// testing accuracy of gauss_legendre_inf_mesh
int main() {
	int N{ 100 };
	int scale{ 100 };

	Two_vectors k_and_w{ gauss_legendre_inf_mesh(N, scale) };
	std::vector<double> k{ k_and_w.v1 };
	std::vector<double> w{ k_and_w.v2 };

	double sum{};
	for (int i = 0; i < N; i++)
	{
		double function{sin(k[i])/k[i]}; // function of k[i]
		sum += function * w[i];
	}
	std::cout << "The integral evaluates to approximately " << sum << std::endl;

	return 0;
}