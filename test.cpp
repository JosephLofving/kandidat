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

int main() {
	N = 100;
	double sum{};
	for (int i; i < N; i++)
	{
		sum += k[i] * w[i];
	}
	std::cout << "The integral evaluates to approximately " << sum;

	return 0;
}