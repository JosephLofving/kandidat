#ifndef MESH_H
#define MESH_H

#include "constants.h"
#include "lapackAPI.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <list>
#include <numeric> //for std::iota
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>

struct kAndWPtrs {
	double* k;
	double* w;
};

std::vector<double> elementwiseMult(std::vector<double> v1, std::vector<double> v2);
double absmax(std::vector<double> vec);
TwoVectors leggauss(int N);
std::vector<double> legval(std::vector<double> x, std::vector<double> c);
std::vector<double> legder(std::vector<double> c);
LapackMat* legcompanion(int N);
kAndWPtrs gaussLegendreLineMesh(int N, int a, int b);
kAndWPtrs gaussLegendreInfMesh(int N, double scale = 100.0);
std::vector<double> vecScale(double a, std::vector<double> v);

#endif