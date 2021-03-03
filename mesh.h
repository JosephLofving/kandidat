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


std::vector<double> elementwise_mult(std::vector<double> v1, std::vector<double> v2);
double absmax(std::vector<double> vec);
Two_vectors leggauss(int N);
std::vector<double> legval(std::vector<double> x, std::vector<double> c);
std::vector<double> legder(std::vector<double> c);
LapackMat* legcompanion(std::vector<double> c);
Two_vectors gauss_legendre_line_mesh(int N, int a, int b);
Two_vectors gauss_legendre_inf_mesh(int N, double scale = 100.0);
std::vector<double> scale(double a, std::vector<double> v);

#endif