#ifndef POTENTIAL_H
#define POTENTIAL_H

#include "chiral_LO.h"
#include "constants.h"
#include "cublasAPI.h"
#include "quantumStates.h"

double get_k0(std::vector<QuantumState> channel, double Tlab);
LapackMat potential(std::vector<QuantumState> channel, std::vector<double> k, std::vector<double>  Tlab);

#endif