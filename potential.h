#ifndef POTENTIAL_H
#define POTENTIAL_H

#include "chiral_LO.h"
#include "constants.h"
#include "lapackAPI.h"
#include "quantumStates.h"

double getk0(std::vector<QuantumState> channel, double Tlab);
LapackMat potential(std::vector<QuantumState> channel, std::vector<double> k, double Tlab);

#endif