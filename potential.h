#pragma once //denna g�r tydligen s� koden bara inkluderas en g�ng. Vet ej hur n�dv�ndig den �r, men bra att veta att kommandot finns typ.
#include "chiral_LO.h"
#include "lapackAPI.h"
#include "quantumStates.h"

double get_k0(std::vector<QuantumState> channel, double Tlab);
LapackMat potential(std::vector<QuantumState> channel, std::vector<double> k, double Tlab);