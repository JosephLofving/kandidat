#pragma once //denna g�r tydligen s� koden bara inkluderas en g�ng. Vet ej hur n�dv�ndig den �r, men bra att veta att kommandot finns typ.
#include "chiral_LO.h"
#include "quantumStates.h"

 std::vector<double> potential(std::vector<QuantumState> channel, std::vector<double> p);