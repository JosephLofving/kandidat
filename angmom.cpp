#include <cstdlib> // Behövs för abs
#include "quantumStates.h"

bool triag(int a, int b, int ab){ // Kontrollerar |a-b| <= ab <= a+b
	return (abs(a - b) <= ab && ab <= a + b);
}

bool kDelta(QuantumState bra, QuantumState ket, std::vector <std::string> qN){
	for(std::string s: qN){
		if (bra.state[s] != ket.state[s]){
			return false;
		}
	}
	return true;
}