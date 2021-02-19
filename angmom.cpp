#include <cstdlib> // Behövs för abs

bool triag(int a, int b, int ab){ // Kontrollerar |a-b| <= ab <= a+b
	if ( abs(a - b) <= ab && ab <= a + b ) {
		return true;
	}
	return false;
}

bool kDelta(QuantumState bra, QuantumState ket, std::vector <std::string> qN){
	for(std::string s: qN){
		if (bra.state[s] != ket.state[s]){
			return false;
		}
	}
	return true;
}