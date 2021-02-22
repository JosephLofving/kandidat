#include <cstdlib> // Behövs för abs
#include <map>
#include <vector>
#include <iterator>
#include <string> 

bool triag(int a, int b, int ab){ // Kontrollerar |a-b| <= ab <= a+b
	return (abs(a - b) <= ab && ab <= a + b);
}

bool kDelta(std::map<std::string, int> bra, std::map<std::string, int> ket, std::vector <std::string> qN){
	for(std::string s: qN){
		if (bra[s] != ket[s]){
			return false;
		}
	}
	return true;
}