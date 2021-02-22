#include <iostream>
#include <map>
#include <string>
#include <vector> 
#include <iterator>
#include <math.h> 
#include <list>

class QuantumState {
	public:
		std::map<std::string, int> state;
		void addQuantumNumber(std::string key, int value);
		void printState();
};
class Basis {
	Basis(int j2min, int j2max, int tzmin, int tzmax);
	void printBasis();
};

bool kDelta(QuantumState bra, QuantumState ket, std::vector <std::string> qN);
int setup_NN_channels();