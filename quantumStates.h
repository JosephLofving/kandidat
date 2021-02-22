#include <iostream>
#include <map>
#include <string>
#include <vector> 
#include <iterator>
#include <math.h> 
#include <list>
#include "angmom.h"

class QuantumState {
	std::map<std::string, int> state;
	public:
		void addQuantumNumber(std::string key, int value);
		void printState();
};
class Basis {
	std::vector <QuantumState> basis;
	public:
		Basis(int j2min, int j2max, int tzmin, int tzmax);
		void printBasis();
};

int setup_NN_channels();