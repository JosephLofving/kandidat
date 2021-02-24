#include <iostream>
#include <map>
#include <string>
#include <vector> 
#include <iterator>
#include <math.h> 
#include <list>
#include "angmom.h"

class QuantumState {
	public:
		std::map<std::string, int> state;
		void addQuantumNumber(std::string key, int value);
		void printState();
};

//std::multimap <double, QuantumState> setup_NN_channels(Basis base);
void printStates(std::vector<QuantumState> states);
std::vector<QuantumState> setup_Base(int j2min, int j2max, int tzmin, int tzmax);
std::map<std::string, std::vector<QuantumState> > setup_NN_channels(std::vector<QuantumState> base)
