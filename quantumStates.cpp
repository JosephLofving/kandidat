#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <iterator>
#include <math.h>
#include <list>
#include "angmom.h"

//g++ -std=c++17 helloworld.cpp -o main
// För att köre C++ ver 17

class QuantumState
{
public:
	std::map<std::string, int> state;

	void addQuantumNumber(std::string key, int value)
	{
		state.insert(std::pair<std::string, int>(key, value));
	}

	void printState()
	{
		std::map<std::string, int>::iterator itr;
		for (itr = state.begin(); itr != state.end(); ++itr)
		{
			std::cout << itr->first
					  << ":" << itr->second << ", ";
		}
	}


};

void printStates(std::vector<QuantumState> states){
	for (QuantumState state : states){
		std::cout << "[";
		state.printState();
		std::cout << "]" << std::endl;
	}
}

std::vector<QuantumState> setup_Base(int j2min, int j2max, int tzmin, int tzmax){
	std::vector<QuantumState> basis;
	for (int tz = tzmin; tz <= tzmax; tz++){
		for (int j = j2min; j <= j2max; j++){
			for (int s = 0; s < 2; s++){
				for (int l = abs(j - s); l <= j + s; l++){
					for (int t = abs(tz); t < 2; t++){
						if ((l + s + t) % 2 != 0){
							QuantumState basestate;
							basestate.addQuantumNumber("tz", tz);
							basestate.addQuantumNumber("l", l);
							basestate.addQuantumNumber("pi", pow(-1, l));
							basestate.addQuantumNumber("s", s);
							basestate.addQuantumNumber("j", j);
							basestate.addQuantumNumber("t", t);

							basis.push_back(basestate);
						}
					}
				}
			}
		}
	}
	return basis;
}

//std::multimap <double, QuantumState> setup_NN_channels(Basis base){
std::multimap<double, QuantumState> setup_NN_channels(std::vector<QuantumState> base){

	//std::vector<QuantumState> base = setup_Base(0, 2, 0, 0);

	std::vector<std::string> qN;
	qN.push_back("s");
	qN.push_back("j");
	qN.push_back("pi");
	qN.push_back("tz");

	std::vector<QuantumState> states;

	for (QuantumState bra : base){
		for (QuantumState ket : base){
			if (kDelta(bra.state, ket.state, qN)){
				QuantumState state;

				state.addQuantumNumber("l", bra.state["l"]);
				state.addQuantumNumber("ll", ket.state["l"]);

				state.addQuantumNumber("s", bra.state["s"]);
				state.addQuantumNumber("j", bra.state["j"]);
				state.addQuantumNumber("t", bra.state["t"]);
				state.addQuantumNumber("tz", bra.state["tz"]);
				state.addQuantumNumber("pi", bra.state["pi"]);

				states.push_back(state);
			}
		}
	}

	std::cout << states.size() << std::endl;

	for (QuantumState bs : states){
		std::cout << "[";
		bs.printState();
		std::cout << "]" << std::endl;
	}

	std::multimap<double, QuantumState> channels;

	double key;

	for (QuantumState state : states){
		key = state.state["j"] * 1.0 / 3.0 + state.state["s"] * 1.0 / 5.0 + state.state["tz"] * 1.0 / 11.0 + state.state["pi"] * 1.0 / 13.0;
		channels.insert(std::pair<double, QuantumState>(key, state));
		std::cout << key << ", " << std::endl;
	}

	std::multimap<double, QuantumState>::iterator itr;
	for (itr = channels.begin(); itr != channels.end(); ++itr){
		std::cout << itr->first << " ";
		itr->second.printState();
	}

	return channels;
}
