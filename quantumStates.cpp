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

/**
    Class QuantumState has a map named state.
	State stores every relavent quantum number qn as a key-value pair <string "qn", int qn>.
*/
class QuantumState
{
public:
	std::map<std::string, int> state;

	/**
    	Adds a key-value quantum number pair to the map state
    	@param key the string key that specifies what quantum number is added.
		@param value the int value that sets the value of the added quantum number
	*/
	void addQuantumNumber(std::string key, int value)
	{
		state.insert(std::pair<std::string, int>(key, value));
	}

	/**
    	Prints the quantum numbers and their values in the state
	*/
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
//End of QuantumState class

/**
    Prints all the quantum numbers of the QuantumStates in the input vector
    @param states The vector of Quantumstates that will be printed.
*/
void printStates(std::vector<QuantumState> states){
	for (QuantumState state : states){
		std::cout << "[";
		state.printState();
		std::cout << "]" << std::endl;
	}
}


/**
    Sets up the allowed quantum states for two nuclei, based on the specified maxiumum total angular momentum J
	and the iso-spin projection Tz. 
    @param Jmin Minimum allowed total angular momentum
	@param Jmax Maximum allowed total angular momentum
	@param Tzmin Minimum allowed iso-spin projection
	@param Tzmin Maximum allowed iso-spin projection
	@return The set of allowed quantum states as a vector
	
*/
std::vector<QuantumState> setupBasis(int Jmin, int Jmax, int Tzmin, int Tzmax){
	std::vector<QuantumState> basis; 
	for (int Tz = Tzmin; Tz <= Tzmax; Tz++){
		for (int J = Jmin; J <= Jmax; J++){
			for (int S = 0; S < 2; S++){ //Spin can be either 1 or 0
				for (int l = abs(J - S); l <= J + S; l++){ // J = L + S, thus the orbital angular momentum L has to be J-S
					for (int T = abs(Tz); T < 2; T++){ // Total iso-spin can either be 1 or 0
						if ((l + S + T) % 2 != 0){
							QuantumState basestate;
							basestate.addQuantumNumber("Tz", Tz);
							basestate.addQuantumNumber("l", l);
							basestate.addQuantumNumber("pi", pow(-1, l)); // The pairity https://en.wikipedia.org/wiki/Parity_(physics)
							basestate.addQuantumNumber("S", S);
							basestate.addQuantumNumber("J", J);
							basestate.addQuantumNumber("T", T);

							basis.push_back(basestate); //Adds the allowed state to the return vector
						}
					}
				}
			}
		}
	}
	return basis;
}

/**
    Calculates the allowed combination of states before and after the a scattering process. Based on the
	conserved quantum numbers: s,J,pi,Tz. The only unconserved quantum number is orbital momentum l. The
	quantum number ll represents l after scattering.
	https://www.asc.ohio-state.edu/physics/ntg/8805/notes/section_5_Scattering_2.pdf

	The function then groups the states based on the conserved quantum numbers.

	@param base A vector containing the allowed quantum states
	@return A map with key-value pairs. The key is a string that specifies the values of the conserved
			quantum numbers. The value is a vector of quantum states that stores all the states with
			these values (only ll can differ).
	
*/
std::map<std::string, std::vector<QuantumState> > setupNNChannels(std::vector<QuantumState> base){


	//A vector of strings that specify which quantum numbers to be conserved
	std::vector<std::string> conservedQuantumNumbers;
	conservedQuantumNumbers.push_back("S");
	conservedQuantumNumbers.push_back("J");
	conservedQuantumNumbers.push_back("pi");
	conservedQuantumNumbers.push_back("Tz");

	std::vector<QuantumState> allowedStates;

	//In this loop bra represents the state before scattering and ket the state after
	for (QuantumState bra : base){
		for (QuantumState ket : base){
			/** kDelta, checks if the value of the specified quantum numbers
			    are the same in bra and ket (before and after).
			*/
			if (kDelta(bra.state, ket.state, conservedQuantumNumbers)){ 
				QuantumState state;

				//Because the orbital momentum l is not conserved the ll can vary
				state.addQuantumNumber("l", bra.state["l"]);
				state.addQuantumNumber("ll", ket.state["l"]);
				
				//The rest are just copied
				state.addQuantumNumber("S", bra.state["S"]);
				state.addQuantumNumber("J", bra.state["J"]);
				state.addQuantumNumber("T", bra.state["T"]);
				state.addQuantumNumber("Tz", bra.state["Tz"]);
				state.addQuantumNumber("pi", bra.state["pi"]);

				allowedStates.push_back(state); //adds state
			}
		}
	}

	std::map<std::string, std::vector<QuantumState> > channels;
	std::string key;

	/**
	 * This loops through all the allowed states and stores the value of their conserved quantum numbers as a string.
	 * this string is then used as a key in the map channels. If the key is already present in the map the quantum state
	 * is added to the vector with that key.
	 */
	for (QuantumState state : allowedStates){
		key = "J:"+std::to_string(state.state["J"])+" S:"+std::to_string(state.state["S"])+" Tz:" +std::to_string(state.state["Tz"]) +" pi:"+ std::to_string(state.state["pi"]);
		if(channels.count(key)==0){ //checks if channel contais key. True if key is not present
			std::vector<QuantumState> vec_state;
			vec_state.push_back(state);
			channels.insert(std::pair<std::string, std::vector<QuantumState> >(key, vec_state));
		}
		else{// if the key is present, the state is added to the vector with that key.
			channels[key].push_back(state);
		}
		
	}


	return channels;
}

/**
    Prints all the channels and the quantum states in that channel.
    @param channels The map containing the channels.
*/
void printChannels(std::map<std::string, std::vector<QuantumState> > channels){
	std::map<std::string, std::vector<QuantumState> >::iterator itr;
	for (itr = channels.begin(); itr != channels.end(); ++itr){
		std::cout << "Key: [" << itr->first<< " ] " << std::endl;
		std::cout << " value: "<< std::endl;
		printStates(itr->second);
	}

}