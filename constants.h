#ifndef CONSTANTS_H 
#define CONSTANTS_H

namespace constants {
	constexpr double pi = 3.1415926535; 
	constexpr double rad2deg = 180 / pi;		 

	/* Unit MeV/c^2. */
	constexpr double protonMass = 938.27208816;											
	constexpr double neutronMass = 939.56542052;											
	constexpr double nucleonAverageMass = (protonMass + neutronMass) / 2;						  
	constexpr double nucleonReducedMass = (protonMass * neutronMass) / (protonMass + neutronMass); 
}

#endif