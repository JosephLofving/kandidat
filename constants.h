#ifndef CONSTANTS_H //n�r man importerar denna s� kollar den s� att man inte redan har importerat den. S� kompilatorn inte laddar den flera g�nger.
#define CONSTANTS_H //tror denna namnger kodblocket

namespace constants {
	constexpr double pi{ 3.1415926535 }; // C++ verkar inte ha pi smidigt definierat (include <cmath>?)
	constexpr double rad2deg{ 180 / pi };		 
	constexpr double protonMass{ 938.27208816 };														  // protonmassa (MeV/c^2)
	constexpr double neutronMass{ 939.56542052 };													   	  // neutronmassa (MeV/c^2)
	constexpr double nucleon_average_mass { (protonMass + neutronMass) / 2 };							  // nucleon average mass
	constexpr double nucleon_reduced_mass { (protonMass * neutronMass) / (protonMass + neutronMass) }; // nucleon reduced mass
}
// when using constants in another file, use e.g. constants::rad2deg

#endif