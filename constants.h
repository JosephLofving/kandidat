#ifndef CONSTANTS_H //när man importerar denna så kollar den så att man inte redan har importerat den. Så kompilatorn inte laddar den flera gånger.
#define CONSTANTS_H //tror denna namnger kodblocket

namespace constants {
	constexpr double pi{ 3.1415926535 }; // C++ verkar inte ha pi smidigt definierat (include <cmath>?)
	constexpr double rad2deg{ 180 / pi };		 
	constexpr double proton_mass{ 938.27208816 };														  // protonmassa (MeV/c^2)
	constexpr double neutron_mass{ 939.56542052 };													   	  // neutronmassa (MeV/c^2)
	constexpr double nucleon_average_mass { (proton_mass + neutron_mass) / 2 };							  // nucleon average mass
	constexpr double nucleon_reduced_mass { (proton_mass * neutron_mass) / (proton_mass + neutron_mass) }; // nucleon reduced mass
}
// when using constants in another file, use e.g. constants::rad2deg

#endif