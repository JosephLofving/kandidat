#ifndef CONSTANTS_H //när man importerar denna så kollar den så att man inte redan har importerat den. Så kompilatorn inte laddar den flera gånger.
#define CONSTANTS_H //tror denna namnger kodblocket

namespace constants {
	inline constexpr double pi{ 3.1415926535 }; // C++ verkar inte ha pi smidigt definierat (include <cmath>?)
	inline constexpr double rad2deg{ 180 / pi };		 
	inline constexpr double proton_mass{ 938.27208816 };														  // protonmassa (MeV/c^2)
	inline constexpr double neutron_mass{ 939.56542052 };													   	  // neutronmassa (MeV/c^2)
	inline constexpr double nucleon_average_mass { (proton_mass + neutron_mass) / 2 };							  // nucleon average mass
	inline constexpr double nucleon_reduced_mass { (proton_mass * neutron_mass) / (proton_mass + neutron_mass) }; // nucleon reduced mass
}
// when using constants in another file, use e.g. constants::rad2deg

#endif