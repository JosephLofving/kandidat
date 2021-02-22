#ifndef CONSTANTS_H //när man importerar denna så kollar den så att man inte redan har importerat den. Så kompilatorn inte laddar den flera gånger.
#define CONSTANTS_H //tror denna namnger kodblocket

// when using constants in another file, use e.g. const::rad2deg
<<<<<<< HEAD
namespace const {
	inline constexpr double pi{ 3.1415926535 }; // C++ verkar inte ha pi smidigt definierat (include <cmath>?)
	inline constexpr double rad2deg{ 180 / pi }; // Don't know if this works, needed pi elsewhere
	inline constexpr double mp{ 938.27208816 }; //protonmassa (MeV/c^2)
	inline constexpr double mn{ 939.56542052 }; //neutronmassa (MeV/c^2)
	inline constexpr double mN{ (mp + mn) / 2 }; //nucleon average mass
	inline constexpr double uN{ (mp * mn) / (mp + mn) }; //nucleon reduced mass
=======
namespace constants {
	inline constexpr double pi{ 3.1415926535 };			 // C++ verkar inte ha pi smidigt definierat (include <cmath>?)
	inline constexpr double rad2deg{ 180 / pi };		 // Don't know if this works, needed pi elsewhere
	inline constexpr double mp{ 938.27208816 };			 // protonmassa (MeV/c^2)
	inline constexpr double mn{ 939.56542052 };			 // neutronmassa (MeV/c^2)
	inline constexpr double mN{ (mp + mn) / 2 };		 // nucleon average mass
	inline constexpr double uN{ (mp * mn) / (mp + mn) }; // nucleon reduced mass
>>>>>>> 34460a4c6863ed27354a60c9743a6273a9f88c22
}
// when using constants in another file, use e.g. constants::rad2deg

#endif