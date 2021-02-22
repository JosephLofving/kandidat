#ifndef CONSTANTS_H //n�r man importerar denna s� kollar den s� att man inte redan har importerat den. S� kompilatorn inte laddar den flera g�nger.
#define CONSTANTS_H //tror denna namnger kodblocket

<<<<<<< HEAD
// when using constants in another file, use e.g. const::rad2deg
namespace const {
	inline constexpr double pi{ 3.1415926535 }; // C++ verkar inte ha pi smidigt definierat (include <cmath>?)
	inline constexpr double rad2deg{ 180 / pi }; // Don't know if this works, needed pi elsewhere
	inline constexpr double mp{ 938.27208816 }; //protonmassa (MeV/c^2)
	inline constexpr double mn{ 939.56542052 }; //neutronmassa (MeV/c^2)
	inline constexpr double mN{ (mp + mn) / 2 }; //nucleon average mass
	inline constexpr double uN{ (mp * mn) / (mp + mn) }; //nucleon reduced mass
=======
// when using constants in another file, use e.g. constants::rad2deg
namespace constants
{
	constexpr double pi{ 3.1415926535 }; // C++ verkar inte ha pi smidigt definierat (include <cmath>?)
	constexpr double rad2deg{ 180 / pi };
>>>>>>> 097a7d3e39ef259b163aa02bd53aed3be1c3a638
}

#endif