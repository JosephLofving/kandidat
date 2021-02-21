#ifndef CONSTANTS_H
#define CONSTANTS_H

// when using constants in another file, use e.g. const::rad2deg
namespace constants
{
	constexpr double pi{ 3.1415926535 }; // C++ verkar inte ha pi smidigt definierat (include <cmath>?)
	constexpr double rad2deg{ 180 / pi }; // Don't know if this works, needed pi elsewhere
}

#endif