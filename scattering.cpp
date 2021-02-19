#include <tuple>
#include <cmath>
#include "constants.h"

std::tuple<double, double, double> blattToStapp(double deltaMinusBB, double deltaPlusBB, double twoEpsilonJBB) {
	double twoEpsilonJ = asin(sin(twoEpsilonJBB)*sin(deltaMinusBB - deltaPlusBB));
	double deltaMinus  = 0.5*(deltaPlusBB + deltaMinusBB + asin(tan(twoEpsilonJ)/tan(twoEpsilonJBB)))*rad2deg;
	double deltaPlus   = 0.5*(deltaPlusBB + deltaMinusBB - asin(tan(twoEpsilonJ)/tan(twoEpsilonJBB)))*rad2deg;
	double epsilon     = 0.5*twoEpsilonJ*rad2deg;

	return {deltaMinus, deltaPlus, epsilon};
}