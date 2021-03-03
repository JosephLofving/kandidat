#ifndef ANGMOM_H
#define ANGMOM_H

#include <cstdlib> // Behövs för abs
#include <map>
#include <vector> 
#include <iterator>
#include <string>

bool triag(int a, int b, int ab);
bool kDelta(std::map<std::string, int> bra, std::map<std::string, int> ket, std::vector <std::string> qN);

#endif 