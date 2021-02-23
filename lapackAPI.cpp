#include <iostream>
#include <iomanip>
#include <vector>
#include <stdlib.h>
#include <string.h>

std::string numberToFixedWidth(double num, int width) {
	std::string s = std::to_string(num);
	if (s.length() > width) {
		s = s.substr(0, width);
	}
	s.append(width+1 - s.length(), ' ');

	return s;
}

class lapackMat {
	public:
	int width;
	int height;
	std::vector<double> contents;

	double matrixElement(int row, int col);
	void printMatrix();
};

double lapackMat::matrixElement(int row, int col) {
	return contents[col + row*height];
}

void lapackMat::printMatrix() {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			std::cout << numberToFixedWidth(matrixElement(i, j), 6);
    	}
    	std::cout << '\n';
  	}
}