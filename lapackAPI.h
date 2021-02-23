#ifndef LAPACKAPI_H
#define LAPACKAPI_H

#include <vector>

class lapackMat {
	public:
	int width;
	int height;
	std::vector<double> contents;

	double matrixElement(int row, int col);
	void printMatrix();
};

#endif