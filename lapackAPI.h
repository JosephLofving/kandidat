#ifndef LAPACKAPI_H
#define LAPACKAPI_H

#include <vector>

class lapackMat {
	public:
	int width;
	int height;
	std::vector<double> contents;

	lapackMat(int x, int y, std::vector<double> z);
	lapackMat(int x, int y);

	double matrixElement(int row, int col);
	void printMatrix();
};

lapackMat multiplyMatrix(lapackMat A, lapackMat B);

#endif