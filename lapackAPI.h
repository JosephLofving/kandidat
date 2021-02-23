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
	lapackMat(int x);

	double getElement(int row, int col);
	void setElement(int row, int col, double value);
	void print();
};

lapackMat matrixMultiplication(lapackMat A, lapackMat B);
lapackMat scalarMultiplication(double scalar, lapackMat A);
lapackMat matrixAddition(lapackMat A, lapackMat C);
lapackMat matrixSubtraction(lapackMat A, lapackMat C);
lapackMat solveMatrixEq(lapackMat A, lapackMat B);

#endif