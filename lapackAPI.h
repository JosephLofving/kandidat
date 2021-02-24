#ifndef LAPACKAPI_H
#define LAPACKAPI_H

#include <vector>

class LapackMat {
	public:
	int width;
	int height;
	std::vector<double> contents;

	LapackMat(int x, int y, std::vector<double> z);
	LapackMat(int x, int y);
	LapackMat(int x);

	double getElement(int row, int col);
	void setElement(int row, int col, double value);
	void print();

	friend LapackMat operator+(LapackMat &A, LapackMat &B);
	friend LapackMat operator-(LapackMat &A, LapackMat &B);
	friend LapackMat operator*(double scalar, LapackMat &A);
	friend LapackMat operator*(LapackMat &A, double scalar);
	friend LapackMat operator*(LapackMat &A, LapackMat &B);
};

struct Two_vectors {
	std::vector<double> v1;
	std::vector<double> v2;
};

LapackMat matrixMultiplication(LapackMat A, LapackMat B);
LapackMat scalarMultiplication(double scalar, LapackMat A);
LapackMat solveMatrixEq(LapackMat A, LapackMat B);

#endif