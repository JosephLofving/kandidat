#ifndef LAPACKAPI_H
#define LAPACKAPI_H

#include <iostream>
#include <vector>
#include <string> // Behövs för numberToFixedWidth()

// A simple class for LAPACK-matrices
class LapackMat {
public:
	int width;  // Amount of columns in matrix
	int height; // Amount of rows in matrix
	std::vector<std::complex<double>> contents; // Matrix contents in column-major order

	LapackMat(int x, int y, std::vector<std::complex<double>> z); // Default constructor
	LapackMat(int x, int y); // Creates a matrix with given dimensions filled with zeroes
	LapackMat(int x); // Creates an identity matrix with given dimension

	std::complex<double> getElement(int row, int col); // Returns the element at given indices
	void setElement(int row, int col, std::complex<double> value); // Sets the element at given indices
	void print(); // Prints the matrix

	friend LapackMat operator+(LapackMat &A, LapackMat &B);  // Matrix addition. A+B
	friend LapackMat operator-(LapackMat &A, LapackMat &B);  // Matrix subtraction. A-B
	friend LapackMat operator*(std::complex<double> scalar, LapackMat &A); // Scalar multiplication. scalar*A
	friend LapackMat operator*(LapackMat &A, std::complex<double> scalar); // Scalar multiplication. A*scalar
	friend LapackMat operator*(LapackMat &A, LapackMat &B);  // Matrix multiplication. A*B

private:
	void init(int x, int y, std::vector<std::complex<double>> z); // Skapas separat från konstruktorerna för att göra dem mer koncisa
};

struct Two_vectors { // Struct for storing two related vectors
	std::vector<double> v1;
	std::vector<double> v2;
};

LapackMat solveMatrixEq(LapackMat A, LapackMat B); // Matrix solver. Solves AX=B and returns X
std::vector<double> eigenValues(LapackMat A); // Finds the eigenvalues of a symmetric real matrix

#endif