#ifndef CUBLASAPI_H
#define CUBLASAPI_H

#include <iostream>
#include <vector>
#include <string> // Behövs för numberToFixedWidth()
#include <complex>

// A simple class for LAPACK-matrices
class Tensor {
public:
	int width;  // Amount of columns in each matrix
	int height; // Amount of rows in each matrix
	int matrixAmt; // Amount of matrices
	std::vector<std::complex<double>> contents; // Contents of all matrices in column-major order

	Tensor(int x, int y, int z); // Default constructor. Creates a zero-filled tensor
	Tensor(int x); // Creates an identity matrix with given dimension

	std::complex<double> getElement(int matrixNum, int row, int col); // Returns the element at given indices
	void setElement(int matrixNum, int row, int col, std::complex<double> value); // Sets the element at given indices

	void print(int matrixNum); // Prints the matrix with the given number

    // friend Tensor operator+(Tensor A, Tensor B);  // Matrix addition. A+B
	// friend Tensor operator-(Tensor A, Tensor B);  // Matrix subtraction. A-B
	// friend Tensor operator*(std::complex<double> scalar, Tensor A); // Scalar multiplication. scalar*A
	// friend Tensor operator*(Tensor A, std::complex<double> scalar); // Scalar multiplication. A*scalar
	// friend Tensor operator*(Tensor A, Tensor B);  // Matrix multiplication. A*B

private:
	void init(int x, int y, int z); // Function called by all constructors. Avoids duplicate code
};

//struct TwoVectors { // Struct for storing two related vectors
//	std::vector<double> v1;
//	std::vector<double> v2;
//};

// Tensor solveMatrixEq(Tensor A, Tensor B); // Matrix solver. Solves AX=B and returns X
// std::vector<double> eigenValues(Tensor A); // Finds the eigenvalues of a symmetric real matrix

#endif