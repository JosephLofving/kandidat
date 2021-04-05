#include "cublasAPI.h"

void Tensor::init(int x, int y, int z) {
		width = x;
		height = y;
		matrixAmt = z;
		contents = std::vector<std::complex<double>>(x*y*z, 0.0);
}

Tensor::Tensor(int x, int y, int z) {
	Tensor::init(x, y, z);
}

Tensor::Tensor(int x) {
	Tensor::init(x, x, 1); // Creates a zero-filled matrix with given size.

	for (int i = 0; i < x; i++) {
		this->setElement(0, i, i, 1.0); // Sets all diagonal elements to 1.0.
	}
}

std::complex<double> Tensor::getElement(int matrixNum, int row, int col) {
	if (row >= this->height) { // Ensures valid row index. Prints both indices for debugging purposes.
		std::cout << "Invalid row: " << row << " (Column: " << col << ")" << std::endl;
		abort();
	}
	if (col >= this->width) { // Ensures valid column index. Prints both indices for debugging purposes.
		std::cout << "Invalid column: " << col << " (Row: " << row << ")" << std::endl;
		abort();
	}
	return contents[matrixNum*this->width*this->height + row + col*height]; // The element at the given indices. Note that the matrix is in column major order.
}

void Tensor::setElement(int matrixNum, int row, int col, std::complex<double> value) {
	if (row >= this->height) { // Ensures valid row index. Prints both indices for debugging purposes.
		std::cout << "Invalid row: " << row << " (Column: " << col << ")" << std::endl;
		abort();
	}
	if (col >= this->width) { // Ensures valid column index. Prints both indices for debugging purposes.
		std::cout << "Invalid column: " << col << " (Row: " << row << ")" << std::endl;
		abort();
	}
	contents[matrixNum*this->width*this->height + row + col*height] = value; // Note that the matrix is in column major order.
}

void Tensor::print(int matrixNum) {
	std::cout.precision(2); // Amount of decimals to be printed.
	std::cout << std::scientific; // Ensures scientific notation.

	for (int currentRow = 0; currentRow < height; currentRow++) {
		for (int currentCol = 0; currentCol < width; currentCol++) {
			std::complex<double> element = getElement(matrixNum, currentRow, currentCol);
			std::cout << element.real() << " + " << element.imag() << "i \t"; // Prints the real and imaginary part and aligns to terminal tabs
    	}
    	std::cout << std::endl; // Line break for new row.
  	}
}

void operator*(Tensor A, Tensor B) {

	const std::complex<double> alpha = 1.0; // Scale the A matrix by 1 (in other words, do nothing)
	const std::complex<double> beta  = 0.0; // Scale the B matrix by 1

	int m = A.width;
	int n = A.height;
	int k = A.matrixAmt;

	cublasHandle_t cublasH;
	cublasCreate(&cublasH);

	cublasDgemmStridedBatched(cublasH,
							  CUBLAS_OP_N,
							  CUBLAS_OP_N,
							  m, n, k,
							  &alpha,
							  A, m, m*n,
							  B, n, n*k,
							  &beta,
							  A, m, m*n,
							  32);
}