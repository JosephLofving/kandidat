#include "lapackAPI.h"

/* Function called by all constructors in order to minimize duplicate code.
   @param width The width of the matrix (the amount of columns)
   @param height The height of the matrix (the amount of rows)
   @param contents The contents of the vector in column major order
*/
void LapackMat::init(int x, int y, std::vector<std::complex<double>> z) {
		width = x;
		height = y;
		contents = z;
}

/* Default constructor.
   @param @param width The width of the matrix (the amount of columns)
   @param height The height of the matrix (the amount of rows)
   @param contents The contents of the vector in column major order
*/
LapackMat::LapackMat(int x, int y, std::vector<std::complex<double>> z) {
	LapackMat::init(x, y, z);
}

/* Zero-matrix constructor. Creates a zero-filled matrix with given dimensions.
   @param @param width The width of the matrix (the amount of columns)
   @param height The height of the matrix (the amount of rows)
*/
LapackMat::LapackMat(int x, int y) {
	LapackMat::init(x, y, std::vector<std::complex<double>>(x*y, 0.0));
}

/* Identity matrix constructor. Creates an identity matrix with given size.
   @param @param size The size (width and height, which are equal) of the matrix.
*/
LapackMat::LapackMat(int x) {
	LapackMat::init(x, x, std::vector<std::complex<double>>(x*x, 0.0)); // Creates a zero-filled matrix with given size.

	for (int i = 0; i < x; i++) {
		this->setElement(i, i, 1.0); // Sets all diagonal elements to 1.0.
	}
}

/* Element getter.
   @param row The row (zero-indexed) of the element.
   @param col The column (zero-indexed) of the element.
   @return The element at given indices.
*/
std::complex<double> LapackMat::getElement(int row, int col) {
	if (row >= this->height) { // Ensures valid row index. Prints both indices for debugging purposes.
		std::cout << "Invalid row: " << row << " (Column: " << col << ")" << std::endl;
		abort();
	}
	if (col >= this->width) { // Ensures valid column index. Prints both indices for debugging purposes.
		std::cout << "Invalid column: " << col << " (Row: " << row << ")" << std::endl;
		abort();
	}
	return contents[row + col*height]; // The element at the given indices. Note that the matrix is in column major order.
}

/* Element setter.
   @param row The row (zero-indexed) of the element.
   @param col The column (zero-indexed) of the element.
   @param value The value to be set at the given indices.
*/
void LapackMat::setElement(int row, int col, std::complex<double> value) {
	if (row >= this->height) { // Ensures valid row index. Prints both indices for debugging purposes.
		std::cout << "Invalid row: " << row << " (Column: " << col << ")" << std::endl;
		abort();
	}
	if (col >= this->width) { // Ensures valid column index. Prints both indices for debugging purposes.
		std::cout << "Invalid column: " << col << " (Row: " << row << ")" << std::endl;
		abort();
	}
	contents[row + col*height] = value; // Note that the matrix is in column major order.
}

/* Prints the contents of the matrix.
*/
void LapackMat::print() {
	std::cout.precision(2); // Amount of decimals to be printed.
	std::cout << std::scientific; // Ensures scientific notation.

	for (int currentRow = 0; currentRow < height; currentRow++) {
		for (int currentCol = 0; currentCol < width; currentCol++) {
			std::complex<double> element = getElement(currentRow, currentCol);
			std::cout << element.real() <<"i \t"; // Prints the real and imaginary part and aligns to terminal tabs
    	}
    	std::cout << std::endl; // Line break for new row.
  	}
}

extern "C" { // Allows the usage of the following C function. Its arguments are explained below.
  void zgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, std::complex<double>* ALPHA, std::complex<double>* A, int* LDA, std::complex<double>* B, int* LDB, std::complex<double>* BETA, std::complex<double>* C, int* LDC);
}

/* The following functions use zgemm_ a lot. zgemm_ performs the operation C = alpha*op(A)*op(B) + beta*C
 * where op(X) is either a transposition or nothing at all. Non obvious arguments are as follows:
 * TRANSX: Whether X should be transposed. TRANSX='N' means no transposition.
 *      M: The height (amount of rows) in A (and C, by mathematical necessity)
 * 	 	N: The width (amount of columns) in B (and C, by mathematical necessity)
 *      K: The width of A and height of B (again by mathematical necessity)
 *    LDX: Leading dimension of X. Can be adjusted to use submatrices, but that's not something we use.
 * Note that the arguments are pointers, and that zgemm_ manipulates existing matrices rather than returning anything.
*/

/* Matrix addition
   @param &A First matrix. (Dimensions m*n)
   @param &B Second matrix. (Dimensions m*n)
   @return A+B. (Dimensions m*n)
*/
LapackMat operator+(LapackMat A, LapackMat B) {
	LapackMat dummyB   = LapackMat(B.width, B.height, B.contents); // Clones the B matrix. zgemm_ modifies the B matrix, so a cloning is performed to not lose any data
	LapackMat identity = LapackMat(A.height); // Identity matrix. A*I + B = A + B
	char TRANS = 'N'; // No transposition
	std::complex<double> ALPHA = 1; // Scale the A matrix by 1 (in other words, do nothing)
	std::complex<double> BETA  = 1; // Scale the B matrix by 1

	zgemm_(&TRANS, &TRANS, &A.height, &identity.width, &A.width, &ALPHA, A.contents.data(), &A.height, identity.contents.data(), &A.width, &BETA, dummyB.contents.data(), &A.height);

	return dummyB;
}

/* Matrix subtraction
   @param &A First matrix. (Dimensions m*n)
   @param &B Second matrix. (Dimensions m*n)
   @return A-B. (Dimensions m*n)
*/
LapackMat operator-(LapackMat A, LapackMat B) {
	LapackMat dummyB   = LapackMat(B.width, B.height, B.contents); // Clones the B matrix. zgemm_ modifies the B matrix, so a cloning is performed to not lose any data
	LapackMat identity = LapackMat(A.height); // Identity matrix. A*I - B = A - B
	char TRANS = 'N'; // No transposition
	std::complex<double> ALPHA = 1; // Scale the A matrix by 1 (in other words, do nothing)
	std::complex<double> BETA  = -1; // Scale the B matrix by -1 to cause a subtraction

	zgemm_(&TRANS, &TRANS, &A.height, &identity.width, &A.width, &ALPHA, A.contents.data(), &A.height, identity.contents.data(), &A.width, &BETA, dummyB.contents.data(), &A.height);

	return dummyB;
}

/* Matrix scalar multiplication
   @param scalar Value to scale A by.
   @param &A Matrix to be scaled. (Dimensions m*n)
   @return scalar*A. (Dimensions m*n)
*/
LapackMat operator*(std::complex<double> scalar, LapackMat A) {
	LapackMat B = LapackMat(A.height); // Creates an identity matrix in order to preserve A when multiplying. scalar*A*I + C = scalar*A + C
	LapackMat C(A.width, A.height); // Creates a zero matrix in order to not add anything. scalar*A*I + 0 = scalar*A
	char TRANS = 'N'; // No transposition
	std::complex<double> BETA = 0; // Scales the zero matrix (C) by 0. Since C is already 0 this isn't really necessary

	zgemm_(&TRANS, &TRANS, &A.height, &B.width, &A.width, &scalar, A.contents.data(), &A.height, B.contents.data(), &A.width, &BETA, C.contents.data(), &A.height);

	return C; // zgemm_ modifies C to contain the results of the operation. See the comment above 'extern "C"' above for further explanation.
}

/* Matrix scalar multiplication
   @param scalar Value to scale A by.
   @param &A Matrix to be scaled. (Dimensions m*n)
   @return scalar*A. (Dimensions m*n)
*/
LapackMat operator*(LapackMat A, std::complex<double> scalar) {
	return scalar*A; // Scalar multiplication is commutative. This function simply rearranges the arguments and calls the other scalar multiplication function.
}

/* Matrix multiplication
   @param &A First matrix. (Dimensions m*k)
   @param &B Second matrix. (Dimensions k*n)
   @return A*B. (Dimensions m*n)
*/
LapackMat operator*(LapackMat A, LapackMat B) {
	LapackMat C(A.height, B.width); // Creates a matrix to write the product to. Initializes to all zeroes in order to not add anything to the product.
	char TRANS = 'N'; // No transposition
	std::complex<double> ALPHA = 1; // Scale A*B by 1 (in other words, do nothing)
	std::complex<double> BETA  = 0; // Scale C by 0 (C is already all zeroes, and this isn't actually necessary)

	zgemm_(&TRANS, &TRANS, &A.height, &B.width, &A.width, &ALPHA, A.contents.data(), &A.height, B.contents.data(), &A.width, &BETA, C.contents.data(), &A.height);

	return C;
}

extern "C" {
	void zgetrf_(int* M, int *N, std::complex<double>* A, int* lda, int* IPIV, int* INFO);
	void zgetrs_(char* C, int* N, int* NRHS, std::complex<double>* A, int* LDA, int* IPIV, std::complex<double>* B, int* LDB, int* INFO);
}

/* Solves matrix equations of the form AX = B and returns X. The command zgetrf_ creates a partial pivot LU-decomposition. The
 * L and U matrices and the pivot vector are then provided to zgetrs_, which uses them to calculate X.
 * Note, however, that the L and U matrices are combined. zgetrf_ takes a pointer to the A matrix and replaces its contents
 * with a combination of the L and U matrices (the strict lower left triangle being U (though lacking an identity diagonal) and
 * the upper right triangle being U). To avoid destroying the initial contents of A this way (and B, which is destroyed similarly
 * for reasons irrelevant to us) two dummy matrices are created and used as arguments for the functions. */

/* Solves a matrix equation of the form A*X = B.
   @param A Matrix A. (Dimensions m*n)
   @param B Matrix B. (Dimensions m*nrhs)
   @return Matrix X.
*/
LapackMat solveMatrixEq(LapackMat A, LapackMat B) {
	LapackMat dummyA = LapackMat(A.width, A.height, A.contents); // zgetrf_ and zgetrs_ manipulate A and B. Dummies are created to avoid loss of data
	LapackMat dummyB = LapackMat(B.width, B.height, B.contents);

	int INFO; // Gives success status on output
	char TRANS = 'N'; // No transposition
	std::vector<int> IPIV(std::min(A.width, A.height)); // Permutation vector. Output from zgetrf_, input to zgetrs_. Describes the permutations performed in the LU-decomposition

	zgetrf_(&A.height, &A.width, dummyA.contents.data(), &A.height, IPIV.data(), &INFO); // LU-decomposition
	zgetrs_(&TRANS, &A.height, &B.width, dummyA.contents.data(), &A.width, IPIV.data(), dummyB.contents.data(), &A.height, &INFO); // Calculates X

	return dummyB; // dummyB has been overwritten by X
}

extern "C" {
	void zheev_(char* JOBZ, char* UPLO, int* N, std::complex<double>* A, int* LDA, double* W, std::complex<double>* WORK, int* LWORK, double* RWORK, int* INFO);
}

/* Computes the eigenvalues of a hermitian matrix A.
   @param A Hermitian matrix.
   @return Eigenvalues of A.
*/
std::vector<double> eigenValues(LapackMat A) {
	LapackMat dummyA = LapackMat(A.width, A.height, A.contents); // A is manipulated by zheev_, so a dummy is created.

	char JOBZ = 'N'; // Compute eigenvalues only. 'V' for eigenvalues and eigenvectors
	char UPLO = 'U'; // Use the upper triangle of A for the calculations. The lower triangle is inferred from the upper because the matrix is hermitian
	std::vector<double> W(A.width); // Vector to store eigenvalues.
	int LWORK = 2*A.width-1; // Length of the WORK vector
	std::vector<double> RWORK(3*A.width-2); // I think this is used internally by the function, but I don't know really
	std::vector<std::complex<double>> WORK(LWORK); // Same as for RWORK
	int INFO = 0; // Success integer

	zheev_(&JOBZ, &UPLO, &A.width, dummyA.contents.data(), &A.width, W.data(), WORK.data(), &LWORK, RWORK.data(), &INFO);

	return W;
}