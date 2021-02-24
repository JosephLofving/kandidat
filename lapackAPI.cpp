#include <iostream>
#include <vector>
#include <string> // Behövs för numberToFixedWidth()

std::string numberToFixedWidth(double num, int width) { // Gör num till en string av bredd width med efterföljande blanksteg
	std::string s = std::to_string(num); // Typkonversion
	if (s.length() > width) {
		s = s.substr(0, width); // Trunkerar strängen om den är för lång
	}
	s.append(width+1 - s.length(), ' '); // Fyller ut strängen och lägger till blanksteg

	return s;
}

class LapackMat {
public:
	int width;  // Matrisbredd
	int height; // Matrishöjd
	std::vector<double> contents;

	LapackMat(int x, int y, std::vector<double>); // Constructor
	LapackMat(int x, int y); // Constructor för nollmatris
	LapackMat(int x); // Constructor för identitetsmatris
	double getElement(int row, int col); // Returnerar element med givna index. Notera att allt är 0-indexerat
	void setElement(int row, int col, double value);
	void print(); // Printar matrisen

private: 
	void init(int x, int y, std::vector<double> z) { // Skapas separat från konstruktorerna för att göra konstruktorerna mer koncisa
		width = x;
		height = y;
		contents = z;
	}
};

LapackMat::LapackMat(int x, int y, std::vector<double> z) {
	init(x, y, z);
}

LapackMat::LapackMat(int x, int y) {
	init(x, y, std::vector<double>(x*y, 0.0)); // Kallar init med en nollvektor av rätt dimension som contents
}

LapackMat::LapackMat(int x) {
	init(x, x, std::vector<double>(x*x, 0.0)); // Skapar en kvadratisk nollmatris som i nollmatriskonstruktorn
	
	for (int i = 0; i < x; i++) {
		this->setElement(i, i, 1.0); // Sätter alla diagonalelement till 1
	}
}

double LapackMat::getElement(int row, int col) {
	return contents[row + col*height]; // Lapack har column-major order. row*height ger början av varje kolonn
}

void LapackMat::setElement(int row, int col, double value) {
	contents[row + col*height] = value;
}

void LapackMat::print() {
	for (int i = 0; i < height; i++) { // Loopar genom raderna
		for (int j = 0; j < width; j++) { // Loopar genom kolonnerna
			std::cout << numberToFixedWidth(getElement(i, j), 6); // Printar elementet
    	}
    	std::cout << '\n';
  	}
}

extern "C" {
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
}

/* I funktionerna nedan används dgemm_ flitigt. dgemm_ genomför i grunden operationen C = alpha*op(A)*op(B) + beta*C
 * där op(X) antingen är en transponering eller ingenting. Nedan följer en förklaring av ickeuppenbara argument.
 * TRANSX: Anger hur (och om) X ska transponeras. TRANSX='N' innebär ingen transponering
 *      M: Antalet rader i A (och C)
 *      N: Antalet kolonner i B (och C)
 *      K: Antalet kolonner i A och rader i B
 *    LDX: Ledande dimension för X. Kan anpassas om man vill använda submatriser, men det vill vi aldrig. */

LapackMat matrixMultiplication(LapackMat A, LapackMat B) { // Returnerar C=A*B
	LapackMat C(A.height, B.width); // Initierar ett C att skriva över. Kanske inte behövs egentligen?
	char TRANS = 'N';
	double ALPHA = 1;
	double BETA  = 0;

	dgemm_(&TRANS, &TRANS, &A.height, &B.width, &A.width, &ALPHA, A.contents.data(), &A.height, B.contents.data(), &A.width, &BETA, C.contents.data(), &A.height);

	return C;
}

LapackMat scalarMultiplication(double scalar, LapackMat A) { // Returnerar C=alpha*A
	LapackMat B = LapackMat(A.height); // Skapar en identitetsmatris för att bevara A vid multiplikation
	LapackMat C(A.height, B.width);
	char TRANS = 'N';
	double BETA  = 0;

	dgemm_(&TRANS, &TRANS, &A.height, &B.width, &A.width, &scalar, A.contents.data(), &A.height, B.contents.data(), &A.width, &BETA, C.contents.data(), &A.height);

	return C;
}

LapackMat matrixAddition(LapackMat A, LapackMat C) { // Returnerar C=A+C
	LapackMat B = LapackMat(A.height);
	char TRANS = 'N';
	double ALPHA = 1;
	double BETA  = 1;

	dgemm_(&TRANS, &TRANS, &A.height, &B.width, &A.width, &ALPHA, A.contents.data(), &A.height, B.contents.data(), &A.width, &BETA, C.contents.data(), &A.height);

	return C;
}

LapackMat matrixSubtraction(LapackMat A, LapackMat C) { // Returnerar C=A-C
	LapackMat B = LapackMat(A.height);
	char TRANS = 'N';
	double ALPHA = 1;
	double BETA  = -1;

	dgemm_(&TRANS, &TRANS, &A.height, &B.width, &A.width, &ALPHA, A.contents.data(), &A.height, B.contents.data(), &A.width, &BETA, C.contents.data(), &A.height);

	return C;
}

extern "C" { // Måste skrivas såhär
	void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
	void dgetrs_(char* C, int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
}

LapackMat solveMatrixEq(LapackMat A, LapackMat B) {
	LapackMat dummyA = LapackMat(A.width, A.height, A.contents); // dgetrf_ och dgetrs_ manipulerar A och B. Dummies skapas för att bevara ursprungliga A och B
	LapackMat dummyB = LapackMat(B.width, B.height, B.contents);

	int INFO;
	char TRANS = 'N';
	std::vector<int> IPIV(std::min(A.width, A.height));

	dgetrf_(&A.height, &A.width, dummyA.contents.data(), &A.height, IPIV.data(), &INFO);
	dgetrs_(&TRANS, &A.height, &B.width, dummyA.contents.data(), &A.width, IPIV.data(), dummyB.contents.data(), &A.height, &INFO);

	return dummyB;
}