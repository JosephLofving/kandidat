#include <iostream>
#include <vector>
#include <string.h> // Behövs för numberToFixedWidth()

extern "C" {
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
}

std::string numberToFixedWidth(double num, int width) { // Gör num till en string av bredd width med efterföljande blanksteg
	std::string s = std::to_string(num); // Typkonversion
	if (s.length() > width) {
		s = s.substr(0, width); // Trunkerar strängen om den är för lång
	}
	s.append(width+1 - s.length(), ' '); // Fyller ut strängen och lägger till blanksteg

	return s;
}

class lapackMat {
	public:
	int width;  // Matrisbredd
	int height; // Matrishöjd
	std::vector<double> contents;

	lapackMat(int x, int y, std::vector<double>); // Constructor
	lapackMat(int x, int y); // Constructor för nollmatris
	double matrixElement(int row, int col); // Returnerar element med givna index. Notera att allt är 0-indexerat
	void printMatrix(); // Printar matrisen
};

lapackMat::lapackMat(int x, int y, std::vector<double> z) {
	width = x;
	height = y;
	contents = z;
}

lapackMat::lapackMat(int x, int y) {
	width = x;
	height = y;
	contents = std::vector<double>(x*y, 0.0);
}

double lapackMat::matrixElement(int row, int col) {
	return contents[row + col*height]; // Lapack har column-major order. row*height ger början av varje kolonn
}

void lapackMat::printMatrix() {
	for (int i = 0; i < height; i++) { // Loopar genom raderna
		for (int j = 0; j < width; j++) { // Loopar genom kolonnerna
			std::cout << numberToFixedWidth(matrixElement(i, j), 6); // Printar elementet
    	}
    	std::cout << '\n';
  	}
}

lapackMat multiplyMatrix(lapackMat A, lapackMat B) { // Returnerar C=A*B
	lapackMat C(A.height, B.width); // Initierar ett C att skriva över. Kanske inte behövs egentligen?
	char TRANSA = 'N'; // Anger att A inte ska transponeras
	char TRANSB = 'N'; // Anger att B inte ska transponeras
	int M = A.height;  // Antalet rader i A (och C)
	int N = B.width;   // Antalet kolonner i B (och C)
	int K = A.width;   // Antalet kolonner i A (och rader i B)
	double ALPHA = 1;  // Skalär som A*B multipliceras med (se nedan)
	double BETA  = 0;  // Skalär som C multipliceras med innan C adderas (se nedan)
	int LDA = M;       // Ledande dimension för A. Kan ändras för att utföra operationer på submatriser
	int LDB = K;       // Ledande dimension för B
	int LDC = M;	   // Ledande dimension för C

	// Tekniskt sett ger dgemm C = alpha*A*B + beta*C
	dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A.contents.data(), &LDA, B.contents.data(), &LDB, &BETA, C.contents.data(), &LDC);

	return C;
}