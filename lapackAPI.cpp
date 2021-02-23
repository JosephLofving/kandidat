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
	lapackMat(int x); // Constructor för identitetsmatris
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

lapackMat::lapackMat(int x, int y, std::vector<double> z) {
	init(x, y, z);
}

lapackMat::lapackMat(int x, int y) {
	init(x, y, std::vector<double>(x*y, 0.0)); // Kallar init med en nollvektor av rätt dimension som contents
}

lapackMat::lapackMat(int x) {
	init(x, x, std::vector<double>(x*x, 0.0)); // Skapar en kvadratisk nollmatris som i nollmatriskonstruktorn
	
	for (int i = 0; i < x; i++) {
		this->setElement(i, i, 1.0); // Sätter alla diagonalelement till 1
	}
}

double lapackMat::getElement(int row, int col) {
	return contents[row + col*height]; // Lapack har column-major order. row*height ger början av varje kolonn
}

void lapackMat::setElement(int row, int col, double value) {
	contents[row + col*height] = value;
}

void lapackMat::print() {
	for (int i = 0; i < height; i++) { // Loopar genom raderna
		for (int j = 0; j < width; j++) { // Loopar genom kolonnerna
			std::cout << numberToFixedWidth(getElement(i, j), 6); // Printar elementet
    	}
    	std::cout << '\n';
  	}
}

lapackMat matrixMultiplication(lapackMat A, lapackMat B) { // Returnerar C=A*B
	lapackMat C(A.height, B.width); // Initierar ett C att skriva över. Kanske inte behövs egentligen?
	char TRANS = 'N'; // Anger att A och B inte ska transponeras
	int M = A.height;  // Antalet rader i A (och C)
	int N = B.width;   // Antalet kolonner i B (och C)
	int K = A.width;   // Antalet kolonner i A (och rader i B)
	double ALPHA = 1;  // Skalär som A*B multipliceras med (se nedan)
	double BETA  = 0;  // Skalär som C multipliceras med innan C adderas (se nedan)
	int LDA = M;       // Ledande dimension för A. Kan ändras för att utföra operationer på submatriser
	int LDB = K;       // Ledande dimension för B
	int LDC = M;	   // Ledande dimension för C

	// Tekniskt sett ger dgemm C = alpha*A*B + beta*C
	dgemm_(&TRANS, &TRANS, &M, &N, &K, &ALPHA, A.contents.data(), &LDA, B.contents.data(), &LDB, &BETA, C.contents.data(), &LDC);

	return C;
}

lapackMat scalarMultiplication(lapackMat A, double scalar) { // Se matrixMultiplication() för närmare förklaringar
	lapackMat B = lapackMat(A.height); // Skapar en identitetsmatris för att bevara A vid multiplikation
	lapackMat C(A.height, B.width);
	char TRANS = 'N';
	int M = A.height;  // Antalet rader i A (och C)
	int N = B.width;   // Antalet kolonner i B (och C)
	int K = A.width;   // Antalet kolonner i A (och rader i B)
	double BETA  = 0;  // Skalär som C multipliceras med innan C adderas (se nedan)
	int LDA = M;       // Ledande dimension för A. Kan ändras för att utföra operationer på submatriser
	int LDB = K;       // Ledande dimension för B
	int LDC = M;	   // Ledande dimension för C

	// Tekniskt sett ger dgemm C = alpha*A*B + beta*C
	dgemm_(&TRANS, &TRANS, &M, &N, &K, &scalar, A.contents.data(), &LDA, B.contents.data(), &LDB, &BETA, C.contents.data(), &LDC);

	return C;
}