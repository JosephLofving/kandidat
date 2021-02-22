#include <iostream>
#include <iomanip>
#include <vector>
#include <stdlib.h>

  /* dgetrf() "inputs":
     M:    radantal i A
     N:    kolonnantal i A
     A:    Matrisen i AX=B
     LDA:  Ledande dimension. Används bara om man kollar på submatriser tror jag
     IPIV: Permutationsmatris (irrelevant vid input)
     INFO: Exit success (irrelevant vid input)

     dgetrf "outputs":
     M: Oförändrad
     N: Oförändrad
     A: L och U. Nedre vänstra hörnet är L (plus identitetsdiagonal), övre högra hörnet är U
        Exempeloutput:
        [2 3 4
         5 6 7
         8 9 10]
        betyder
        L = [1 0 0  och U = [2 3 4
             5 1 0           0 6 7
             8 9 1]          0 0 10]
    LDA: Oförändrad
    IPIV: Permutationsmatris. Anges i kolonncykelnotation, så t ex [2 3 3] motsvarar
         [1 0 0       [0 1 0       [0 0 1
          0 1 0   ->   1 0 0   ->   1 0 0
          0 0 1]       0 0 1]       0 1 0]
          där den sista matrisen är permutationsmatrisen P i A = P*L*U
    INFO: Exit success. 0 om allt gick bra. (Kan bli annat om matrisen inte går att faktorisera)
  */

  /* dgetrs() tar in outputs från dgetrf() som inputs, tillsammans med variabeln NRHS som anger
     antalet kolonner i matrisen B */

extern "C" { // Måste skrivas såhär
  void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
  void dgetrs_(char* C, int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
}

void print(const char* head, int m, int n, double* a){
  std::cout << std::endl << head << std::endl << "---------" << std::endl;
  for(int i=0; i<m; ++i){
    for(int j=0; j<n; ++j){
      std::cout << a[i+j*m]<< ' ';
    }
    std::cout << std::endl;
  }
  std::cout << "---------" << std::endl;
}

void printMatrix(int n, std::vector<double> A, std::string name) {
  std::cout << "\n---" << name << "---\n";

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        std::cout << A[i + j*n] << ' ';
    }
    std::cout << '\n';
  }
}

void printIntVect(std::vector<int> A, std::string name) {
  std::cout << "\n---" << name << "---\n";
  for (std::vector<int>::const_iterator i = A.begin(); i != A.end(); ++i) {
        std::cout << '\n';
    std::cout << *i << ' ';
  }
}

// https://blog.bossylobster.com/2018/09/learn-you-a-lapack.html
int main() {
  std::cout << std::fixed << std::setprecision(3) << std::setfill('0'); // Fixerar längden på tal som printas så allt blir snyggt

  int dim;  // Matrisdimension. Bara ett värde eftersom en kvadratisk matris skapas
  std::cout << "Enter Dimensions of Matrix : \n";
  std::cin >> dim;

  // "Matrisen" a är kvadratisk med storlek dim*dim, men sparas som en vektor med längd dim².
  std::vector<double> a(dim*dim); // a*x = b ska lösas
  /* Matrisens element får vektorindex som
     1 4 7
     2 5 8
     3 6 9
    för 3*3-matriser t ex, dvs det går kolonnvis */
  std::vector<double> b(dim);
  srand(1);              // seed the random # generator with a known value
  double maxr = (double)RAND_MAX;

  /* Tar ett element i b i taget, och en rad i a i taget med variabel r
  Variabeln i avser kolonnindex */
  for (int r = 0; r < dim; r++) {
    for (int i = 0; i < dim; i++) {
      a[r + i*dim] = rand() / maxr;
    }
    b[r] = rand() / maxr;
  }

  // Skapar några variabler som kommer ändras av funktionerna senare. Värdena är bara dummies
  int info;                   // Returnerar 0 om funktionen körs utan errors
  int one = 1;                // Anger kolonnantalet i b
  char N = 'N';               // Anger att systemet inte ska transponeras för dgetrs()
  std::vector<int> ipiv(dim); // Vektor med längd dim. Alla element är 0 till att börja med

  // ipiv.data() "Returns a direct pointer to the memory array used internally by the vector to store its owned elements."  
  // http://www.cplusplus.com/reference/vector/vector/data/

  printMatrix(dim, a, "A");
  print("B",dim,1,b.data());

  // std::cout << "\nINITIAL: \n";
  // printMatrix(dim, a, "a");
  // printMatrix(1, b, "b");
  // printIntVect(ipiv, "ipiv");
  
  dgetrf_(&dim, &dim, a.data(), &dim, ipiv.data(), &info);

  // std::cout << "\nLU-FACTORIZATION: \n";
  // printMatrix(dim, a, "a");
  // printMatrix(1, b, "b");
  // printIntVect(ipiv, "ipiv");

  dgetrs_(&N, &dim, &one, a.data(), &dim, ipiv.data(), b.data(), &dim, &info);

  // std::cout << "\nSOLVED: \n";
  // printMatrix(dim, a, "a");
  // printMatrix(1, b, "b");
  // printIntVect(ipiv, "ipiv");

  print("X",dim,1,b.data());

  return 0;
}