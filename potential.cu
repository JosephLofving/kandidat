#include "potential.h"

/* TODO: Explain what happens here */
int getArrayIndex(QuantumState state) {
    /* Array to be filled with potential elements. The ordering of the elements are:
     * V_array[0]="V_S0": S=0, L=LL=J
     * V_array[1]="V_S1": S=1, L=LL=J
     * V_array[2]="V_pp": S=1, L=J+1, LL=J+1
     * V_array[3]="V_mm": S=1, L=J-1, LL=J-1
     * V_array[4]="V_pm": S=1, L=J+1, LL=J-1
     * V_array[5]="V_mp": S=1, L=J-1, LL=J+1
     * So the 1S0-element will be given by V_array[0] */
    if(state.state["s"]==0){
         return 0;
     }

    if (state.state["l"]==state.state["ll"]) {
        if (state.state["l"]==state.state["J"]) {
            return 1;
        }
        if (state.state["l"]==state.state["J"] + 1) {
            return 2;
        }
        if (state.state["l"]==state.state["J"] - 1) {
            return 3;
        }
     }

     if (state.state["l"]>state.state["ll"]) {
         return 4;
     }

     else {
        return 5;
     }
}


void potential(cuDoubleComplex* VMatrix, std::vector<QuantumState> channel, double* k, double* Tlab, double* k0, int quadratureN, int TLabLength, bool coupled, int matLength) {

    /* Declare a NULL pointer of the potential-class type */
    chiral_LO* potentialClassPtr = nullptr;
    
    /* Set the pointer to a new instance of the class */
    potentialClassPtr = new chiral_LO();

    double* VArray = new double [6];
    
    int arrayIndex;
    cuDoubleComplex* VMat2 = new cuDoubleComplex[matLength * matLength];
    for (QuantumState state : channel){
        int L = state.state["l"];
        int S = state.state["s"];
        int J = state.state["J"];
        int T = state.state["t"];
        int Tz = state.state["Tz"];

        arrayIndex=getArrayIndex(state);

        //Creates Vmatrix without on-shell
        for (int kIn = 0; kIn < matLength-1; kIn++) {
            for (int kOut = 0; kOut < matLength-1; kOut++) {
                potentialClassPtr->V(k[kIn], k[kOut], coupled, S, J, T, Tz, VArray);
                setElement(VMat2, kIn, kOut, 0, matLength, make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0));
            }
        }

        // Copies Vmatrix for each energy
        for (int row = 0; row < matLength-1; ++row) {
            for (int col = 0; col < matLength-1; ++col) {
                for (int energyIndex = 0; energyIndex < TLabLength; ++energyIndex) {
                    setElement(VMatrix, row, col, energyIndex, matLength, getElement(VMat2, row, col, 0, matLength));
                }
            }
        }

        //Sets on-shell points for each energy
        for (int energyIndex=0; energyIndex < TLabLength; ++energyIndex){

            //Sets on-shell points for last row
            for (int col = 0; col < matLength-1; ++col) {
                potentialClassPtr->V(k0[energyIndex], k[col], coupled, S, J, T, Tz, VArray);
                setElement(VMatrix, matLength - 1, col, energyIndex, matLength, make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0));
            }

            //Sets on-shell points for last column
            for (int row = 0; row < matLength-1; ++row) {
                potentialClassPtr->V(k[row], k0[energyIndex], coupled, S, J, T, Tz, VArray);
                setElement(VMatrix, row, matLength - 1, energyIndex, matLength, make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0));
            }

            //Sets on-shell points for corner elements
            potentialClassPtr->V(k0[energyIndex], k0[energyIndex], coupled, S, J, T, Tz, VArray);
            setElement(VMatrix, matLength - 1, matLength - 1, energyIndex, matLength, make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0));
        }
    }
    delete potentialClassPtr;
    delete[] VArray;
    delete[] VMat2;


 }
    

   