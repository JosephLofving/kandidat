
#include "potential.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuComplex.h>

int getArrayIndex(QuantumState state){
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
    if(state.state["l"]==state.state["ll"]){
        if(state.state["l"]==state.state["j"]){
            return 1;
        }
        if(state.state["l"]==state.state["j"]+1){
            return 2;
        }
        if(state.state["l"]==state.state["j"]-1){
            return 3;
        }
     }

     if(state.state["l"]>state.state["ll"]){
         return 4;
     }
    else{
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
    int rowIndex=0;
    int colIndex=0;
    for (QuantumState state : channel){
        int L = state.state["l"];
        int S = state.state["s"];
        int J = state.state["j"];
        int T = state.state["t"];
        int Tz = state.state["tz"];

        arrayIndex=getArrayIndex(state);

        if(arrayIndex==2){
            rowIndex = 1;
            colIndex = 1;
        }

        if(arrayIndex==3){
            rowIndex = 0;
            colIndex = 0;
        }

        if(arrayIndex==4){
            rowIndex = 0;
            colIndex = 1;
        }

        if(arrayIndex==5){
            rowIndex = 1;
            colIndex = 0;
        }

 //       	printf("\n MAIN_2 V_h[0] = %.10e", cuCreal(V_h[0]));
	//printf("\n MAIN_2 V_h[1] = %.10e", cuCreal(V_h[1]));
	//printf("\n MAIN_2 V_h[2] = %.10e", cuCreal(V_h[2]));
	//printf("\n MAIN_2 V_h[3] = %.10e", cuCreal(V_h[3]));
	//printf("\n MAIN_2 V_h[4] = %.10e", cuCreal(V_h[4]));
	//printf("\n MAIN_2 V_h[5] = %.10e", cuCreal(V_h[5]));

        /*
        for (int kIn = 0; kIn < quadratureN + 1; kIn++) {
            for (int kOut = 0; kOut < quadratureN + 1; kOut++) {
                potentialClassPtr->V(kNew[kIn], kNew[kOut], coupled, S, J, T, Tz, VArray);
                VMatrix[(kIn+rowIndex*(quadratureN + 1))+(kOut+colIndex*(quadratureN + 1)) * (quadratureN + 1)] = make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0);
            }
        }*/
        cuDoubleComplex* VMat2 = new cuDoubleComplex[matLength * matLength];
        for (int kIn = 0; kIn < quadratureN; kIn++) {
            for (int kOut = 0; kOut < quadratureN; kOut++) {
                potentialClassPtr->V(k[kIn], k[kOut], coupled, S, J, T, Tz, VArray);
                VMat2[(kIn + rowIndex * (quadratureN + 1)) + (kOut + colIndex * (quadratureN + 1)) * (quadratureN + 1)] =
                        make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0);
            }
        }
        for (int i = 0; i < matLength * matLength; ++i) {
            for (int energyIndex = 0; energyIndex < TLabLength; ++energyIndex) {
                VMatrix[i + (energyIndex * matLength * matLength)] = VMat2[i];
            }
        }

        for (int energyIndex=0; energyIndex < TLabLength; ++energyIndex){
            for (int kOut = 0; kOut < matLength-1; ++kOut) {
                    potentialClassPtr->V(k0[energyIndex], k[kOut], coupled, S, J, T, Tz, VArray);
                    VMatrix[((quadratureN + 1) + rowIndex * (quadratureN + 1)) + (kOut * (quadratureN + 1) + colIndex * (quadratureN + 1)) * (quadratureN + 1)+energyIndex*matLength*matLength] =
                        make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0);

            }
            for (int kIn = 0; kIn < matLength-1; ++kIn) {
                potentialClassPtr->V(k[kIn], k0[energyIndex], coupled, S, J, T, Tz, VArray);
                VMatrix[(kIn+ rowIndex * (quadratureN + 1)) + ((quadratureN + 1) * (quadratureN + 1) + colIndex * (quadratureN + 1)) * (quadratureN + 1)+energyIndex*matLength*matLength] =
                    make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0);

            }
            potentialClassPtr->V(k0[energyIndex], k0[energyIndex], coupled, S, J, T, Tz, VArray);
            VMatrix[(quadratureN + 1)+ rowIndex * (quadratureN + 1)) + ((quadratureN + 1) * (quadratureN + 1) + colIndex * (quadratureN + 1)) * (quadratureN + 1)+energyIndex*matLength*matLength] =
                make_cuDoubleComplex(constants::pi / 2.0 * VArray[arrayIndex], 0);
        }
    }
 }

   