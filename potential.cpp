
#include "potential.h"

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


cuDoubleComplex* potential(std::vector<QuantumState> channel, double* k, double Tlab, double* k0, int NKvadratur) {
//std::vector<double> potential(int argc, char* argv[]){
    

    /* Declare a NULL pointer of the potential-class type */
    chiral_LO* potentialClassPtr = nullptr;
    
    /* Set the pointer to a new instance of the class */
    potentialClassPtr = new chiral_LO();

    double* VArray = new double [6];
    bool coupled;
    

    double* kNew = new double[NKvadratur + 1];
    for (int i = 0; i < NKvadratur; ++i) {
        kNew[i] = k[i];
    }
    kNew[NKvadratur] = k0;

    coupled = false;
    cuDoubleComplex* VMatrix = new cuDoubleComplex[(NKvadratur + 1) * (NKvadratur + 1)];

    if(channel.size()>1){
        coupled = true;
        VMatrix = new cuDoubleComplex[(NKvadratur + 1)*2 * (NKvadratur + 1)*2];
    }
    


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


        for (int kIn = 0; kIn < NKvadratur + 1; kIn++) {
            for (int kOut = 0; kOut < NKvadratur + 1; kOut++) {
                potentialClassPtr->V(kNew[kIn], kNew[kOut], coupled, S, J, T, Tz, VArray);
                VMatrix[(kIn+rowIndex*(NKvadratur + 1))+(kOut+colIndex*(NKvadratur + 1)) * (NKvadratur + 1)] = make_cuDoubleComplex(constants::pi / 2.0 * VArray[0], 0);
            }
        }
    }
    return VMatrix;
 }

    /* Define the 1S0 partial-wave quantum numbers 
    int L = 0; // This argument is actually redundant due to a boolaen "coupled" we use later (which you may see when you compile this program)
    int S = 0;
    int J = 0;
    int T = 1; // I use generalised Pauli principle (L+S+T=odd) to determine T=1 since L=S=0
    int Tz = 0;
    */

    /* I set this to be a proton-neutron scattering system, meaning Tz=0.
     * For pp-scattering we would have Tz=-1 and for nn we would have Tz=+1 */

    
    /* Array to be filled with potential elements. The ordering of the elements are:
     * V_array[0]="V_S0": S=0, L=LL=J
     * V_array[1]="V_S1": S=1, L=LL=J
     * V_array[2]="V_pp": S=1, L=J+1, LL=J+1
     * V_array[3]="V_mm": S=1, L=J-1, LL=J-1
     * V_array[4]="V_pm": S=1, L=J+1, LL=J-1
     * V_array[5]="V_mp": S=1, L=J-1, LL=J+1
     * So the 1S0-element will be given by V_array[0] */

    //double* V_array = new double [6];

    /* Our potentials usually use an argument called "coupled", which is a bool-type.
     * This tells the potential-program which parts the input-array V_array to calculate.
     * If coupled=true, then it calculates V_S0 and V_S1, otherwise it calculates V_pp, V_mm, V_pm, and V_mp
     * (since these are coupled states with respect to L!=J and LL!=J) */

    //bool coupled = false;
    
    /* Call class-member function V */
    /* Declare and set in- and out-momenta in c.m.*/