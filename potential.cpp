
#include "potential.h"


double getk0(std::vector<QuantumState> channel, double Tlab){
    int tzChannel = channel[0].state["tz"];
    double k0Squared = 0;
	if (tzChannel == -1)	  // Proton-proton scattering
		k0Squared = constants::protonMass*Tlab/2;
	else if (tzChannel == 0) // Proton-neutron scattering
		k0Squared = pow(constants::neutronMass,2)*Tlab*(Tlab+2*constants::protonMass)/((pow(constants::protonMass+constants::neutronMass,2)+2*Tlab*constants::neutronMass));
	else if (tzChannel == 1) // Neutron-neutron scattering
		k0Squared = constants::neutronMass*Tlab/2;
    
    return sqrt(k0Squared); // Does not handle case where tz is NOT -1, 0 or 1.
}

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

 LapackMat potential(std::vector<QuantumState> channel, std::vector<double> k, double Tlab) {

    /* Declare a NULL pointer of the potential-class type */
    chiral_LO* potentialClassPtr = nullptr;
    /* Set the pointer to a new instance of the class */
    potentialClassPtr = new chiral_LO();

    double* VArray = new double [6];

    double k0 = getk0(channel,Tlab); //Calculates on shell point
    k.push_back(k0);


    bool coupled = false;
    LapackMat VMatrix = LapackMat(k.size(), k.size());

    if(channel.size()>1){
        coupled = true;
        VMatrix = LapackMat(k.size()*2, k.size()*2);
    }

    double factor = constants::pi / 2.0;
    int arrayIndex;
    int rowIndex=0;
    int colIndex=0;
    for (QuantumState state : channel){
        

        int L = state.state["l"];
        int S = state.state["s"];
        int J = state.state["j"];
        int T = state.state["t"];
        int Tz = state.state["tz"];

    
    

    //double* V_array = new double [6];

    /* Our potentials usually use an argument called "coupled", which is a bool-type.
     * This tells the potential-program which parts the input-array V_array to calculate.
     * If coupled=false, then it calculates V_S0 and V_S1, otherwise it calculates V_pp, V_mm, V_pm, and V_mp
     * (since these are coupled states with respect to L!=J and LL!=J) */

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



        for (int kIn = 0; kIn < k.size(); kIn++) {
            for (int kOut = 0; kOut < k.size(); kOut++) {
                potentialClassPtr->V(k[kIn], k[kOut], coupled, S, J, T, Tz, VArray);
                VMatrix.setElement(kIn+rowIndex*k.size(), kOut+colIndex*k.size(),factor*VArray[arrayIndex]);
            }
        }   
  }
    return VMatrix;
}