
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

 LapackMat potential(std::vector<QuantumState> channel, std::vector<double> k, double Tlab, double k0) {
//std::vector<double> potential(int argc, char* argv[]){
    

    /* Declare a NULL pointer of the potential-class type */
    chiral_LO* potentialClassPtr = nullptr;
    
    /* Set the pointer to a new instance of the class */
    potentialClassPtr = new chiral_LO();

    double* VArray = new double [6];
    bool coupled = false;

    LapackMat VMatrix = LapackMat(k.size()+1, k.size()+1);

    k.push_back(k0);

    for (QuantumState state : channel){
        int L = state.state["l"];
        int S = state.state["s"];
        int J = state.state["j"];
        int T = state.state["t"];
        int Tz = state.state["tz"];


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

    

    for (int kIn = 0; kIn < k.size(); kIn++) {
        for (int kOut = 0; kOut < k.size(); kOut++) {
            potentialClassPtr->V(k[kIn], k[kOut], coupled, S, J, T, Tz, VArray);
            VMatrix.setElement(kIn, kOut, constants::pi / 2.0 * VArray[0]);
        }
    }
  }
    return VMatrix;
}