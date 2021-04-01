
#include "quantumStates.h"
#include "cublasAPI.h"
#include "chiral_LO.h"
#include "constants.h"


double getk0(std::vector<QuantumState> channel, double Tlab){
    int tzChannel = channel[0].state["tz"];
    double k0Squared = 0;
    if (tzChannel == -1)      // Proton-proton scattering
        k0Squared = 2*constants::protonMass*Tlab;
    else if (tzChannel == 0) // Proton-neutron scattering
        k0Squared = pow(constants::neutronMass,2)*Tlab*(Tlab+2*constants::protonMass)
        /((pow(constants::protonMass+constants::neutronMass,2)+2*Tlab*constants::neutronMass));
    else if (tzChannel == 1) // Neutron-neutron scattering
        k0Squared = 2*constants::neutronMass*Tlab;
    
    return sqrt(k0Squared); 
}

 Tensor potential(std::vector<QuantumState> channel, std::vector<double> k, std::vector<double> Tlab) {

    /* Declare a NULL pointer of the potential-class type */
    chiral_LO* potentialClassPtr = nullptr;
    
    /* Set the pointer to a new instance of the class */
    potentialClassPtr = new chiral_LO();

    double* VArray = new double [6];
    bool coupled = false;

    k.push_back(0); //Making space for k0
    double k0;

     Tensor vTensor = Tensor(k.size(),k.size(),Tlab.size());

    for (QuantumState state : channel){
        int L = state.state["l"];
        int S = state.state["s"];
        int J = state.state["j"];
        int T = state.state["t"];
        int Tz = state.state["tz"];

        for(int Tindex=0; Tindex<Tlab.size(); Tindex++){
            k0 = getk0(channel,Tlab[Tindex]);
            k[k.size()-1]=k0; //Replace last value with k0

            for (int kIn = 0; kIn < k.size(); kIn++) {
                for (int kOut = 0; kOut < k.size(); kOut++) {
                    potentialClassPtr->V(k[kIn], k[kOut], coupled, S, J, T, Tz, VArray);
                    vTensor.setElement(Tindex, kIn, kOut, constants::pi / 2.0 * VArray[0]);
                }
            }
    }
    }
    return vTensor;
}
