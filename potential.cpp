
#include "chiral_LO.h"
#include "lapackAPI.h"
#include "quantumStates.h"
#include "constants.h"


double get_k0(std::vector<QuantumState> channel, double Tlab){
    int tz_channel= channel[0].state["tz"];
    double k0_squared{};
	if (tz_channel == -1)	  // Proton-proton scattering
		k0_squared = 2*constants::proton_mass*Tlab;
	else if (tz_channel == 0) // Proton-neutron scattering
		k0_squared = pow(constants::neutron_mass,2)*Tlab*(Tlab+2*constants::proton_mass)/((pow(constants::proton_mass+constants::neutron_mass,2)+2*Tlab*constants::neutron_mass));
	else if (tz_channel == 1) // Neutron-neutron scattering
		k0_squared = 2*constants::neutron_mass*Tlab;
    return sqrt(k0_squared); // Does not handle case where tz is NOT -1, 0 or 1.
	std::cout << "Incorrect tz_channel";
	abort();
}

 LapackMat potential(std::vector<QuantumState> channel, std::vector<double> p, double Tlab) {
//std::vector<double> potential(int argc, char* argv[]){
    

    /* Declare a NULL pointer of the potential-class type */
    chiral_LO* potential_class_ptr = NULL;
    
    /* Set the pointer to a new instance of the class */
    potential_class_ptr = new chiral_LO();

    double* V_array = new double [6];
    bool coupled = false;

    double k_0 = get_k0(channel,Tlab);

    LapackMat V_matrix = LapackMat(p.size()+1, p.size()+1);

    p.push_back(k_0);

    for (QuantumState state : channel){

        int L = state.state["l"];
        int S = state.state["s"];
        int J = state.state["j"];
        int T = state.state["t"];
        int Tz = state.state["tz"];

        double temp =5;

        potential_class_ptr->V(temp, temp, coupled, S, J, T, Tz, V_array);

        std::cout <<"V(5)" << V_array[0];
    

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

    

    for (int p_in = 0; p_in < p.size(); p_in++) {
        for (int p_o = 0; p_o < p.size(); p_o++) {
            potential_class_ptr->V(p[p_in], p[p_o], coupled, S, J, T, Tz, V_array);
            V_matrix.setElement(p_in,p_o,V_array[0]);
        }
    }
  }
    return V_matrix;
}