
#include "chiral_LO.h"
#include "lapackAPI.h"
#include "quantumStates.h"

 LapackMat potential(std::vector<QuantumState> channel, std::vector<double> p) {
//std::vector<double> potential(int argc, char* argv[]){
    

    /* Declare a NULL pointer of the potential-class type */
    chiral_LO* potential_class_ptr = NULL;
    
    /* Set the pointer to a new instance of the class */
    potential_class_ptr = new chiral_LO();

    double* V_array = new double [6];
     bool coupled = false;

    for (QuantumState state : channel){

        int L = state["l"];
        int S = state["s"];
        int J = state["j"];
        int T = state["t"];
        int Tz = state["tz"];
    }

    /* Define the 1S0 partial-wave quantum numbers */
    //int L = 0; // This argument is actually redundant due to a boolaen "coupled" we use later (which you may see when you compile this program)
    //int S = 0;
    //int J = 0;
    //int T = 1; // I use generalised Pauli principle (L+S+T=odd) to determine T=1 since L=S=0

    /* I set this to be a proton-neutron scattering system, meaning Tz=0.
     * For pp-scattering we would have Tz=-1 and for nn we would have Tz=+1 */
    int Tz = 0;
    
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
    bool coupled = false;
    
    /* Call class-member function V */
    /* Declare and set in- and out-momenta in c.m.*/

    
    LapackMat V_matrix = LapackMat(p.size(), p.size());
    for (int p_in = 1; p_in < p.size(); p_in++) {
        for (int p_o = 1; p_o < p.size(); p_o++) {
            potential_class_ptr->V(p[p_in], p[po], coupled, S, J, T, Tz, V_array);
            std::cout << V_array[0] << std::endl;
            V_matrix.setElement(p_in,p_o,V_array[0]);
        }
    }

    return V_matrix;
}