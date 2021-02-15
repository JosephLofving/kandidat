#include <iostream>
#include <map>
#include <string>
#include <vector> 
#include <iterator>
#include <math.h> 

using namespace std; 

//g++ -std=c++11 helloworld.cpp -o main

// För att köre C++ ver 11

class BaseState{
	public:
		std::map<std::string, int> baseState;

		void addQuantumNumber(string key,int value){
			baseState.insert(std::pair<std::string, int>(key,value));
		}

		void printBaseState(){
			std::map<std::string, int>::iterator itr;
			for (itr = baseState.begin(); itr != baseState.end(); ++itr) { 
        		std::cout << itr->first 
             	<< ":" << itr->second << ", "; 
			}	
		}
};

class Basis {
	public:
		vector <BaseState> basis; //vector is supposed to be faster for iteration, which will be done. List could be used instead.
		Basis(int j2min,int j2max,int tzmin, int tzmax){
			for (int tz = tzmin; tz <= tzmax; tz++){
				for (int j = j2min; j <= j2max; j++){
					for (int s = 0; s < 2; s++){
						for (int l = abs(j-s); l <= j+s; l++){
							for (int t = abs(tz); t < 2; t++)
							{
								if ((l+s+t)%2 != 0){
									BaseState basestate;
									basestate.addQuantumNumber("tz",tz);
									basestate.addQuantumNumber("l",l);
									basestate.addQuantumNumber("pi",pow(-1,l));
									basestate.addQuantumNumber("s",s);
									basestate.addQuantumNumber("j",j);
									basestate.addQuantumNumber("t",t);

									basis.push_back(basestate);
								}
							}
							

						}
						

					}
					
				}
				
			}
			

		}
		void printBasis(){
			for(BaseState bs: basis){
				std::cout <<"["	;
				bs.printBaseState();
				std::cout <<"]"<<endl;
			}
		}

};

int main()
{
	std::cout << "Createing basis"<<endl;

	Basis base(0,2,0,2);
	base.printBasis();
	
	std::cout << "Basis length: "<<base.basis.size()<<endl;



	return 0;


	
}