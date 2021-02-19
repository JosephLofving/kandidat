bool kDelta(QuantumState bra, QuantumState ket, std::vector <std::string> qN){
	for(std::string s: qN){
		if (bra.state[s] != ket.state[s]){
			return false;
		}
	}
	return true;
}