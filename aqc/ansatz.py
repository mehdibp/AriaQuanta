
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import RY, RZ, CX, CRX, RXX, RYY, H
import numpy as np

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# parametrized circuit

class Ansatz(Circuit):

    #def __init__(self, num_of_qubits, num_of_clbits=0, num_of_ancilla=0, list_of_qubits=[]):

    def __init__(self, num_of_qubits, params_names, num_of_clbits=0, num_of_ancilla=0, list_of_qubits=[]):

        #  #ansatz = Ansatz(2, ['theta1'])
        #  #ansatz | H(1) | RX('theta1',0) | H(0) | CX(0,1) 
        self.params_names = params_names      # ['theta1']
        self.params_values = np.empty((len(self.params_names),)) # [0.69314718] numypy genertates some random number
        self.params_gates = [] # [(<AriaQuanta.aqc.gatelibrary.gatesingle.RX object at 0x7baebecb5ba0>, '_theta', 0)]

        super().__init__(num_of_qubits, num_of_clbits, num_of_ancilla, list_of_qubits) 

    #----------------------------------------------
    def set_params_values(self, params_values):
        self.params_values = params_values
        params_gates = self.params_gates
        for item in params_gates:
            gate_i = item[0]
            key_i = item[1]
            value_i = params_values[item[2]]
            setattr(gate_i, key_i, value_i)
            gate_i.update_matrix()

    #----------------------------------------------
    def add_gate(self, gate):
        
        if max(gate.qubits) > self.num_of_qubits:
            raise ValueError("{} is out-of-range for the qubit ID. The valid ID is between 0 and {}".format(max(gate.qubits),self.num_of_qubits-1))
        
        # save the gates with params   
        gate_dict = gate.__dict__
        params_names = self.params_names
        params_gates = self.params_gates

        for this_key, this_value in gate_dict.items():
            if (isinstance(this_value, str)) and this_value in params_names:
                index = params_names.index(this_value)
                params_gates.append((gate, this_key, index))
      
        #if isinstance(gate, GateSingleQubit):
        #    for i in range(len(gate.qubits)):    
        #        gate_copy = deepcopy(gate)
        #        gate_copy.qubits = [gate.qubits[i]]   
        #        self.gates.append(gate_copy)                
        else:
            self.gates.append(gate)

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
def EfficientSU2Ansatz():
    params_names = []
    for i in range(16):
        params_names.append('theta'+str(i))
    myansatz = Ansatz(2, params_names)  

    for i in range(4):
        myansatz | RY('theta'+str(i*4),0) | RY('theta'+str(i*4+1),1) 
        myansatz | RZ('theta'+str(i*4+2),0) | RZ('theta'+str(i*4+3),1) 
        if i in [0,1,2]:  
            myansatz | CX(0,1)  
    return myansatz                  

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
def H2Ansatz():
    
    ansatz = Ansatz(4, ['theta0', 'theta1', 'theta2'])

    ansatz | CRX('theta0', 0, 1)
    ansatz | CRX('theta1', 2, 3)
    
    ansatz | RXX('theta2', 0, 2)
    ansatz | RYY('theta2', 1, 3)

    ansatz | CX(0, 1)
    ansatz | CX(2, 3)
    ansatz | CX(1, 2)

    return ansatz

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
# def H2DoubleAnsatz():
#    
#    ansatz = Ansatz(2, ['theta'])
#
#    ansatz | H(0) | H(1)
#    ansatz | CX(0, 1)
#    ansatz | RZ('theta', 1)
#    ansatz | CX(0, 1)
#
#    return ansatz

"""

def UCCSDAnsatz():
    def __init__(self, num_qubits):
        super().__init__(num_qubits)

    def build_uccsd_ansatz(self, theta_singles, theta_doubles):
        #
        #Constructs the UCCSD ansatz for H2 using a Trotterized Pauli exponentiation.
        #- theta_singles: Parameters for single excitations.
        #- theta_doubles: Parameters for double excitations.
        
        assert len(theta_singles) == 2, "UCCSD for H2 requires 2 parameters for single excitations"
        assert len(theta_doubles) == 1, "UCCSD for H2 requires 1 parameter for double excitations"
        
        # Apply single excitations (UCCS)
        self | ("CRX", theta_singles[0], 0, 1)  # Excitation between qubits 0 and 1
        self | ("CRX", theta_singles[1], 2, 3)  # Excitation between qubits 2 and 3
        
        # Apply double excitations (UCCD)
        self | ("RXX", theta_doubles[0], 0, 2)  # Simulating a two-electron excitation
        self | ("RYY", theta_doubles[0], 1, 3)  
        
        # Entangling layers to capture correlations
        self | ("CX", 0, 1)
        self | ("CX", 2, 3)
        self | ("CX", 1, 2)

# Example instantiation:
uccsd_ansatz = UCCSDAnsatz(4)  # 4 qubits for Jordan-Wigner mapping
initial_theta_singles = [0.1, 0.2]  # Initial guesses for single excitations
initial_theta_doubles = [0.3]       # Initial guess for double excitation
uccsd_ansatz.build_uccsd_ansatz(initial_theta_singles, initial_theta_doubles)

"""