from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import RY, RZ, CX, CRX, RXX, RYY


# parametrized circuit ----------------------------------------------------------------------
class Ansatz(Circuit):

    def __init__(self, num_of_qubits, params_names, num_of_clbits=0, num_of_ancilla=0, list_of_qubits=None):

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

# -------------------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------------------
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
