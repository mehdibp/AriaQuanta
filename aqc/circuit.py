from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from AriaQuanta._utils import np, reorder_state
from AriaQuanta.aqc.qubit import Qubit, MultiQubit
from AriaQuanta.aqc.gatelibrary import Custom
from AriaQuanta.aqc.measure import Measure
from AriaQuanta.aqc.operations import If_cbit
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase
from AriaQuanta.aqc.gatelibrary import GateSingleQubit


# -------------------------------------------------------------------------------------------
class Circuit:
    def __init__(self, 
                 num_of_qubits: int, 
                 num_of_clbits: int=0, 
                 num_of_ancilla: int=0, 
                 list_of_qubits: Optional[List[Qubit]] = None) -> None:
        
        self.num_of_qubits  = num_of_qubits
        self.num_of_clbits  = num_of_clbits
        self.num_of_ancilla = num_of_ancilla

        # result of the measurement
        self.measurequbit_values: Dict[str, str] = {}       # {'q0': '1', 'q1': '0'}
        self.gates: List[GateBase] = []
        self.width: int = num_of_qubits + num_of_clbits     # number of wires
            

        if list_of_qubits is None: multiqubit = MultiQubit(num_of_qubits)
        else: multiqubit = MultiQubit(num_of_qubits, list_of_qubits)
        initial_state = multiqubit.multistate
        
        self.initial_state = initial_state
        self.statevector   = initial_state


    # ------------------------------------------------------------
    def __or__(self, gate: GateBase) -> "Circuit":
        self.add_gate(gate)
        return self

    # ------------------------------------------------------------
    def add_gate(self, gate: GateBase) -> None:

        if max(gate.qubits) >= self.num_of_qubits:
            raise ValueError("{} is out-of-range for the qubit ID. The valid ID is between 0 to {}"
                             .format(max(gate.qubits),self.num_of_qubits-1)) 

        if isinstance(gate, GateSingleQubit):
            target_qubits = gate.target_qubits
            size_target_qubits = np.size(target_qubits)
            for i in range(size_target_qubits):
                gate_copy = deepcopy(gate)
                gate_copy.target_qubits = [target_qubits[i]]
                self.gates.append(gate_copy)
        else:
            self.gates.append(gate)   

    # ------------------------------------------------------------
    def run(self) -> np.ndarray:
        state = self.statevector

        measurequbit_values: Dict[str, str] = {}
        for gate in self.gates:
            if isinstance(gate, Measure):
                state = gate.apply(self.num_of_qubits, self.statevector)
                clbit_values_dict = gate.clbit_values_dict
                qubit_values_dict = gate.qubit_values_dict
                
                measurequbit_values.update(qubit_values_dict)    # modifies z with keys and values of y

            elif isinstance(gate, If_cbit):
                conditions = gate.conditions
                if clbit_values_dict[conditions[0]] == str(conditions[1]):
                    state = gate.apply(self.num_of_qubits, self.statevector)
            else:
                state = gate.apply(self.num_of_qubits, self.statevector)
            self.statevector = state

        
        # if the dictionary is empty. put all the qubits as the keys:
        if not measurequbit_values:
            for i in range(self.num_of_qubits):
                measurequbit_values['q'+str(i)] = ''
        
        measurequbit_values = dict(sorted(measurequbit_values.items())) # sort the dictionary
        self.measurequbit_values = measurequbit_values                  # save as the circuit's property

        return state
        
    # ------------------------------------------------------------
    def measure_all(self):
        # only measurement (not changing the statevector)
        num_of_qubits  = self.num_of_qubits
        num_of_ancilla = self.num_of_ancilla
        num_of_remaining_qubits = num_of_qubits - num_of_ancilla
        num_of_remaining_states = 2**num_of_remaining_qubits

        bin_format = '#0' + str(num_of_remaining_qubits + 2) + 'b'
        all_states = [format(x, bin_format)[2:] for x in range(num_of_remaining_states)]
        #all_states = [x[::-1] for x in all_states]


        state_reorder = reorder_state(self.statevector)

        for _ in range(num_of_ancilla):
            size_state_reorder = int(np.shape(state_reorder)[0]/2)
            state_reorder = state_reorder[:size_state_reorder]

        state_remaining = reorder_state(state_reorder)

        probabilities = np.abs(state_remaining) ** 2
        probabilities = probabilities / np.sum(probabilities)  # Normalize probabilities to sum to 1
        probabilities = probabilities.flatten()


        measurement_index = np.random.choice(len(state_remaining), p=probabilities)
        measurement_state = all_states[measurement_index]
        measurement = '|' + measurement_state + '>'      

        return measurement

    # ------------------------------------------------------------
    def copy(self) -> "Circuit": 
        qc_copy = deepcopy(self)
        return qc_copy

    # ------------------------------------------------------------
    def get_depth(self) -> int:
        depth = 0
        if len(self.gates) > 0:
            depth += 1
            gates = self.gates
            qubits_i = gates[0].qubits
            qubits_previous = []

            qubits_previous = qubits_i

            for i in range(1, len(gates)):
                qubits_i = gates[i].qubits
                flag = np.in1d(qubits_i,qubits_previous).any()

                if flag:
                    depth += 1
                    qubits_previous = qubits_i
                else:
                    qubits_previous = np.concatenate((qubits_previous, qubits_i))                        

        return depth     

    # ------------------------------------------------------------
    def reset(self) -> "Circuit":
        self.statevector = self.initial_state
        self.gates = []
        self.measurequbit_values = {}
        return self

    # ------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Circuit(num_of_qubits={self.num_of_qubits}, "
            f"num_of_clbits={self.num_of_clbits}, "
            f"num_of_ancilla={self.num_of_ancilla}, "
            f"gates={self.size}, depth={self.depth})"
        )


    # ------------------------------------------------------------
    @property   # show statevector as in Qiskit ------------------
    def statevector_reorder(self) -> np.ndarray:
        return reorder_state(self.statevector)

    @property
    def size(self) -> int:
        return len(self.gates)

    @property   # Type and number of gates used in the circuit ---
    def gatesinfo(self) -> Dict[str, int]:
        gatesinfo_dict: Dict[str, int] = {}

        for gate in self.gates:
            if gate.name in gatesinfo_dict: gatesinfo_dict[gate.name] += 1
            else:                           gatesinfo_dict[gate.name]  = 1  
        return gatesinfo_dict

    @property
    def density_matrix(self) -> np.ndarray:
        this_state = self.statevector
        density_matrix = this_state @ this_state.conj().T
        return density_matrix

    @property
    def density_matrix_reorder(self) -> np.ndarray:
        this_state = self.statevector_reorder
        density_matrix_reorder = this_state @ this_state.conj().T
        return density_matrix_reorder
    
    @property
    def depth(self) -> int:
        return self.get_depth()


# -------------------------------------------------------------------------------------------
def sv_to_probabilty(statevector: np.ndarray, plot: bool=True) -> Dict[str, float]:
    plt.rc('font', family='sans-serif')
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5

    num_of_qubits = int(np.log2(statevector.shape[0]))
    num_of_states = 2**num_of_qubits
    bin_format = '#0' + str(num_of_qubits + 2) + 'b'

    probabilities = np.abs(statevector)**2
    probabilities /= np.sum(probabilities)  # Normalize probabilities to sum to 1
    probabilities = probabilities.flatten()

    bin_format = '#0' + str(num_of_qubits + 2) + 'b'
    all_states = [format(x, bin_format)[2:] for x in range(num_of_states)]
    
    xtickes = all_states  

    probabilities_dict = {}
    for i in range(num_of_states):
        probabilities_dict[all_states[i]] = probabilities[i]   

    if plot==True:
        fig, ax = plt.subplots()
        xx = np.arange(np.shape(probabilities)[0])
        ax.bar(xx, probabilities)
        plt.xticks(xx, xtickes, rotation=45)
        ax.set_ylabel('Probability')

    return probabilities_dict

# quantum circuit ---------------------------------------------------------------------------  
def to_gate(qc: Circuit):

    num_of_qubits = qc.num_of_qubits

    this_qc = Circuit(num_of_qubits)
    this_qc.gates = qc.gates

    state_0 = this_qc.statevector
    state_1 = this_qc.run()
 
    state_0_norm = state_0 / np.linalg.norm(state_0)    # normalize initial state
    state_1_norm = state_1 / np.linalg.norm(state_1)    # normalized last state

    v = state_1_norm - state_0_norm
    v = np.reshape(v, (v.size, 1))
    v_dagger = np.reshape(v, (1, v.size))

    V_Vdagger = v @ v_dagger
    Vdagger_V = v_dagger @ v
    I = np.eye(2 ** num_of_qubits)

    A = I - 2 * V_Vdagger / Vdagger_V
    
    circuit_gate = Custom(matrix=A, target_qubits=list(range(0, num_of_qubits)))
    circuit_gate.matrix = A
    circuit_gate.name = 'Circuit_gate'

    return circuit_gate
