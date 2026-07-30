from typing import Dict, List, Optional, Union

from AriaQuanta._utils import np

# -------------------------------------------------------------------------------------------
class Measure:
    def __init__(self, name: str, qubits: List[int], clbits: Optional[List[Union[int, str]]], resize: bool, seed: Optional[int]=None):
        qubits = list(qubits)

        if len(qubits) == 0:
            raise ValueError("'qubits' must contain at least one qubit to measure.")
        if len(set(qubits)) != len(qubits):
            raise ValueError("'qubits' must not contain duplicates, got {}.".format(qubits))
        if clbits is not None and len(clbits) != len(qubits):
            raise ValueError( "'clbits' must have the same length as 'qubits' ({} given, {} expected).".format(len(clbits), len(qubits)) )
        
        self.name   = name
        self.qubits = qubits
        self.clbits = clbits
        self.resize = resize


        self._rng = np.random.default_rng(seed) if seed is not None else None
        
        self.clbit_values_dict: Dict[str, str] = {}
        self.qubit_values_dict: Dict[str, str] = {}


    # ------------------------------------------------------------
    def _sample_measurement_index(self, probabilities: np.ndarray) -> int:
        probabilities = np.asarray(probabilities, dtype=float).flatten()
        probabilities = probabilities / np.sum(probabilities)           # Normalize probabilities to sum to 1
    
        if self._rng is None:
            return int(np.random.choice(len(probabilities), p=probabilities))
        return int(self._rng.choice(len(probabilities), p=probabilities))

    # ------------------------------------------------------------
    def apply(self, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
        # example:
        # qc.measure_all([1, 2], [0, 1])
        # Measures qubit 1 into classical bit 0 and qubit 2 into classical bit 1 
        
        self.clbit_values_dict = {}
        self.qubit_values_dict = {}

        qubits = self.qubits
        if max(qubits) >= num_of_qubits:
            raise ValueError( "{} is out-of-range for the qubit ID. The valid ID is between 0 to {}".format(max(qubits), num_of_qubits - 1) )

        clbits = self.clbits
        if clbits is None:
            clbits = ['c' + str(q) for q in qubits]         

        # Creating a list of all states --------------------------
        state = multistate
        num_of_states = np.shape(state)[0]
        num_of_qubits = int(np.log2(num_of_states))

        probabilities = np.abs(state) ** 2
        probabilities = probabilities.flatten()

        # --------------------------------------------------------
        measurement_index = self._sample_measurement_index(probabilities)

        bin_format = '#0' + str(num_of_qubits + 2) + 'b'  # #05b
        measurement_state = format(measurement_index, bin_format)[2:]  # 010: q0, q1, q2

        # save measurement outputs -------------------------------
        for q, c in zip(qubits, clbits):
            measurement_i = measurement_state[q]
            self.qubit_values_dict['q' + str(q)] = measurement_i
            self.clbit_values_dict[str(c)] = measurement_i

        # find the remaining basis states ------------------------
        indices = np.arange(num_of_states)
        mask = np.ones(num_of_states, dtype=bool)
        for q in qubits:
            outcome_bit = int(self.qubit_values_dict['q' + str(q)])

            shift     = num_of_qubits-1-q
            basis_bit = (indices >> shift) & 1
            mask     &= (basis_bit == outcome_bit)
        last_indices = indices[mask]

        # Update statevector -------------------------------------
        probabilities_selected = probabilities[last_indices]
        scale_probabilities = 1 / np.sum(probabilities_selected)
        scale_probabilities_sqrt = np.sqrt(scale_probabilities)

        if self.resize:
            multistate  = multistate[last_indices]
            multistate *= scale_probabilities_sqrt
        else:
            remove_mask = np.ones(num_of_states, dtype=bool)
            remove_mask[last_indices] = False
            multistate [remove_mask] = 0
            multistate *= scale_probabilities_sqrt

        return multistate
        

# -------------------------------------------------------------------------------------------
class MeasureQubit(Measure):
    def __init__(self, qubits: List[int], clbits: Optional[List[Union[int, str]]] = None, seed: Optional[int] = None) -> None:
        super().__init__(name='MeasureQubit', qubits=qubits, clbits=clbits, resize=False, seed=seed)

# -------------------------------------------------------------------------------------------
class MeasureQubitResize(Measure):
    def __init__(self, qubits: List[int], clbits: Optional[List[Union[int, str]]] = None, seed: Optional[int] = None) -> None:
        super().__init__(name='MeasureQubitResize', qubits=qubits, clbits=clbits, resize=True, seed=seed)



