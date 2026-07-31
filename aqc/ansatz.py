from copy import deepcopy
from typing import List, Optional, Tuple, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.qubit import Qubit
from AriaQuanta.aqc.gatelibrary import RY, RZ, CX, CRX, RXX, RYY, GateSingleQubit
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase


# parametrized circuit ----------------------------------------------------------------------
class Ansatz(Circuit):
    def __init__(self, num_of_qubits: int, params_names: List[str], num_of_clbits: int=0,
                 num_of_ancilla: int=0, list_of_qubits: Optional[List[Qubit]]=None) -> None:

        if len(params_names) == 0:
            raise ValueError("'params_names' must contain at least one parameter name.")
        if len(set(params_names)) != len(params_names):
            raise ValueError("'params_names' must not contain duplicate names, got {}.".format(params_names))

        #  ansatz = Ansatz(2, ['theta1'])
        #  ansatz | H(1) | RX('theta1',0) | H(0) | CX(0,1)
        self.params_names: List[str] = list(params_names)                        # ['theta1']
        self.params_values: np.ndarray = np.zeros(len(self.params_names))        # bound numeric values (0.0 until set)
        self.params_gates: List[Tuple[GateBase, str, int]] = []                  # [(gate, attribute_name, params_names index), ...]

        super().__init__(num_of_qubits, num_of_clbits, num_of_ancilla, list_of_qubits)


    # ------------------------------------------------------------
    def set_params_values(self, params_values: Union[List[float], np.ndarray]) -> None:
        params_values = np.asarray(params_values, dtype=float).flatten()
        if params_values.size != len(self.params_names):
            raise ValueError( "Expected {} parameter value(s) for {}, got {}.".format(len(self.params_names), self.params_names, params_values.size) )

        self.params_values = params_values
        for gate_i, key_i, index_i in self.params_gates:
            value_i = params_values[index_i]
            setattr(gate_i, key_i, value_i)
            gate_i.update_matrix()

    # ------------------------------------------------------------
    def _register_params(self, gate: GateBase) -> None:
        gate_dict = gate.__dict__
        params_names = self.params_names

        for this_key, this_value in gate_dict.items():
            if isinstance(this_value, str) and this_value in params_names:
                index = params_names.index(this_value)
                self.params_gates.append((gate, this_key, index))

    # ------------------------------------------------------------
    def add_gate(self, gate: GateBase) -> None:

        if max(gate.qubits) >= self.num_of_qubits:
            raise ValueError("{} is out-of-range for the qubit ID. The valid ID is between 0 and {}".format(max(gate.qubits), self.num_of_qubits - 1))

        # a single-qubit gate applied to several target qubits at once is split into one
        # independent gate (and its own parameter binding) per qubit -- mirrors Circuit.add_gate
        if isinstance(gate, GateSingleQubit):
            target_qubits = gate.target_qubits
            for tq in target_qubits:
                gate_copy = deepcopy(gate)
                gate_copy.target_qubits = [tq]
                self._register_params(gate_copy)
                self.gates.append(gate_copy)
        else:
            self._register_params(gate)
            self.gates.append(gate)

    # ------------------------------------------------------------
    def run(self) -> np.ndarray:
        self.statevector = self.initial_state
        return super().run()


# -------------------------------------------------------------------------------------------
class EfficientSU2Ansatz(Ansatz):
    def __init__(self, num_of_qubits: int, reps: int=3) -> None:
        if num_of_qubits < 1:
            raise ValueError("'num_of_qubits' must be at least 1, got {}.".format(num_of_qubits))
        if reps < 0:
            raise ValueError("'reps' must be non-negative, got {}.".format(reps))

        num_of_layers = reps + 1
        params_names = ['theta' + str(i) for i in range(num_of_layers * 2 * num_of_qubits)]
        super().__init__(num_of_qubits, params_names)

        for layer in range(num_of_layers):
            base = layer * 2 * num_of_qubits

            for q in range(num_of_qubits):
                self.add_gate(RY(params_names[base + q], q))
            for q in range(num_of_qubits):
                self.add_gate(RZ(params_names[base + num_of_qubits + q], q))

            if layer < reps:                                 # no entangler after the final rotation layer
                for q in range(num_of_qubits - 1):
                    self.add_gate(CX(q, q + 1))              

# -------------------------------------------------------------------------------------------
class H2Ansatz(Ansatz):
    def __init__(self) -> None:
        super().__init__(4, ['theta0', 'theta1', 'theta2'])

        self.add_gate(CRX('theta0', 0, 1))
        self.add_gate(CRX('theta1', 2, 3))

        self.add_gate(RXX('theta2', 0, 2))
        self.add_gate(RYY('theta2', 1, 3))

        self.add_gate(CX(0, 1))
        self.add_gate(CX(2, 3))
        self.add_gate(CX(1, 2))
