from typing import List, Optional, Union

from AriaQuanta._utils import np, is_unitary
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase
from AriaQuanta.aqc.gatelibrary import X, Y, Z


# -------------------------------------------------------------------------------------------
class GateControlN(GateBase):
    _num_target_qubits: Optional[int] = None

    def __init__(self, name: str, matrix: Optional[np.ndarray], base_matrix: Optional[np.ndarray], 
                 control_qubits: Union[int, List[int]], target_qubits: Union[int, List[int]]) -> None:

        control_qubits = np.atleast_1d(np.asarray(control_qubits, dtype=int)).flatten()
        if control_qubits.size < 1:
            raise ValueError( "GateControlN needs at least 1 control qubit; use GateControl for exactly 1, or a plain gate for 0." )
        if len(set(control_qubits.tolist())) != control_qubits.size:
            raise ValueError("control_qubits must not contain duplicates, got {}.".format(control_qubits.tolist()))

        self.control_qubits = control_qubits
        self.base_matrix = np.asarray(base_matrix)
        super().__init__(name=name, matrix=matrix, target_qubits=target_qubits)


    # ------------------------------------------------------------
    @property
    def qubits(self) -> List[int]:
        return self.control_qubits.tolist() + self._qubits


    # ------------------------------------------------------------
    def _full_matrix(self, num_of_qubits: int) -> np.ndarray:
        if self.matrix is None:
            raise ValueError(
                "Gate '{}' has a symbolic parameter (e.g. an angle given as a string placeholder for circuit diagrams) "
                "and therefore no numeric matrix to apply. Bind it to a numeric value first.".format(self.name)
            )

        acted_qubits = self.control_qubits.tolist() + self.target_qubits.tolist()
        other_qubits = [q for q in range(num_of_qubits) if q not in acted_qubits]
        order = acted_qubits + other_qubits

        remaining = len(other_qubits)
        if remaining > 0: I2 = np.identity(2 ** remaining, dtype=complex)
        else: I2 = np.array([[1]], dtype=complex)

        # self.matrix's own basis convention is [control_qubits..., target_qubits...]
        block = np.kron(self.matrix, I2)

        dim = 2 ** num_of_qubits
        block_tensor = block.reshape([2] * num_of_qubits + [2] * num_of_qubits)
        inv_order = list(np.argsort(order))

        # one permutation for the "output" tensor axes, the same one again
        # (offset by num_of_qubits) for the "input" tensor axes
        axes = inv_order + [num_of_qubits + i for i in inv_order]
        full_matrix = np.transpose(block_tensor, axes=axes).reshape(dim, dim)
        return full_matrix

    # ------------------------------------------------------------
    def apply(self, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
        full_matrix = self._full_matrix(num_of_qubits)
        return np.dot(full_matrix, multistate)

    # ------------------------------------------------------------
    def apply_density(self, num_of_qubits: int, density_matrix: np.ndarray) -> np.ndarray:
        full_matrix = self._full_matrix(num_of_qubits)
        return full_matrix @ density_matrix @ np.conj(full_matrix.T)



# -------------------------------------------------------------------------------------------
def _multi_controlled_matrix(base_matrix: np.ndarray, num_control_qubits: int) -> np.ndarray:
    d = base_matrix.shape[0]
    dim = (2 ** num_control_qubits) * d
    matrix = np.eye(dim, dtype=complex)
    matrix[-d:, -d:] = base_matrix

    return matrix

# -------------------------------------------------------------------------------------------
class CNX(GateControlN):
    def __init__(self, control_qubits: Union[int, List[int]], target_qubits: int=0) -> None:
        base = X().matrix
        control_qubits_arr = np.atleast_1d(np.asarray(control_qubits, dtype=int)).flatten()
        matrix = _multi_controlled_matrix(base, control_qubits_arr.size)
        super().__init__(name='CNX', matrix=matrix, base_matrix=base, control_qubits=control_qubits, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class CNY(GateControlN):
    def __init__(self, control_qubits: Union[int, List[int]], target_qubits: int=0) -> None:
        base = Y().matrix
        control_qubits_arr = np.atleast_1d(np.asarray(control_qubits, dtype=int)).flatten()
        matrix = _multi_controlled_matrix(base, control_qubits_arr.size)
        super().__init__(name='CNY', matrix=matrix, base_matrix=base, control_qubits=control_qubits, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class CNZ(GateControlN):
    def __init__(self, num_of_qubits: int, target_qubits: int=0) -> None:
        if not (0 <= target_qubits < num_of_qubits):
            raise ValueError( "target_qubits ({}) must be within [0, {}) for a CNZ spanning {} qubits.".format(target_qubits, num_of_qubits, num_of_qubits) )

        control_qubits = [q for q in range(num_of_qubits) if q != target_qubits]

        base = Z().matrix
        matrix = _multi_controlled_matrix(base, len(control_qubits))

        super().__init__(name='CNZ', matrix=matrix, base_matrix=base, control_qubits=control_qubits, target_qubits=target_qubits)


# -------------------------------------------------------------------------------------------
class _AxisRotationGate(GateControlN):
    _gate_name: str = ''

    def __init__(self, phase: Union[float, str], control_qubits: Union[int, List[int]], target_qubits: int=0) -> None:
        self.control_qubits = np.atleast_1d(np.asarray(control_qubits, dtype=int)).flatten()
        self._phase = phase
        matrix = self.update_matrix()
        base_matrix = None if isinstance(phase, str) else self._base_matrix_for_param(phase)
        super().__init__(name=self._gate_name, matrix=matrix, base_matrix=base_matrix, control_qubits=self.control_qubits, target_qubits=target_qubits)


    @property
    def phase(self) -> Union[float, str]:
        return self._phase

    def _base_matrix_for_param(self, phase: float) -> np.ndarray:
        raise NotImplementedError

    def update_matrix(self) -> Optional[np.ndarray]:
        if isinstance(self._phase, str):
            matrix = None
        else:
            base = self._base_matrix_for_param(self._phase)
            matrix = _multi_controlled_matrix(base, self.control_qubits.size)

        self.matrix = matrix
        return matrix

# -------------------------------------------------------------------------------------------
class CNP(_AxisRotationGate):
    _gate_name = 'CNP'

    def __init__(self, phi: Union[float, str], control_qubits: Union[int, List[int]], target_qubits: int=0) -> None:
        super().__init__(phase=phi, control_qubits=control_qubits, target_qubits=target_qubits)

    def _base_matrix_for_param(self, phi: float) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * phi)]])


# -------------------------------------------------------------------------------------------
class CNU(GateControlN):
    def __init__(self, base_matrix: np.ndarray, control_qubits: Union[int, List[int]], target_qubits: Union[int, List[int]]):
        base_matrix = np.asarray(base_matrix, dtype=complex)

        if not is_unitary(base_matrix):
            raise ValueError('Custom matrix is not unitary')

        control_qubits_arr = np.atleast_1d(np.asarray(control_qubits, dtype=int)).flatten()
        target_qubits_arr = np.atleast_1d(np.asarray(target_qubits, dtype=int)).flatten()

        expected_base_dim = 2 ** target_qubits_arr.size
        if base_matrix.shape[0] != expected_base_dim:
            raise ValueError( "CNU's base_matrix must be {0}x{0} to match {1} target qubit(s), got shape {2}.".format(expected_base_dim, target_qubits_arr.size, base_matrix.shape) )

        matrix = _multi_controlled_matrix(base_matrix, control_qubits_arr.size)

        super().__init__(name='CNU', matrix=matrix, base_matrix=base_matrix, control_qubits=control_qubits, target_qubits=target_qubits)

