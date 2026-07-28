from typing import List, Optional, Union

from AriaQuanta._utils import np, is_unitary
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase
from AriaQuanta.aqc.gatelibrary import I


# -------------------------------------------------------------------------------------------
class GateCustom(GateBase):
    _num_target_qubits: Optional[int] = None

    def __init__(self, name: str, matrix: Optional[np.ndarray], target_qubits: Union[int, List[int]]) -> None:

        target_qubits_arr = np.atleast_1d(np.asarray(target_qubits, dtype=int)).flatten()
        if len(set(target_qubits_arr.tolist())) != target_qubits_arr.size:
            raise ValueError( "target_qubits must not contain duplicates, got {}.".format(target_qubits_arr.tolist()) )

        super().__init__(name=name, matrix=matrix, target_qubits=target_qubits)


    # ------------------------------------------------------------
    def _full_matrix(self, num_of_qubits: int) -> np.ndarray:
        if self.matrix is None:
            raise ValueError(
                "Gate '{}' has a symbolic parameter (e.g. an angle given as a string placeholder for circuit diagrams) "
                "and therefore no numeric matrix to apply. Bind it to a numeric value first.".format(self.name)
            )

        acted_qubits = self.target_qubits.tolist()
        other_qubits = [q for q in range(num_of_qubits) if q not in acted_qubits]
        order = acted_qubits + other_qubits

        remaining = len(other_qubits)
        if remaining > 0: I2 = np.identity(2 ** remaining, dtype=complex)
        else: I2 = np.array([[1]], dtype=complex)

        block = np.kron(self.matrix, I2)

        dim = 2 ** num_of_qubits
        block_tensor = block.reshape([2] * num_of_qubits + [2] * num_of_qubits)
        inv_order = list(np.argsort(order))
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
class Custom(GateCustom):
    def __init__(self, matrix: Optional[np.ndarray] = None, target_qubits: Union[int, List[int]] = 0, name: str = 'Custom') -> None:
        if matrix is None: 
            matrix = I().matrix

        matrix = np.asarray(matrix, dtype=complex)
        if not is_unitary(matrix):
            raise ValueError('Custom matrix is not unitary')

        super().__init__(name=name, matrix=matrix, target_qubits=target_qubits)


