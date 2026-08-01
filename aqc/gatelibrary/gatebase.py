from typing import List, Optional, Union
from AriaQuanta._utils import np


# -------------------------------------------------------------------------------------------
class GateBase:
    _num_target_qubits: Optional[int] = None    # set by subclasses

    def __init__(self, name: str, matrix: Optional[np.ndarray], target_qubits: Union[int, List[int]]) -> None:

        expected_dim = ( 2**self._num_target_qubits if self._num_target_qubits is not None else None )

        if matrix is not None:
            matrix = np.asarray(matrix, dtype=complex)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError( "Gate '{}' matrix must be square, got shape {}.".format(name, matrix.shape) )
            if expected_dim is not None and matrix.shape[0] != expected_dim:
                raise ValueError( "Gate '{0}' matrix must be {1}x{1} for a {2}-qubit gate, got shape {3}.".format(name, expected_dim, int(np.log2(expected_dim)), matrix.shape) )
            if not np.all(np.isfinite(matrix)):
                raise ValueError("Gate '{}' matrix contains NaN or Inf values.".format(name))


        self.name = name
        self.matrix = matrix
        self.target_qubits = target_qubits      # via the setter below


    # ------------------------------------------------------------
    @property
    def qubits(self) -> List[int]:
        return self._qubits

    @property
    def target_qubits(self) -> np.ndarray:
        return self._target_qubits
    
    @property
    def phase(self) -> np.ndarray:
        pass

    # ------------------------------------------------------------
    @target_qubits.setter
    def target_qubits(self, val: Union[int, List[int]]) -> None:
        target_qubits = np.atleast_1d(np.asarray(val, dtype=int)).flatten()

        if self._num_target_qubits is not None and target_qubits.size != self._num_target_qubits:
            raise ValueError( "{} expects exactly {} target qubit(s), got {}.".format(type(self).__name__, self._num_target_qubits, target_qubits.size) )

        self._target_qubits = target_qubits
        self._qubits = target_qubits.tolist()


    # ------------------------------------------------------------
    def _full_matrix(self, num_of_qubits: int) -> np.ndarray:
        raise NotImplementedError

    def apply(self, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
        pass
    
    def apply_density(self, num_of_qubits: int, density_matrix: np.ndarray) -> np.ndarray:
        pass


