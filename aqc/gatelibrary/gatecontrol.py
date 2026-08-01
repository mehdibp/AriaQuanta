from typing import List, Optional, Union

from AriaQuanta._utils import np, swap_qubits, swap_qubits_density, is_unitary
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase
from AriaQuanta.aqc.gatelibrary import X, Z, S, I, SWAP


# -------------------------------------------------------------------------------------------
class GateControl(GateBase):
    _num_target_qubits: Optional[int] = None

    def __init__(self, name: str, matrix: Optional[np.ndarray], base_matrix: Optional[np.ndarray],
                 control_qubits: Union[int, List[int]], target_qubits: Union[int, List[int]]):

        self.base_matrix = np.asarray(base_matrix)
        self.control_qubits = np.atleast_1d(np.asarray(control_qubits, dtype=int)).flatten()
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

        num_of_control_qubits = np.shape(self.control_qubits)[0]
        num_of_target_qubits  = np.shape(self.target_qubits )[0]

        remaining = num_of_qubits - num_of_control_qubits - num_of_target_qubits
        if remaining > 0: I2 = np.identity(2 ** remaining, dtype=complex)
        else: I2 = 1

        return np.kron(self.matrix, I2)

    # ------------------------------------------------------------
    def apply(self, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
        control_qubits = self.control_qubits
        target_qubits = self.target_qubits
        num_of_target_qubits = np.shape(target_qubits)[0]

        # shift all the qubits to one ID higher, and increase the size of system to n+1
        one_state  = np.array([[1], [0]], dtype=complex)
        multistate = np.kron(one_state, multistate)
        control_qubits = control_qubits + 1
        target_qubits  = target_qubits  + 1
        num_of_qubits += 1

        multistate_swapped = swap_qubits(0, control_qubits[0], num_of_qubits, multistate)
        for k1 in range(1, num_of_target_qubits + 1):
            multistate_swapped = swap_qubits(k1, target_qubits[k1 - 1], num_of_qubits, multistate_swapped)
        
        full_matrix = self._full_matrix(num_of_qubits)
        multistate_swapped = np.dot(full_matrix, multistate_swapped)

        for k1 in reversed(range(1, num_of_target_qubits + 1)):
            multistate_swapped = swap_qubits(target_qubits[k1 - 1], k1, num_of_qubits, multistate_swapped)
        multistate = swap_qubits(control_qubits[0], 0, num_of_qubits, multistate_swapped)

        # shift qubit IDs back down and drop the ancilla, n+1 -> n qubits
        multistate = multistate[: int(np.size(multistate) / 2)]

        return multistate

    # ------------------------------------------------------------
    def apply_density(self, num_of_qubits: int, density_matrix: np.ndarray) -> np.ndarray:
        control_qubits = self.control_qubits
        target_qubits  = self.target_qubits
        num_of_target_qubits = np.shape(target_qubits)[0]

        density_matrix_swapped = swap_qubits_density(0, control_qubits[0], num_of_qubits, density_matrix)
        for k1 in range(1, num_of_target_qubits + 1):
            density_matrix_swapped = swap_qubits_density(k1, target_qubits[k1 - 1], num_of_qubits, density_matrix_swapped)

        full_matrix = self._full_matrix(num_of_qubits)
        density_matrix_swapped = full_matrix @ density_matrix_swapped @ np.conj(full_matrix.T)

        for k1 in reversed(range(1, num_of_target_qubits + 1)):
            density_matrix_swapped = swap_qubits_density(target_qubits[k1 - 1], k1, num_of_qubits, density_matrix_swapped)
        density_matrix = swap_qubits_density(control_qubits[0], 0, num_of_qubits, density_matrix_swapped)

        return density_matrix
    

# -------------------------------------------------------------------------------------------
class CX(GateControl):
    def __init__(self, control_qubits: int=0, target_qubits: int=1) -> None:
        matrix_books = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 1, 0]])
        super().__init__(name='CX', matrix=matrix_books, base_matrix=X().matrix, control_qubits=control_qubits, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class CZ(GateControl):
    def __init__(self, control_qubits: int=0, target_qubits: int=1) -> None:
        matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]])
        super().__init__(name='CZ', matrix=matrix, base_matrix=Z().matrix, control_qubits=control_qubits, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class CS(GateControl):
    def __init__(self, control_qubits: int=0, target_qubits: int=1) -> None:
        matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1j]])
        super().__init__(name='CS', matrix=matrix, base_matrix=S().matrix, control_qubits=control_qubits, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class CSX(GateControl):
    def __init__(self, control_qubits: int=0, target_qubits: int=1) -> None:
        base_matrix = [[np.exp(+1j * np.pi / 4), np.exp(-1j * np.pi / 4)],
                       [np.exp(-1j * np.pi / 4), np.exp(+1j * np.pi / 4)]]
        matrix = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, (1 + 1j) / 2, (1 - 1j) / 2],
                  [0, 0, (1 - 1j) / 2, (1 + 1j) / 2]]
        super().__init__(name='CSX', matrix=matrix, base_matrix=base_matrix, control_qubits=control_qubits, target_qubits=target_qubits)


# -------------------------------------------------------------------------------------------
class _AxisRotationGate(GateControl):
    _gate_name: str = ''    # set by each subclass

    def __init__(self, phase: Union[float, str], control_qubits: int=0, target_qubits: int=1) -> None:
        self._phase= phase
        matrix = self.update_matrix()
        base_matrix = None if isinstance(phase, str) else self._base_matrix_for_param(phase)
        super().__init__(name=self._gate_name, matrix=matrix, base_matrix=base_matrix,
                          control_qubits=control_qubits, target_qubits=target_qubits)

    @property
    def phase(self) -> Union[float, str]:
        return self._phase


    def _base_matrix_for_param(self, phase: float) -> np.ndarray:
        raise NotImplementedError

    def update_matrix(self) -> Optional[np.ndarray]:
        if isinstance(self._phase, str): matrix = None
        else:
            base = self._base_matrix_for_param(self._phase)
            matrix = np.eye(4, dtype=complex)
            matrix[2:, 2:] = base

        self.matrix = matrix
        return matrix

# -------------------------------------------------------------------------------------------
class CP(_AxisRotationGate):
    _gate_name = 'CP'

    def __init__(self, phi: Union[float, str], control_qubits: int=0, target_qubits: int=1) -> None:
        super().__init__(phase=phi, control_qubits=control_qubits, target_qubits=target_qubits)

    def _base_matrix_for_param(self, phi: float) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * phi)]])
    
# -------------------------------------------------------------------------------------------
class CRX(_AxisRotationGate):
    _gate_name = 'CRX'

    def __init__(self, theta: Union[float, str], control_qubits: int=0, target_qubits: int=1) -> None:
        super().__init__(phase=theta, control_qubits=control_qubits, target_qubits=target_qubits)

    def _base_matrix_for_param(self, theta: float) -> np.ndarray:
        return np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ])

# -------------------------------------------------------------------------------------------
class CRY(_AxisRotationGate):
    _gate_name = 'CRY'

    def __init__(self, theta: Union[float, str], control_qubits: int=0, target_qubits: int=1) -> None:
        super().__init__(phase=theta, control_qubits=control_qubits, target_qubits=target_qubits)

    def _base_matrix_for_param(self, theta: float) -> np.ndarray:
        return np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])

# -------------------------------------------------------------------------------------------
class CRZ(_AxisRotationGate):
    _gate_name = 'CRZ'

    def __init__(self, theta: Union[float, str], control_qubits: int=0, target_qubits: int=1) -> None:
        super().__init__(phase=theta, control_qubits=control_qubits, target_qubits=target_qubits)

    def _base_matrix_for_param(self, theta: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * theta / 2), 0.0],
            [0.0, np.exp(1j * theta / 2)]
        ])


# Toffoli, controlled-controlled NOT --------------------------------------------------------
class CCX(GateControl):
    def __init__(self, qubits_1: int=0, qubits_2: int=1, qubits_3: int=2) -> None:
        matrix = np.eye(8)

        # ------------------------
        # based on q0 as control | Qiskit representation
        # matrix[3,7] = 1
        # matrix[7,3] = 1
        # matrix[3,3] = 0
        # matrix[7,7] = 0

        matrix[6, 7] = 1
        matrix[7, 6] = 1
        matrix[6, 6] = 0
        matrix[7, 7] = 0

        controls = [qubits_1]
        targets = sorted([qubits_2, qubits_3])
        super().__init__(name='CCX', matrix=matrix, base_matrix=CX().matrix, control_qubits=controls, target_qubits=targets)

# Fredkin, controlled swap ------------------------------------------------------------------
class CSWAP(GateControl):
    def __init__(self, qubits_1: int=0, qubits_2: int=1, qubits_3: int=2) -> None:
        matrix = np.eye(8)
        # ------------------------
        # based on q0 as control | Qiskit representation
        #matrix[3,3] = 0
        #matrix[5,5] = 0
        #matrix[3,5] = 1
        #matrix[5,3] = 1

        matrix[5, 6] = 1
        matrix[6, 5] = 1
        matrix[5, 5] = 0
        matrix[6, 6] = 0

        controls = [qubits_1]
        targets = sorted([qubits_2, qubits_3])
        super().__init__(name='CSWAP', matrix=matrix, base_matrix=SWAP().matrix, control_qubits=controls, target_qubits=targets)

# Control with an arbitray matrix - defined by the user -------------------------------------
class CU(GateControl):
    def __init__(self, base_matrix: np.ndarray, control_qubits: int=0, target_qubits: Union[int, List[int]]=1) -> None:
        base_matrix = np.asarray(base_matrix, dtype=complex)

        if not is_unitary(base_matrix):
            raise ValueError('Custom matrix is not unitary')

        target_qubits_arr = np.atleast_1d(np.asarray(target_qubits, dtype=int)).flatten()
        expected_base_dim = 2 ** target_qubits_arr.size
        if base_matrix.shape[0] != expected_base_dim:
            raise ValueError(
                "CU's base_matrix must be {0}x{0} to match {1} target qubit(s), got shape {2}."
                .format(expected_base_dim, target_qubits_arr.size, base_matrix.shape)
            )

        num_of_I_matrices = int(np.log2(base_matrix.shape[0]))
        zero = np.array([[1], [0]])
        one  = np.array([[0], [1]])
        zero_zero = np.kron(zero, zero.T)
        one_one   = np.kron(one, one.T)
        I_matrix  = I().matrix

        # based on books (and not qiskit) --> |0><0| (x) I + |1><1| (x) G
        cmatrix_1 = np.kron(zero_zero, I_matrix)
        for _ in range(num_of_I_matrices - 1):
            cmatrix_1 = np.kron(cmatrix_1, I_matrix)
        cmatrix_2 = np.kron(one_one, base_matrix)
        controlled_matrix = cmatrix_1 + cmatrix_2

        super().__init__(name='CU', matrix=controlled_matrix, base_matrix=base_matrix, control_qubits=control_qubits, target_qubits=target_qubits)
