from typing import List, Optional, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase


# -------------------------------------------------------------------------------------------
class GateSingleQubit(GateBase):
    _num_target_qubits = None   ###
    
    # ------------------------------------------------------------
    def _full_matrix(self, num_of_qubits: int) -> np.ndarray:
        if self.matrix is None:
            raise ValueError(
                "Gate '{}' has a symbolic parameter (e.g. theta given as a string placeholder for circuit diagrams) "
                "and therefore no numeric matrix to apply. Bind it to a numeric value first.".format(self.name)
            )

        target_qubit = int(self.target_qubits[0])

        if target_qubit > 0: I1 = np.identity(2 ** target_qubit, dtype=complex)
        else: I1 = 1

        remaining = num_of_qubits - target_qubit - 1
        if remaining > 0: I2 = np.identity(2 ** remaining, dtype=complex)
        else: I2 = 1

        full_matrix = np.kron(I1, self.matrix)
        full_matrix = np.kron(full_matrix, I2)
        return full_matrix

    # ------------------------------------------------------------
    def apply(self, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
        full_matrix = self._full_matrix(num_of_qubits)
        multistate = np.dot(full_matrix, multistate)
        return multistate
    
    # ------------------------------------------------------------
    def apply_density(self, num_of_qubits: int, density_matrix: np.ndarray) -> np.ndarray:
        full_matrix = self._full_matrix(num_of_qubits)
        density_matrix = full_matrix @ density_matrix @ np.conj(full_matrix.T)
        return density_matrix        


# 1-Qubit Gates -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
class I(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = np.eye(2)
        super().__init__(name='I', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class GlobalPhase(GateSingleQubit):
    def __init__(self, delta: float, target_qubits: Union[int, List[int]]=0):
        self.delta = delta
        matrix = np.exp(+1j * delta) * np.eye(2)
        super().__init__(name='GPh', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class X(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = np.array([[0, 1], [1, 0]])
        super().__init__(name='X', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Y(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = np.array([[0, -1j], [1j, 0]])
        super().__init__(name='Y', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Z(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = np.array([[1, 0], [0, -1]])
        super().__init__(name='Z', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class S(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = np.array([[1, 0], [0, 1j]])
        super().__init__(name='S', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Sdg(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = np.array([[1, 0], [0, -1j]])
        super().__init__(name='Sdg', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Xsqrt(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = 1/2 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]])
        super().__init__(name='Xsqrt', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class H(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        super().__init__(name='H', matrix=matrix, target_qubits=target_qubits)  

# -------------------------------------------------------------------------------------------
class P(GateSingleQubit):
    def __init__(self, phi: float, target_qubits: Union[int, List[int]]=0):
        self.phase = phi
        matrix = np.array([[1, 0], [0, np.exp(1j * phi)]])
        super().__init__(name='P', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class T(GateSingleQubit):
    def __init__(self, target_qubits: Union[int, List[int]]=0):
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
        super().__init__(name='T', matrix=matrix, target_qubits=target_qubits)


# -------------------------------------------------------------------------------------------
class _AxisRotationGate(GateSingleQubit):
    _gate_name: str = ''    # set by each subclass

    def __init__(self, phase: Union[float, str], target_qubits: Union[int, List[int]]=0) -> None:
        self._phase = phase
        matrix = self.update_matrix()
        super().__init__(name=self._gate_name, matrix=matrix, target_qubits=target_qubits)

    @property
    def phase(self) -> Union[float, str]:
        return self._phase


    def _matrix_for_theta(self, phase: float) -> np.ndarray:
        raise NotImplementedError

    def update_matrix(self) -> Optional[np.ndarray]:
        if isinstance(self.phase, str): matrix = None
        else: matrix = self._matrix_for_theta(self.phase)

        self.matrix = matrix
        return matrix

# -------------------------------------------------------------------------------------------
class RX(_AxisRotationGate):
    _gate_name = 'RX'

    def _matrix_for_theta(self, theta: float) -> np.ndarray:
        return np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ])
 
# -------------------------------------------------------------------------------------------
class RY(_AxisRotationGate):
    _gate_name = 'RY'

    def _matrix_for_theta(self, theta: float) -> np.ndarray:
        return np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])

# -------------------------------------------------------------------------------------------
class RZ(_AxisRotationGate):
    _gate_name = 'RZ'

    def _matrix_for_theta(self, theta: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * theta / 2), 0.0],
            [0.0, np.exp(1j * theta / 2)]
        ])

# ------------------------------------------------------------------------------------------- 
class Rot(GateSingleQubit):
    def __init__(self, theta: float, phi: float, lambda_: float, target_qubits: Union[int, List[int]]=0):
        self.theta   = theta
        self.phi     = phi
        self.lambda_ = lambda_
        matrix = np.array([
            [np.cos(self.theta / 2), -np.exp(1j * self.lambda_) * np.sin(self.theta / 2)],
            [np.exp(1j * self.phi) * np.sin(self.theta / 2), np.exp(1j * (self.lambda_ + self.phi)) * np.cos(self.theta / 2)]
        ])
        super().__init__(name='Rot', matrix=matrix, target_qubits=target_qubits)   
