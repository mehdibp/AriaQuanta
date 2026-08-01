from typing import Optional, Union
from scipy.linalg import expm

from AriaQuanta._utils import np, swap_qubits, swap_qubits_density
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase


# -------------------------------------------------------------------------------------------
class GateDoubleQubit(GateBase):
    _num_target_qubits = 2

    # ------------------------------------------------------------
    def _full_matrix(self, num_of_qubits: int) -> np.ndarray:
        if self.matrix is None:
            raise ValueError(
                "Gate '{}' has a symbolic parameter (e.g. an angle given as a string placeholder for circuit diagrams) "
                "and therefore no numeric matrix to apply. Bind it to a numeric value first.".format(self.name)
            )

        remaining = num_of_qubits - self._num_target_qubits
        if remaining > 0: I2 = np.identity(2 ** remaining, dtype=complex)
        else: I2 = 1

        return np.kron(self.matrix, I2)

    # ------------------------------------------------------------
    def apply(self, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
        target_qubits = self.target_qubits

        multistate_swapped = swap_qubits(0, target_qubits[0], num_of_qubits, multistate)
        multistate_swapped = swap_qubits(1, target_qubits[1], num_of_qubits, multistate_swapped)

        full_matrix = self._full_matrix(num_of_qubits)
        multistate_swapped = np.dot(full_matrix, multistate_swapped)

        multistate_swapped = swap_qubits(target_qubits[1], 1, num_of_qubits, multistate_swapped)
        multistate = swap_qubits(target_qubits[0], 0, num_of_qubits, multistate_swapped)

        return multistate

    # ------------------------------------------------------------
    def apply_density(self, num_of_qubits: int, density_matrix: np.ndarray) -> np.ndarray:
        target_qubits = self.target_qubits

        density_matrix_swapped = swap_qubits_density(0, target_qubits[0], num_of_qubits, density_matrix)
        density_matrix_swapped = swap_qubits_density(1, target_qubits[1], num_of_qubits, density_matrix_swapped)

        full_matrix = self._full_matrix(num_of_qubits)
        density_matrix_swapped = full_matrix @ density_matrix_swapped @ np.conj(full_matrix.T)

        density_matrix_swapped = swap_qubits_density(target_qubits[1], 1, num_of_qubits, density_matrix_swapped)
        density_matrix = swap_qubits_density(target_qubits[0], 0, num_of_qubits, density_matrix_swapped)

        return density_matrix


# 2-Qubit Gates -----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
class SWAP(GateDoubleQubit):
    def __init__(self, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        matrix = [[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='SWAP', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class ISWAP(GateDoubleQubit):
    def __init__(self, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        matrix = [[1, 0, 0, 0],
                  [0, 0, +1j, 0],
                  [0, +1j, 0, 0],
                  [0, 0, 0, 1]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='ISWAP', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class SWAPsqrt(GateDoubleQubit):
    def __init__(self, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        matrix = [[1, 0, 0, 0],
                  [0, 1 / 2 * (1 + 1j), 1 / 2 * (1 - 1j), 0],
                  [0, 1 / 2 * (1 - 1j), 1 / 2 * (1 + 1j), 0],
                  [0, 0, 0, 1]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='SWAPsqrt', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class ISWAPsqrt(GateDoubleQubit):
    def __init__(self, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        matrix = [[1, 0, 0, 0],
                  [0, 1 / np.sqrt(2), +1j / np.sqrt(2), 0],
                  [0, +1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                  [0, 0, 0, 1]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='ISWAPsqrt', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class SWAPalpha(GateDoubleQubit):
    def __init__(self, alpha: float, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        self.alpha = alpha
        matrix = [[1, 0, 0, 0],
                  [0, 1 / 2 * (1 + np.exp(+1j * np.pi * alpha)), 1 / 2 * (1 - np.exp(+1j * np.pi * alpha)), 0],
                  [0, 1 / 2 * (1 - np.exp(+1j * np.pi * alpha)), 1 / 2 * (1 + np.exp(+1j * np.pi * alpha)), 0],
                  [0, 0, 0, 1]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='SWAPalpha', matrix=matrix, target_qubits=target_qubits)


# -------------------------------------------------------------------------------------------
class _AxisRotationGate(GateDoubleQubit):
    _gate_name: str = ''    # set by each subclass

    def __init__(self, phase: Union[float, str], target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        self._phase = phase
        matrix = self.update_matrix()
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name=self._gate_name, matrix=matrix, target_qubits=target_qubits)
    
    @property
    def phase(self) -> Union[float, str]:
        return self._phase
    
    
    def _matrix_for_param(self, phase: float) -> np.ndarray:
        raise NotImplementedError

    def update_matrix(self) -> Optional[np.ndarray]:
        if isinstance(self._phase, str): matrix = None
        else: matrix = self._matrix_for_param(self._phase)

        self.matrix = matrix
        return matrix

# -------------------------------------------------------------------------------------------
class RXX(_AxisRotationGate):
    _gate_name = 'RXX'

    def _matrix_for_param(self, phi: float) -> np.ndarray:
        return [[np.cos(phi / 2), 0, 0, -1j * np.sin(phi / 2)],
                [0, np.cos(phi / 2), -1j * np.sin(phi / 2), 0],
                [0, -1j * np.sin(phi / 2), np.cos(phi / 2), 0],
                [-1j * np.sin(phi / 2), 0, 0, np.cos(phi / 2)]]

# -------------------------------------------------------------------------------------------
class RYY(_AxisRotationGate):
    _gate_name = 'RYY'

    def _matrix_for_param(self, phi: float) -> np.ndarray:
        return [[np.cos(phi / 2), 0, 0, +1j * np.sin(phi / 2)],
                [0, np.cos(phi / 2), -1j * np.sin(phi / 2), 0],
                [0, -1j * np.sin(phi / 2), np.cos(phi / 2), 0],
                [+1j * np.sin(phi / 2), 0, 0, np.cos(phi / 2)]]

# -------------------------------------------------------------------------------------------
class RZZ(_AxisRotationGate):
    _gate_name = 'RZZ'

    def _matrix_for_param(self, phi: float) -> np.ndarray:
        return [[np.exp(-1j * phi / 2), 0, 0, 0],
                [0, np.exp(+1j * phi / 2), 0, 0],
                [0, 0, np.exp(+1j * phi / 2), 0],
                [0, 0, 0, np.exp(-1j * phi / 2)]]

# -------------------------------------------------------------------------------------------
class RXY(_AxisRotationGate):
    _gate_name = 'RXY'

    def _matrix_for_param(self, phi: float) -> np.ndarray:
        return [[1, 0, 0, 0],
                [0, np.cos(phi / 2), -1j * np.sin(phi / 2), 0],
                [0, -1j * np.sin(phi / 2), np.cos(phi / 2), 0],
                [0, 0, 0, 1]]


# -------------------------------------------------------------------------------------------
class Barenco(GateDoubleQubit):
    def __init__(self, alpha: float, phi: float, theta: float, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        self.alpha = alpha
        self.phi = phi
        self.theta = theta
        matrix = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, np.exp(+1j * alpha) * np.cos(theta), -1j * np.exp(+1j * (alpha - phi)) * np.sin(theta)],
                  [0, 0, -1j * np.exp(+1j * (alpha + phi)) * np.sin(theta), np.exp(+1j * alpha) * np.cos(theta)]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='Barenco', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Berkeley(GateDoubleQubit):
    def __init__(self, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        a = np.pi / 8
        b = 3 * np.pi / 8
        
        matrix = [[np.cos(a), 0, 0, +1j * np.sin(a)],
                  [0, np.cos(b), +1j * np.sin(b), 0],
                  [0, +1j * np.sin(b), np.cos(b), 0],
                  [+1j * np.sin(a), 0, 0, np.cos(a)]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='Berkeley', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Canonical(GateDoubleQubit):
    def __init__(self, a: float, b: float, c: float, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        matrix = [[np.exp(+1j * c) * np.cos(a - b), 0, 0, +1j * np.exp(+1j * c) * np.sin(a - b)],
                  [0, np.exp(-1j * c) * np.cos(a + b), +1j * np.exp(-1j * c) * np.sin(a + b), 0],
                  [0, +1j * np.exp(-1j * c) * np.sin(a + b), np.exp(-1j * c) * np.cos(a + b), 0],
                  [+1j * np.exp(+1j * c) * np.sin(a - b), 0, 0, np.exp(+1j * c) * np.cos(a - b)]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='Canonical', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Givens(GateDoubleQubit):
    def __init__(self, theta: float, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        self.theta = theta
        matrix = [[1, 0, 0, 0],
                  [0, np.cos(theta), -np.sin(theta), 0],
                  [0, np.sin(theta), np.cos(theta), 0],
                  [0, 0, 0, 1]]
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='Givens', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Magic(GateDoubleQubit):
    def __init__(self, target_qubits_1: int=0, target_qubits_2: int=1) -> None:
        array = np.array([[1, +1j, 0, 0],
                          [0, 0, +1j, 1],
                          [0, 0, +1j, -1],
                          [1, -1j, 0, 0]])
        matrix = 1 / np.sqrt(2) * array
        target_qubits = sorted([target_qubits_1, target_qubits_2])
        super().__init__(name='Magic', matrix=matrix, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class Ruccsd(_AxisRotationGate, GateDoubleQubit):
    _gate_name = 'Ruccsd'

    def _matrix_for_param(self, theta: float) -> np.ndarray:
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        mat = np.kron(x, y)
        return expm(-1j * theta * mat)


