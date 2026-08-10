from typing import TYPE_CHECKING
from functools import lru_cache
from AriaQuanta.config import Config, get_array_module


# -------------------------------------------------------------------------------------------
if TYPE_CHECKING: import numpy as np
else: np = get_array_module(Config.use_gpu)

# -------------------------------------------------------------------------------------------
def is_unitary(matrix: np.ndarray) -> bool:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: return False
    return bool(np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-10))

# -------------------------------------------------------------------------------------------
def reorder_state(state: np.ndarray) -> np.ndarray:
    num_of_states = np.shape(state)[0]
    num_of_qubits = int(np.log2(num_of_states))
    if 2 ** num_of_qubits != num_of_states:
        raise ValueError("state has {} rows, which isn't a power of two.".format(num_of_states))

    bin_format = '#0' + str(num_of_qubits + 2) + 'b'                        # #05b (b stands for binary)
    all_states = [format(x, bin_format)[2:] for x in range(num_of_states)]  # convert to binary format
    all_states = [x[::-1] for x in all_states]
    new_indices = [int(x, 2) for x in all_states]                          # binary to decimal
    reordered_state = state[new_indices]
    return reordered_state

# -------------------------------------------------------------------------------------------
def sv_to_density_matrix(statevector: np.ndarray) -> np.ndarray:
    return statevector @ statevector.conj().T


# -------------------------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _swap_index_permutation(idx1: int, idx2: int, num_of_qubits: int) -> np.ndarray:
    size = 2 ** num_of_qubits
    indices = np.arange(size)
    bit_pos1 = num_of_qubits - 1 - idx1
    bit_pos2 = num_of_qubits - 1 - idx2

    bit1 = (indices >> bit_pos1) & 1
    bit2 = (indices >> bit_pos2) & 1
    differing_bits = bit1 ^ bit2
    flip_mask = (differing_bits << bit_pos1) | (differing_bits << bit_pos2)

    return indices ^ flip_mask

# -------------------------------------------------------------------------------------------
def swap_qubits(idx1: int, idx2: int, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
    _validate_qubit_index(idx1, num_of_qubits, 'idx1')
    _validate_qubit_index(idx2, num_of_qubits, 'idx2')
    _validate_state_shape(multistate, num_of_qubits)

    if idx1 == idx2:
        return multistate

    idx1, idx2 = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
    permutation = _swap_index_permutation(idx1, idx2, num_of_qubits)

    return multistate[permutation]

# -------------------------------------------------------------------------------------------
def swap_qubits_density(idx1: int, idx2: int, num_of_qubits: int, density_matrix: np.ndarray) -> np.ndarray:
    _validate_qubit_index(idx1, num_of_qubits, 'idx1')
    _validate_qubit_index(idx2, num_of_qubits, 'idx2')
    _validate_density_shape(density_matrix, num_of_qubits)

    if idx1 == idx2:
        return density_matrix.copy()

    idx1, idx2 = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)      # was missing entirely -- idx1 > idx2 silently gave the wrong permutation
    permutation = _swap_index_permutation(idx1, idx2, num_of_qubits)

    density_matrix_swapped = density_matrix[:, permutation]
    density_matrix_swapped = density_matrix_swapped[permutation, :].copy()   # fancy indexing already returns a new array, but .copy() keeps that guarantee explicit rather than incidental
    return density_matrix_swapped



# validations -------------------------------------------------------------------------------
def _validate_qubit_index(idx: int, num_of_qubits: int, name: str) -> None:
    if not (0 <= idx < num_of_qubits):
        raise ValueError("'{}' must be between 0 and {} (num_of_qubits - 1), got {}.".format(name, num_of_qubits - 1, idx))

def _validate_state_shape(multistate: np.ndarray, num_of_qubits: int) -> None:
    expected = 2 ** num_of_qubits
    if multistate.shape[0] != expected:
        raise ValueError("multistate has {} rows, expected 2**num_of_qubits = {}.".format(multistate.shape[0], expected))

def _validate_density_shape(density_matrix: np.ndarray, num_of_qubits: int) -> None:
    expected = 2 ** num_of_qubits
    if density_matrix.ndim != 2 or density_matrix.shape[0] != expected or density_matrix.shape[1] != expected:
        raise ValueError("density_matrix has shape {}, expected ({}, {}).".format(density_matrix.shape, expected, expected))
