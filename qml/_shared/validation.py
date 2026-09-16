from typing import List, Sequence, Union, Callable

from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary import RX, RY, RZ, P
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.algorithms.eigen_solver import Hamiltonian


# P has no symbolic-parameter support (see gatesingle.P), so it's excluded from anything that
# needs to bind a *trainable* (string-named) angle -- only from RX/RY/RZ-style data encoding.
TRAINABLE_ROTATION_GATES = {'RX': RX, 'RY': RY, 'RZ': RZ}
ROTATION_GATES           = {'RX': RX, 'RY': RY, 'RZ': RZ, 'P': P}


# ------------------------------------------------------------
def validate_rotation_names(rotation: Union[str, Sequence[str]], trainable: bool=False) -> List[str]:
    rotations = [rotation] if isinstance(rotation, str) else list(rotation)
    if not rotations:
        raise ValueError("'rotation' must name at least one gate.")

    table = TRAINABLE_ROTATION_GATES if trainable else ROTATION_GATES
    for r in rotations:
        if r not in table:
            raise ValueError("rotation must be one of {}, got {!r}.".format(list(table), r))
    return rotations

# ------------------------------------------------------------
def validate_features(data) -> np.ndarray:
    features = np.asarray(data, dtype=float).flatten()
    if features.size == 0:
        raise ValueError("'data' must contain at least one feature.")
    if not np.all(np.isfinite(features)):
        raise ValueError("'data' contains NaN or Inf values.")
    return features

# ------------------------------------------------------------
def validate_binary_data(data) -> List[int]:
    arr = np.asarray(data, dtype=float).flatten()
    if arr.size == 0:
        raise ValueError("'data' must contain at least one bit.")

    bits: List[int] = []
    for value in arr:
        rounded = int(round(float(value)))
        if rounded not in (0, 1) or abs(float(value) - rounded) > 1e-6:
            raise ValueError("basis_encoding expects 0/1 values, got {}.".format(value))
        bits.append(rounded)
    return bits



# ------------------------------------------------------------
def validate_pauli_blocks(paulis: Union[str, Sequence[str]]) -> List[str]:
    blocks = [paulis] if isinstance(paulis, str) else list(paulis)
    if not blocks:
        raise ValueError("'paulis' must name at least one Pauli block (e.g. 'Z' or ['Z', 'ZZ']).")
    for b in blocks:
        if not isinstance(b, str) or not b or any(c not in 'XYZ' for c in b):
            raise ValueError("Each Pauli block must be a non-empty string over {{X, Y, Z}}, got {!r}.".format(b))
    return blocks


# ------------------------------------------------------------
def validate_parameter_shift(ansatz: Ansatz, evaluate_fn: Callable[[Ansatz], float], shift: float, base_values: np.array) -> None:
    if not isinstance(ansatz, Ansatz):
        raise TypeError("'ansatz' must be an Ansatz instance, got {}.".format(type(ansatz).__name__))
    if not callable(evaluate_fn):
        raise TypeError("'evaluate_fn' must be callable, got {}.".format(type(evaluate_fn).__name__))
    if np.isclose(np.sin(shift), 0.0, atol=1e-12):
        raise ValueError(
            "'shift' must not be a multiple of pi (sin(shift) would be 0, making the "
            "parameter-shift estimator undefined), got {}.".format(shift)
        )
    if base_values.size != len(ansatz.params_names):
        raise ValueError(
            "'params_values' must have length {} (= number of ansatz parameters), got {}."
            .format(len(ansatz.params_names), base_values.size)
        )


# ------------------------------------------------------------
def validate_parameter_shift_expectation(hamiltonian, num_of_iter_measure: int) -> None:
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("'hamiltonian' must be a Hamiltonian instance, got {}.".format(type(hamiltonian).__name__))
    if not isinstance(num_of_iter_measure, int) or isinstance(num_of_iter_measure, bool):
        raise TypeError("'num_of_iter_measure' must be an int, got {}.".format(type(num_of_iter_measure).__name__))
    if num_of_iter_measure < 1:
        raise ValueError("'num_of_iter_measure' must be at least 1, got {}.".format(num_of_iter_measure))


