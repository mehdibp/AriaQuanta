from typing import List, Optional, Sequence, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary import RX, RY, RZ, P

# -------------------------------------------------------------------------------------------
# Shared building blocks for AriaQuanta.qml.encoding -- kept in one place so every encoding
# method (and the future PQC/VQC layers that will reuse angle-style encoding) validates data
# and builds gates the same way, instead of each file re-deriving its own conventions.
# -------------------------------------------------------------------------------------------

ROTATION_GATES = {'RX': RX, 'RY': RY, 'RZ': RZ, 'P': P}
# P has no symbolic-parameter support (see gatesingle.P), so it's excluded from anything that
# needs to bind a *trainable* (string-named) angle -- only from RX/RY/RZ-style data encoding.
TRAINABLE_ROTATION_GATES = {'RX': RX, 'RY': RY, 'RZ': RZ}


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
def amplitude_statevector(data, num_qubits: Optional[int], normalize: bool):
    arr = np.asarray(data, dtype=complex).flatten()
    if arr.size == 0:
        raise ValueError("'data' must contain at least one amplitude.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("'data' contains NaN or Inf values.")

    min_qubits = max(1, int(np.ceil(np.log2(arr.size))))
    n = num_qubits if num_qubits is not None else min_qubits
    dim = 2 ** n
    if arr.size > dim:
        raise ValueError(
            "'data' has {} amplitude(s), which needs at least {} qubit(s); got num_of_qubits={}."
            .format(arr.size, min_qubits, n)
        )

    padded = np.zeros(dim, dtype=complex)
    padded[:arr.size] = arr

    norm = float(np.linalg.norm(padded))
    if normalize:
        if norm == 0:
            raise ValueError("Cannot amplitude-encode an all-zero vector (norm is 0).")
        padded = padded / norm
    elif not np.isclose(norm, 1.0, atol=1e-8):
        raise ValueError(
            "'data' is not normalized (||data|| = {:.6f} != 1); pass normalize=True or supply a unit vector.".format(norm)
        )

    return padded.reshape(dim, 1), n

