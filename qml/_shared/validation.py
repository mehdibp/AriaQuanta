from typing import List, Sequence, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary import RX, RY, RZ, P


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

