from typing import List, Optional, Sequence, Union

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
