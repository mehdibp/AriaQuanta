from typing import List, Optional, Sequence, Union

from AriaQuanta._utils import np

# -------------------------------------------------------------------------------------------
# Shared building blocks for AriaQuanta.qml.encoding -- kept in one place so every encoding
# method (and the future PQC/VQC layers that will reuse angle-style encoding) validates data
# and builds gates the same way, instead of each file re-deriving its own conventions.
# -------------------------------------------------------------------------------------------


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
