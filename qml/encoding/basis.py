from typing import List, Optional, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import X
from AriaQuanta.qml._shared import validate_binary_data


# -------------------------------------------------------------------------------------------
def basis_encoding(data: Union[List[int], np.ndarray], num_qubits: Optional[int]=None) -> Circuit:
    """
    Basis encoding: maps a classical bit string b = (b_0, ..., b_{n-1}) directly onto the
    computational basis state |b_0 b_1 ... b_{n-1}>, by applying an X gate on every qubit
    whose corresponding bit is 1.

    The simplest and cheapest (gate-wise) of the encodings, but the least expressive: it
    only ever prepares a single basis state, never a superposition.

    :param data: A sequence of 0/1 values, one per qubit to encode.
    :param num_of_qubits: Number of qubits. Defaults to len(data); if larger, the extra
                           (trailing) qubits are left in |0>.
    :return: A new Circuit with the basis-encoding X gates applied (not yet run).
    """
    bits = validate_binary_data(data)
    n = num_qubits if num_qubits is not None else len(bits)
    if n < len(bits):
        raise ValueError("num_qubits ({}) must be at least len(data) ({}).".format(n, len(bits)))

    qc = Circuit(n)
    for i, bit in enumerate(bits):
        if bit == 1:
            qc | X(i)
    return qc
