import math
from typing import List

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import H, CP, SWAP


# -------------------------------------------------------------------------------------------
def qft(qc: Circuit, qubits: List[int]) -> Circuit:
    """
    Apply the Quantum Fourier Transform (QFT) on the specified qubits.

    This is the exact inverse of AriaQuanta.algorithms.iqft_algorithm.iqft: applying
    qft(qc, qubits) followed by iqft(qc, qubits) returns the state to what it was
    before the qft call.

    :param qc: Circuit to apply the QFT to (modified in place).
    :param qubits: List of qubit indices to apply the QFT on.
    :return: The same circuit qc, for chaining.
    """
    if not isinstance(qubits, (list, tuple)) or len(qubits) == 0:
        raise ValueError("qubits must be a non-empty list of qubit indices.")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"qubits must not contain duplicates, got {list(qubits)}.")

    n = len(qubits)

    for i in range(n):
        # Apply Hadamard gate to qubit i
        qc | H(qubits[i])

        # Apply controlled phase shifts
        for j in range(i+1, n):
            angle = math.pi / (2**(j-i))
            qc | CP(angle, qubits[j], qubits[i])    # Controlled phase gate

    # Reverse the order of qubits using SWAP gates
    qubits_reversed = qubits[::-1]
    for i in range(n // 2):
        qc | SWAP(qubits[i], qubits_reversed[i])

    return qc


