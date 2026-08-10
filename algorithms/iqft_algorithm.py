import math
from typing import List

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import H, CP, SWAP


# -------------------------------------------------------------------------------------------
def iqft(qc: Circuit, qubits: List[int]) -> Circuit:
    """
    Apply the Inverse Quantum Fourier Transform (IQFT) on the specified qubits.

    This is the exact inverse of AriaQuanta.algorithms.qft_algorithm.qft: applying
    qft(qc, qubits) followed by iqft(qc, qubits) returns the state to what it was
    before the qft call.

    :param qc: Circuit to apply the IQFT to (modified in place).
    :param qubits: List of qubit indices to apply the IQFT on.
    :return: The same circuit qc, for chaining.
    """
    if not isinstance(qubits, (list, tuple)) or len(qubits) == 0:
        raise ValueError("qubits must be a non-empty list of qubit indices.")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"qubits must not contain duplicates, got {list(qubits)}.")

    n = len(qubits)

    # Undo the qubit-order-reversing SWAPs qft() applies at the end (SWAP is self-inverse)
    qubits_reversed = qubits[::-1]
    for i in range(n // 2):
        qc | SWAP(qubits[i], qubits_reversed[i])

    # Apply IQFT: exact reverse of qft's gate sequence, with each controlled-phase
    # angle negated (qubit blocks in reverse order, and within each block the
    # phase corrections applied before the Hadamard).
    for i in reversed(range(n)):
        for j in reversed(range(i + 1, n)):
            angle = -math.pi / (2 ** (j - i))
            qc | CP(angle, qubits[j], qubits[i])  # Controlled phase gate

        qc | H(qubits[i])

    return qc   

