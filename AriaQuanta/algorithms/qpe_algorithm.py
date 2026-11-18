from typing import List

from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary import H, CU
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.algorithms import iqft


# Quantum Phase Estimation Algorithm --------------------------------------------------------
def qpe(unitary_matrix: np.ndarray, t_counting_qubits: int, namedraw: str = 'CU') -> Circuit:
    """
    Implements the Quantum Phase Estimation algorithm.

    Estimates the phase phi of an eigenvalue e^(2*pi*i*phi) of unitary_matrix, given the
    circuit's target qubit(s) are prepared in the corresponding eigenstate before running.
    The counting qubits (indices 0..t_counting_qubits-1) end up holding the t-bit binary
    expansion of phi (qubit 0 = most significant bit), and the target qubits are the last
    log2(unitary_matrix.shape[0]) qubits of the circuit.

    :param unitary_matrix: The unitary operator whose eigenphase is being estimated; a
                            2^k x 2^k matrix (k target qubits, k=1 for a single-qubit unitary).
    :param t_counting_qubits: Number of counting qubits (controls the estimate's precision).
    :param namedraw: Label to use for the controlled-unitary gate when drawing the circuit.
    :return: The built Circuit (not yet run; the target qubits still need to be prepared
             in an eigenstate of unitary_matrix before qc.run()).
    """
    
    dim = unitary_matrix.shape[0]
    num_target_qubits = int(round(np.log2(dim)))
    unitary_matrix = np.asarray(unitary_matrix, dtype=complex)
    _check_validation(t_counting_qubits, unitary_matrix, num_target_qubits, dim)


    target_qubits: List[int] = list(range(t_counting_qubits, t_counting_qubits + num_target_qubits))
    qc = Circuit(t_counting_qubits + num_target_qubits)

    # Step 1: Apply Hadamard gates to counting qubits
    for i in range(t_counting_qubits):
        qc | H(i)

    # Step 2: Apply controlled unitary gates: counting qubit i controls U^(2^(t-1-i)),
    # so qubit 0 (the most significant counting qubit) controls the largest power.
    powers = [2 ** i for i in range(t_counting_qubits - 1, -1, -1)]

    for i in range(t_counting_qubits):
        myCU = CU(unitary_matrix, control_qubits=i, target_qubits=target_qubits)
        myCU.namedraw = namedraw
        for _ in range(powers[i]):
            qc | myCU

    # Step 3: Apply Inverse Quantum Fourier Transform on the counting qubits
    qc = iqft(qc, list(range(t_counting_qubits)))

    # Step 4: Measure counting qubits
    #qc.measure_all(range(t_counting_qubits), range(t_counting_qubits))

    return qc


# -------------------------------------------------------------------------------------------
def _check_validation(t_counting_qubits: int, unitary_matrix: np.ndarray, num_target_qubits: int, dim):
    if not isinstance(t_counting_qubits, int) or isinstance(t_counting_qubits, bool):
        raise TypeError("'t_counting_qubits' must be an int, got {}.".format(type(t_counting_qubits).__name__))
    if t_counting_qubits < 1:
        raise ValueError("'t_counting_qubits' must be at least 1, got {}.".format(t_counting_qubits))
    if unitary_matrix.ndim != 2 or unitary_matrix.shape[0] != unitary_matrix.shape[1]:
        raise ValueError("'unitary_matrix' must be a square matrix, got shape {}.".format(unitary_matrix.shape))
    if 2 ** num_target_qubits != dim:
        raise ValueError("'unitary_matrix' dimension must be a power of 2, got {}.".format(dim))
