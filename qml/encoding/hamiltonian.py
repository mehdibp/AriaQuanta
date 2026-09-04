from typing import Optional, Sequence

from scipy.linalg import expm

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import Custom, X, Y, Z
from AriaQuanta.algorithms.eigen_solver import Hamiltonian, _parse_pauli_string


# NOTE: `_parse_pauli_string` is reused as-is from AriaQuanta.algorithms.eigen_solver rather
# than re-implemented here, so the Pauli-string grammar ('Z0X1', 'I', ...) stays identical
# across expectation-value estimation (VQE/QAOA) and encoding. It's currently a private
# helper of eigen_solver; if this cross-module reuse feels wrong, the fix is just to drop
# the leading underscore and export it from eigen_solver as a small public utility.

_PAULI_MATRIX = {'X': X().matrix, 'Y': Y().matrix, 'Z': Z().matrix}


# ------------------------------------------------------------
def hamiltonian_encoding(data, pauli_strings: Sequence[str], t: float = 1.0, num_of_qubits: Optional[int] = None) -> Circuit:
    """
    Hamiltonian (time-evolution) encoding: builds a data-dependent Hamiltonian
    H(x) = sum_i x_i * P_i from the classical features x and a *fixed* list of Pauli
    strings P_i, then applies the unitary U(x) = exp(-i * t * H(x)) to the circuit as one
    block.

    This is a natural fit when the features are physically meaningful couplings/fields
    (e.g. re-scaling terms of a spin-model Hamiltonian): each feature directly modulates
    one term's strength, and the encoding *is* the resulting time evolution. It's also the
    most expensive of the encodings here (a dense 2**n x 2**n matrix exponential), so it's
    best kept to small qubit counts unless/until AriaQuanta grows a Trotterized version.

    :param data: One coefficient per entry of pauli_strings (i.e. len(data) == len(pauli_strings)).
    :param pauli_strings: Fixed list of Pauli strings, e.g. ['Z0', 'Z1', 'X0X1'].
    :param t: Evolution time.
    :param num_of_qubits: Number of qubits. Defaults to 1 + the highest qubit index
                           referenced in pauli_strings.
    :return: A new Circuit with U(x) applied as a single Custom gate (not yet run).
    """
    data_arr = np.asarray(data, dtype=float).flatten()
    if data_arr.size != len(pauli_strings):
        raise ValueError(
            "'data' must have one value per Pauli string ({}), got {}."
            .format(len(pauli_strings), data_arr.size)
        )

    hamiltonian = Hamiltonian(list(zip(pauli_strings, data_arr.tolist())))   # validates format/duplicates

    if num_of_qubits is None:
        max_qubit = 0
        for pauli_string in hamiltonian.paulis:
            parsed = _parse_pauli_string(pauli_string)
            if parsed:
                max_qubit = max(max_qubit, max(q for q, _ in parsed))
        num_of_qubits = max_qubit + 1

    matrix = hamiltonian_matrix(hamiltonian.paulis, hamiltonian.coefs, num_of_qubits)
    unitary = expm(-1j * t * matrix)

    qc = Circuit(num_of_qubits)
    qc | Custom(matrix=unitary, target_qubits=list(range(num_of_qubits)), name='HamEnc')
    return qc


# ------------------------------------------------------------
def hamiltonian_matrix(pauli_strings: Sequence[str], coefficients: Sequence[float], num_of_qubits: int) -> np.ndarray:
    """Builds the full 2**n x 2**n matrix of H = sum_i coefficients[i] * pauli_strings[i]."""
    if len(pauli_strings) != len(coefficients):
        raise ValueError(
            "pauli_strings and coefficients must have the same length, got {} and {}."
            .format(len(pauli_strings), len(coefficients))
        )

    dim = 2 ** num_of_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for pauli_string, coef in zip(pauli_strings, coefficients):
        matrix = matrix + coef * _term_matrix(pauli_string, num_of_qubits)
    return matrix


# ------------------------------------------------------------
def _term_matrix(pauli_string: str, num_of_qubits: int) -> np.ndarray:
    active = dict(_parse_pauli_string(pauli_string, num_of_qubits))   # {qubit: 'X'|'Y'|'Z'}
    matrix = np.array([[1.0]], dtype=complex)
    for q in range(num_of_qubits):
        block = _PAULI_MATRIX[active[q]] if q in active else np.eye(2, dtype=complex)
        matrix = np.kron(matrix, block)
    return matrix
