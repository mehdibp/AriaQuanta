from typing import Optional, Sequence

from scipy.linalg import expm

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import Custom
from AriaQuanta.aqc.hamiltonian import Hamiltonian, hamiltonian_matrix


# ------------------------------------------------------------
def hamiltonian_encoding(data, pauli_strings: Sequence[str], t: float=1.0, num_of_qubits: Optional[int]=None) -> Circuit:
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
    :param num_of_qubits: Number of qubits. Defaults to 1 + the highest qubit index referenced in pauli_strings.
    :return: A new Circuit with U(x) applied as a single Custom gate (not yet run).
    """
    data_arr = np.asarray(data, dtype=float).flatten()
    if data_arr.size != len(pauli_strings):
        raise ValueError("'data' must have one value per Pauli string ({}), got {}.".format(len(pauli_strings), data_arr.size))

    hamiltonian = Hamiltonian(list(zip(pauli_strings, data_arr.tolist())))   # validates format/duplicates

    if num_of_qubits is None:
        hamiltonian.min_num_of_qubits()

    matrix = hamiltonian_matrix(hamiltonian.paulis, hamiltonian.coefs, num_of_qubits)
    unitary = expm(-1j * t * matrix)

    qc = Circuit(num_of_qubits)
    qc | Custom(matrix=unitary, target_qubits=list(range(num_of_qubits)), name='HamEnc')
    return qc

