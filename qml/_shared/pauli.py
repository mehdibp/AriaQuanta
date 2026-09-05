from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary import X, Y, Z


# -------------------------------------------------------------------------------------------
# Small shared kernel used by AriaQuanta.qml.encoding.hamiltonian and every
# AriaQuanta.qml.feature_map -- both need to turn a string of Pauli letters into a matrix,
# just in slightly different ways (padded to the full circuit vs. left as a small block for
# GateCustom's own embedding). Kept here, one level above encoding/feature_map/ansatz, so
# neither sub-package has to import from the other for it.
# -------------------------------------------------------------------------------------------

PAULI_MATRIX = {'X': X().matrix, 'Y': Y().matrix, 'Z': Z().matrix, 'I': np.eye(2, dtype=complex)}

# -------------------------------------------------------------------------------------------
def pauli_tensor_matrix(pauli_chars) -> np.ndarray:
    """Kronecker product of PAULI_MATRIX[c] for c in pauli_chars, in order (no padding)."""
    if not pauli_chars:
        raise ValueError("pauli_chars must be a non-empty string/sequence of Pauli letters.")

    matrix = np.array([[1.0]], dtype=complex)
    for c in pauli_chars:
        if c not in PAULI_MATRIX:
            raise ValueError("Unknown Pauli letter {!r}; expected one of {}.".format(c, list(PAULI_MATRIX)))
        matrix = np.kron(matrix, PAULI_MATRIX[c])
    return matrix

# -------------------------------------------------------------------------------------------
def pauli_evolution_matrix(pauli_chars, phi: float) -> np.ndarray:
    """
    Closed-form exp(-i * phi * P) for a Pauli tensor P = kron(PAULI_MATRIX[c] for c in
    pauli_chars). Uses P^2 = I (true for any tensor product of single-qubit Paulis/identity),
    so exp(-i*phi*P) = cos(phi)*I - i*sin(phi)*P exactly -- no numerical matrix exponential
    (and no unitarity-tolerance risk) needed, unlike the general multi-term case in
    AriaQuanta.qml.encoding.hamiltonian_encoding.
    """
    P = pauli_tensor_matrix(pauli_chars)
    dim = P.shape[0]
    return np.cos(phi)*np.eye(dim, dtype=complex) - 1j*np.sin(phi)*P    # cos(φ)*I - i*sin(φ)*P
