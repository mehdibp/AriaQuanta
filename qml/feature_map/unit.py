from typing import List, Tuple, Union, Sequence
from itertools import combinations
from AriaQuanta._utils import np

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import Custom, X, Y, Z, H


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



# -------------------------------------------------------------------------------------------
# Shared building blocks for AriaQuanta.qml.feature_map -- the Havlicek-style ("Supervised
# learning with quantum-enhanced feature spaces", Nature 2019) family of feature maps used
# mainly for quantum kernel methods. All of pauli_feature_map / z_feature_map /
# zz_feature_map / iqp_feature_map are thin, differently-parametrized calls into the same
# layer-builder here.
# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
def validate_pauli_blocks(paulis: Union[str, Sequence[str]]) -> List[str]:
    blocks = [paulis] if isinstance(paulis, str) else list(paulis)
    if not blocks:
        raise ValueError("'paulis' must name at least one Pauli block (e.g. 'Z' or ['Z', 'ZZ']).")
    for b in blocks:
        if not isinstance(b, str) or not b or any(c not in 'XYZ' for c in b):
            raise ValueError("Each Pauli block must be a non-empty string over {{X, Y, Z}}, got {!r}.".format(b))
    return blocks

# -------------------------------------------------------------------------------------------
def resolve_qubit_subsets(block_size: int, num_of_qubits: int,
                           entanglement: Union[str, Sequence[Tuple[int, ...]]]) -> List[Tuple[int, ...]]:
    """
    All qubit subsets a Pauli block of size `block_size` gets applied to.

    :param entanglement: 'full' (every combination of block_size qubits), 'linear' (only
                          consecutive windows, e.g. (0,1),(1,2),... for block_size=2), or an
                          explicit list of qubit-index tuples (used as-is, each of length
                          block_size). Ignored when block_size == 1 (applies to every qubit).
    """
    if block_size == 1:
        return [(q,) for q in range(num_of_qubits)]

    if isinstance(entanglement, str):
        if entanglement == 'full':
            return list(combinations(range(num_of_qubits), block_size))
        elif entanglement == 'linear':
            return [tuple(range(i, i + block_size)) for i in range(num_of_qubits - block_size + 1)]
        else:
            raise ValueError(
                "entanglement must be 'full', 'linear', or an explicit list of qubit tuples, got {!r}.".format(entanglement)
            )

    subsets = [tuple(s) for s in entanglement]
    for s in subsets:
        if len(s) != block_size:
            raise ValueError(
                "Every explicit qubit subset must have length {} (the Pauli block size), got {}.".format(block_size, s)
            )
        if max(s) >= num_of_qubits:
            raise ValueError("Qubit subset {} references a qubit >= num_of_qubits ({}).".format(s, num_of_qubits))
    return subsets

# -------------------------------------------------------------------------------------------
def default_data_map(features: np.ndarray, qubits: Tuple[int, ...]) -> float:
    """
    The standard Havlicek et al. data-mapping function: phi_S(x) = x_i for a single qubit
    S = {i}, or prod_{i in S} (pi - x_i) for |S| > 1. The product form for multi-qubit terms
    is what makes them carry genuinely new (nonlinear) information rather than just
    repeating a single feature's phase on several qubits at once.
    """
    if len(qubits) == 1:
        return float(features[qubits[0]])
    phi = 1.0
    for q in qubits:
        phi *= (np.pi - features[q])
    return float(phi)

# -------------------------------------------------------------------------------------------
def apply_pauli_feature_layer(target: Circuit, features: np.ndarray, blocks: List[str],
                               entanglement: Union[str, Sequence[Tuple[int, ...]]] = 'full',
                               data_map=default_data_map) -> None:
    """
    Applies one repetition of a Havlicek-style Pauli feature map onto `target` (a Circuit,
    via its '|' operator): a Hadamard on every qubit, followed by exp(-i * phi_S(x) * P_S)
    for every Pauli block in `blocks` and every qubit subset it applies to -- as separate
    gates in sequence (a *product* of evolutions, one per block/subset), not one joint
    exponential of a summed Hamiltonian (contrast with
    AriaQuanta.qml.encoding.hamiltonian_encoding, which does the latter).

    Without the Hadamard layer, a lone Z-type rotation on |0> would only be a global phase
    and encode nothing observable -- the Hadamards are what turn the data-dependent phases
    into something that shows up in measurement statistics/interference.
    """
    n = target.num_of_qubits
    for q in range(n):
        target | H(q)

    for block in blocks:
        subsets = resolve_qubit_subsets(len(block), n, entanglement)
        for qubits in subsets:
            phi = data_map(features, qubits)
            matrix = pauli_evolution_matrix(block, phi)
            target | Custom(matrix=matrix, target_qubits=list(qubits), name='P[{}]'.format(block))
