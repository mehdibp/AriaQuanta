from typing import List, Tuple, Union, Sequence
from AriaQuanta._utils import np

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import Custom, H
from AriaQuanta.qml._shared.pauli import pauli_evolution_matrix
from AriaQuanta.qml._shared.connectivity import resolve_qubit_subsets


# -------------------------------------------------------------------------------------------
# Shared building blocks for AriaQuanta.qml.feature_map -- the Havlicek-style ("Supervised
# learning with quantum-enhanced feature spaces", Nature 2019) family of feature maps used
# mainly for quantum kernel methods. All of pauli_feature_map / z_feature_map /
# zz_feature_map / iqp_feature_map are thin, differently-parametrized calls into the same
# layer-builder here.
# -------------------------------------------------------------------------------------------

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

