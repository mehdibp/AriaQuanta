from typing import Optional, Sequence, Union

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.qml.encoding.validation import validate_features
from AriaQuanta.qml.feature_map.unit import validate_pauli_blocks, apply_pauli_feature_layer, default_data_map


# -------------------------------------------------------------------------------------------
def pauli_feature_map(data, paulis: Union[str, Sequence[str]]=('Z', 'ZZ'), reps: int=2,
                       entanglement: Union[str, Sequence]='full', num_of_qubits: Optional[int]=None,
                       data_map=default_data_map) -> Circuit:
    """
    General Pauli feature map (Havlicek et al., "Supervised learning with quantum-enhanced
    feature spaces", Nature 2019): one classical feature per qubit, encoded through `reps`
    repetitions of [Hadamard layer] -> [exp(-i * phi_S(x) * P_S) for every Pauli block in
    `paulis` and every matching qubit subset].

    This is used almost exclusively as a *quantum kernel* feature map: the interesting
    quantity is the overlap |<Phi(x)|Phi(y)>|^2 between two encoded states, not the state
    |Phi(x)> in isolation. That's also why it starts every repetition with Hadamards --
    unlike angle_encoding, a lone data-dependent rotation on |0> here would just be a global
    phase with nothing observable.

    :param data: One feature per qubit.
    :param paulis: Pauli block string, or list of block strings over {X, Y, Z}, e.g.
                    ('Z', 'ZZ') (the default, matching Qiskit's ZZFeatureMap) or
                    ('Z', 'ZZ', 'ZZZ'). A block of length 1 applies to every qubit; a block
                    of length k > 1 applies to every qubit subset selected by `entanglement`.
    :param reps: Number of repetitions of the Hadamard + evolution layer.
    :param entanglement: 'full' (every combination of the block's qubits), 'linear'
                          (consecutive windows only), or an explicit list of qubit tuples.
                          Ignored for length-1 blocks (always applied to every qubit).
    :param num_of_qubits: Number of qubits. Defaults to len(data) (one feature per qubit;
                           unlike angle_encoding, feature maps don't pack multiple features
                           per qubit).
    :param data_map: The phi_S(x) function. Defaults to the paper's phi_i(x) = x_i /
                      phi_S(x) = prod_{i in S}(pi - x_i). Signature: (features, qubit_tuple) -> float.
    :return: A new Circuit with the feature map applied (not yet run).
    """
    if reps < 1:
        raise ValueError("'reps' must be at least 1, got {}.".format(reps))

    blocks = validate_pauli_blocks(paulis)
    features = validate_features(data)
    n = num_of_qubits if num_of_qubits is not None else features.size
    if features.size != n:
        raise ValueError("'data' must have exactly one feature per qubit ({}), got {}.".format(n, features.size))

    qc = Circuit(n)
    for _ in range(reps):
        apply_pauli_feature_layer(qc, features, blocks, entanglement=entanglement, data_map=data_map)
    return qc
