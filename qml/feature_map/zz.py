from typing import Optional, Union, Sequence

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.qml.feature_map.pauli import pauli_feature_map
from AriaQuanta.qml.feature_map.unit import default_data_map


# -------------------------------------------------------------------------------------------
def z_feature_map(data, reps: int=2, num_of_qubits: Optional[int]=None, data_map=default_data_map) -> Circuit:
    """
    Z feature map: pauli_feature_map with only single-qubit Z blocks (no entangling terms).
    Equivalent to Qiskit's ZFeatureMap. Every qubit stays an independent product state
    throughout, so this is efficiently simulable classically -- mainly useful as a
    baseline/ablation against zz_feature_map / pauli_feature_map, which add entanglement.
    """
    return pauli_feature_map(data, paulis=('Z',), reps=reps, num_of_qubits=num_of_qubits, data_map=data_map)

# -------------------------------------------------------------------------------------------
def zz_feature_map(data, reps: int=2, entanglement: Union[str, Sequence]='full',
                    num_of_qubits: Optional[int]=None, data_map=default_data_map) -> Circuit:
    """
    ZZ feature map (Havlicek et al.'s main construction; matches Qiskit's ZZFeatureMap):
    pauli_feature_map with single-Z and pairwise-ZZ blocks. The most commonly used feature
    map for quantum kernel methods -- the entangling ZZ terms are what makes the induced
    kernel conjectured to be classically hard to estimate, the source of a potential
    quantum advantage for kernel-based classification/regression.
    """
    return pauli_feature_map(data, paulis=('Z', 'ZZ'), reps=reps, entanglement=entanglement, 
                             num_of_qubits=num_of_qubits, data_map=data_map)
