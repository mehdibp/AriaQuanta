from typing import Optional, Sequence, Union

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.qml.feature_map.pauli import pauli_feature_map
from AriaQuanta.qml.feature_map.unit import default_data_map


# Instantaneous Quantum Polynomial ----------------------------------------------------------
def iqp_feature_map(data, degree: int=2, reps: int=1, entanglement: Union[str, Sequence]='full',
                     num_of_qubits: Optional[int]=None, data_map=default_data_map) -> Circuit:
    """
    IQP-style feature map: a Pauli feature map restricted to all-Z blocks up to `degree`
    qubits at once (single Z terms, plus ZZ, plus ZZZ, ... up to a 'Z'*degree block). Every
    gate is diagonal in the Z basis after the Hadamard layer, so all of them commute --
    literally an "Instantaneous Quantum Polynomial" circuit (Shepherd & Bremner), which is
    the construction Havlicek et al. built their feature map on top of.

    Note: 'IQP feature map' isn't a single standardized definition across libraries/papers
    -- e.g. PennyLane's IQPEmbedding uses a plain product phi_S(x) = prod(x_i) instead of
    the (pi - x_i)-based form used here. This implementation deliberately reuses the same
    pauli_feature_map/data_map convention as z_feature_map/zz_feature_map for consistency
    within AriaQuanta; pass a custom `data_map` if a different convention is needed.

    :param degree: Highest Pauli-block order to include (2 -> Z and ZZ, same set as
                   zz_feature_map; 3 -> also ZZZ; etc.).
    :param reps: Number of repetitions of the Hadamard + evolution layer.
    """
    if degree < 1:
        raise ValueError("'degree' must be at least 1, got {}.".format(degree))

    paulis = tuple('Z' * k for k in range(1, degree+1))
    return pauli_feature_map(data, paulis=paulis, reps=reps, entanglement=entanglement,
                              num_of_qubits=num_of_qubits, data_map=data_map)

