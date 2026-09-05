from typing import Optional, Sequence, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.qml._shared import validate_rotation_names, validate_features, ROTATION_GATES


# -------------------------------------------------------------------------------------------
def angle_encoding(data, num_of_qubits: Optional[int] = None, rotation: Union[str, Sequence[str]]='RY',
                   scale: float=1.0, start_index: int=0) -> Circuit:
    """
    Angle (rotation / phase) encoding: encodes each classical feature x_i as a rotation
    angle scale * x_i on qubit i, via a single-qubit rotation gate. This is also the
    single-qubit special case of Hamiltonian/time-evolution encoding: RX/RY/RZ(theta) is
    exactly exp(-i * theta/2 * P) for P in {X, Y, Z} -- i.e. "evolve qubit i under Pauli P
    for a data-dependent time".

    Passing more than one gate name in `rotation` (e.g. ['RY', 'RZ']) gives *dense* angle
    encoding: qubit i receives one rotation gate per name, consuming one feature per
    (qubit, gate) pair -- packing len(rotation) features per qubit instead of one.

    :param data: Sequence of classical feature values.
    :param num_of_qubits: Number of qubits. Defaults to ceil(len(data) / len(rotation)).
    :param rotation: Rotation gate name, or a list of names for dense encoding.
                      One of 'RX', 'RY', 'RZ', 'P' (see AriaQuanta.qml.encoding.base).
    :param scale: Multiplies every feature before it becomes a rotation angle
                  (e.g. scale=math.pi for features already normalized to [-1, 1]).
    :return: A new Circuit with the angle-encoding rotations applied (not yet run).
    """
    rotations = validate_rotation_names(rotation)
    features  = validate_features(data)

    features_per_qubit = len(rotations)
    n_needed = int(np.ceil(features.size / features_per_qubit))
    n = num_of_qubits if num_of_qubits is not None else n_needed
    if n < n_needed:
        raise ValueError(
            "{} feature(s) with {} rotation(s)/qubit need at least {} qubit(s); got num_of_qubits={}."
            .format(features.size, features_per_qubit, n_needed, n)
        )

    qc = Circuit(n)

    idx = start_index
    for q in range(n):
        for name in rotations:
            if idx >= len(features):
                return qc
            gate_cls = ROTATION_GATES[name]
            qc | gate_cls(float(scale * features[idx]), q)
            idx += 1
    return qc
