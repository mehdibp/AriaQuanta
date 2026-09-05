from typing import Sequence, Tuple, Union

from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.gatelibrary import CX, CZ
from AriaQuanta.qml._shared import TRAINABLE_ROTATION_GATES, validate_rotation_names, resolve_qubit_subsets


_ENTANGLER_GATES = {'CX': CX, 'CZ': CZ}


# -------------------------------------------------------------------------------------------
class HardwareEfficientAnsatz(Ansatz):
    """
    Hardware-efficient ansatz (a.k.a. "TwoLocal"): `reps` repetitions of a trainable
    rotation layer (one or more single-qubit rotation gates per qubit) followed by a fixed
    entangling layer (a chosen two-qubit gate over a chosen connectivity pattern), plus one
    final rotation layer with no trailing entangler.

    This is the same template AriaQuanta.aqc.ansatz.EfficientSU2Ansatz already implements
    (RY+RZ rotations, CX entanglers, linear connectivity) -- generalized here to a
    configurable rotation-gate set, entangler gate, and connectivity pattern, so e.g.
    rotation='RY' alone reproduces Qiskit's RealAmplitudes ansatz, and
    rotation=('RY','RZ') with the defaults reproduces EfficientSU2.

    :param num_of_qubits: Number of qubits.
    :param rotation: Rotation gate name(s) applied every layer, in order, e.g. 'RY' or
                      ('RY', 'RZ'). One of 'RX', 'RY', 'RZ'.
    :param entangler: Two-qubit gate used to entangle each pair. One of 'CX', 'CZ'.
    :param entanglement: 'linear' (neighboring pairs), 'circular' (neighboring pairs plus a
                          wraparound pair), 'full' (every pair), or an explicit list of
                          qubit-index pairs.
    :param reps: Number of rotation+entangler repetitions (there are reps+1 rotation layers
                 total: one before each of the `reps` entangler layers, plus a final one).
    """

    def __init__(self, num_of_qubits: int, rotation: Union[str, Sequence[str]]=('RY', 'RZ'),
                 entangler: str='CX', entanglement: Union[str, Sequence[Tuple[int, int]]]='linear', reps: int=3) -> None:

        self._check_validation(num_of_qubits, entangler, reps)

        rotations = validate_rotation_names(rotation, trainable=True)
        pairs     = resolve_qubit_subsets(2, num_of_qubits, entanglement)
        entangler_cls = _ENTANGLER_GATES[entangler]

        num_of_layers = reps + 1
        params_names = [
            'theta_l{}_q{}_{}'.format(layer, q, name)
            for layer in range(num_of_layers) for q in range(num_of_qubits) for name in rotations
        ]
        super().__init__(num_of_qubits, params_names)

        self.rotations = rotations
        self.entangler = entangler
        self.pairs = pairs

        param_iter = iter(params_names)
        for layer in range(num_of_layers):
            for q in range(num_of_qubits):
                for name in rotations:
                    gate_cls = TRAINABLE_ROTATION_GATES[name]
                    self.add_gate(gate_cls(next(param_iter), q))

            if layer < reps:
                for i, j in pairs:
                    self.add_gate(entangler_cls(i, j))

    # ------------------------------------------------------------
    @staticmethod
    def _check_validation(num_of_qubits, entangler, reps):
        if num_of_qubits < 1:
            raise ValueError("'num_of_qubits' must be at least 1, got {}.".format(num_of_qubits))
        if reps < 0:
            raise ValueError("'reps' must be non-negative, got {}.".format(reps))
        if entangler not in _ENTANGLER_GATES:
            raise ValueError("entangler must be one of {}, got {!r}.".format(list(_ENTANGLER_GATES), entangler))
