from typing import Optional, Sequence, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.gatelibrary import CX
from AriaQuanta.qml.encoding.validation import validate_rotation_names, validate_features, TRAINABLE_ROTATION_GATES, ROTATION_GATES


# -------------------------------------------------------------------------------------------
class DataReUploadingAnsatz(Ansatz):
    """
    Data re-uploading circuit (Perez-Salinas et al., "Data re-uploading for a
    universal quantum classifier", 2020): re-uploads the *same* classical feature vector
    once per layer, each upload immediately followed by a trainable rotation layer and a
    ring of entangling CX gates. Re-exposing the data to the circuit repeatedly lets even a
    single-qubit-per-feature circuit realize a much richer function of the data than one
    encoding layer alone could (a single angle-encoding layer can only ever produce a
    function that is linear in each feature's rotation angle; interleaving trainable layers
    breaks that limitation).

    The data enters as *literal* (non-trainable) gate angles -- it is never registered as
    one of self.params_names -- so a fresh instance is needed per data sample. Only the
    trainable weights (self.params_names) are meant to be optimized/rebound, via
    set_params_values(), exactly like any other AriaQuanta Ansatz. This is also exactly the
    shape a future VQE/VQC-style training loop over a QML model will expect: build one
    DataReUploadingAnsatz per sample (or re-use one instance and only rebuild the
    data-dependent gates -- a possible future optimization once profiling asks for it),
    bind weights with set_params_values(), then run/measure.

    :param data: Classical feature vector, re-uploaded (identically) at every layer.
    :param n_layers: Number of re-uploading layers.
    :param encoding_rotation: Rotation gate name (or list, for dense encoding) used for the
                               fixed data-upload layers. One of 'RX', 'RY', 'RZ', 'P'.
    :param trainable_rotation: Rotation gate name (or list) used for the trainable weight
                                layers. One of 'RX', 'RY', 'RZ' (P has no trainable form).
    :param entangle: If True, a ring of CX gates follows each trainable layer.
    :param num_of_qubits: Number of qubits. Defaults to the minimum needed for len(data)
                           with encoding_rotation's features-per-qubit.
    :param scale: Multiplies every feature before it becomes an encoding angle.
    """

    def __init__(self, data, n_layers: int=1,
                 encoding_rotation: Union[str, Sequence[str]]='RY',
                 trainable_rotation: Union[str, Sequence[str]]='RY',
                 entangle: bool=True, num_of_qubits: Optional[int]=None, scale: float=1.0, start_index: int=0) -> None:

        if n_layers < 1:
            raise ValueError("'n_layers' must be at least 1, got {}.".format(n_layers))

        encoding_rotations  = validate_rotation_names(encoding_rotation)
        trainable_rotations = validate_rotation_names(trainable_rotation, trainable=True)
        features            = validate_features(data)

        features_per_qubit = len(encoding_rotations)
        n_needed = int(np.ceil(features.size / features_per_qubit))
        n = num_of_qubits if num_of_qubits is not None else n_needed
        if n < n_needed:
            raise ValueError(
                "{} feature(s) with {} encoding rotation(s)/qubit need at least {} qubit(s); got num_of_qubits={}."
                .format(features.size, features_per_qubit, n_needed, n)
            )

        params_names = [
            'theta_l{}_q{}_{}'.format(layer, q, name)
            for layer in range(n_layers) for q in range(n) for name in trainable_rotations
        ]
        super().__init__(n, params_names)

        self.data = features
        self.n_layers = n_layers
        self.encoding_rotations = encoding_rotations
        self.trainable_rotations = trainable_rotations

        param_iter = iter(params_names)
        for layer in range(n_layers):
            flag = False
            idx = start_index
            for q in range(n):                  # fixed data upload
                for name in encoding_rotations:
                    if idx >= len(features): flag = True; break
                    gate_cls = ROTATION_GATES[name]
                    self.add_gate(gate_cls(float(scale * features[idx]), q))
                    idx += 1
                if flag: break

            for q in range(n):                  # trainable layer
                for name in trainable_rotations:
                    gate_cls = TRAINABLE_ROTATION_GATES[name]
                    self.add_gate(gate_cls(next(param_iter), q))

            if entangle and n > 1:              # ring entangler
                for q in range(n):
                    self.add_gate(CX(q, (q + 1) % n))

