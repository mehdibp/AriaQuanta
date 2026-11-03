from typing import Optional, Sequence, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.gatelibrary import CX, Barrier
from AriaQuanta.qml._shared import validate_rotation_names, validate_features, TRAINABLE_ROTATION_GATES, ROTATION_GATES


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

    Qiskit-style parameter split: the data enters through *symbolic input parameters*
    ('x0', 'x1', ... = self.input_names), separate from the trainable weights
    (self.params_names). The circuit is built ONCE; a sample is bound with
    set_input_values(x) and the weights with set_params_values(), exactly like Qiskit's
    input_params / weight_params. Inputs are never part of params_names, so
    parameter_shift_gradient / spsa_gradient only differentiate w.r.t. the weights.

    Two ways to use it:
        - with VQC:        ansatz = DataReUploadingAnsatz(n_features=4, n_layers=3)
                            VQC(feature_map=None, ansatz=ansatz, ...)   # VQC binds each sample itself
        - standalone:      DataReUploadingAnsatz(data=x, n_layers=3)   # inputs bound immediately,
                            rebind later with set_input_values(x_new)   # (the old behaviour)

    :param data: Optional classical feature vector. If given, it is bound immediately
                    (and fixes n_features). If None, n_features is required.
    :param n_layers: Number of re-uploading layers.
    :param encoding_rotation: Rotation gate name (or list, for dense encoding) used for the
                               fixed data-upload layers. One of 'RX', 'RY', 'RZ', 'P'.
    :param trainable_rotation: Rotation gate name (or list) used for the trainable weight
                                layers. One of 'RX', 'RY', 'RZ' (P has no trainable form).
    :param entangle: If True, a ring of CX gates follows each trainable layer.
    :param num_of_qubits: Number of qubits. Defaults to the minimum needed for n_features
                            with encoding_rotation's features-per-qubit.
    :param scale: Multiplies every feature before it becomes an encoding angle (applied inside set_input_values).
    :param n_features: Length of the input vector (required when data is None).
    """
    def __init__(self, data=None, n_layers: int=1,
                 encoding_rotation: Union[str, Sequence[str]]='RY',
                 trainable_rotation: Union[str, Sequence[str]]='RY',
                 entangle: bool=True, num_of_qubits: Optional[int]=None, scale: float=1.0, 
                 start_index: int=0, n_features: Optional[int]=None) -> None:

        features, n_features = self._check_validation(data, n_layers, n_features)

        encoding_rotations  = validate_rotation_names(encoding_rotation)
        trainable_rotations = validate_rotation_names(trainable_rotation, trainable=True)

        features_per_qubit = len(encoding_rotations)
        n_needed = int(np.ceil(n_features / features_per_qubit))
        n = num_of_qubits if num_of_qubits is not None else n_needed
        if n < n_needed:
            raise ValueError(
                "{} feature(s) with {} encoding rotation(s)/qubit need at least {} qubit(s); got num_of_qubits={}."
                .format(n_features, features_per_qubit, n_needed, n)
            )

        params_names = [
            'theta_l{}_q{}_{}'.format(layer, q, name)
            for layer in range(n_layers) for q in range(n) for name in trainable_rotations
        ]
        input_names = ['x{}'.format(i) for i in range(n_features)]
        self.scale = scale                                  # must exist before set_input_values can run
        super().__init__(n, params_names, input_names=input_names)

        self.data = features
        self.n_features = n_features
        self.n_layers = n_layers
        self.encoding_rotations = encoding_rotations
        self.trainable_rotations = trainable_rotations

        param_iter = iter(params_names)
        for layer in range(n_layers):
            flag = False
            idx = start_index
            for q in range(n):                  # data upload (symbolic inputs)
                for name in encoding_rotations:
                    if idx >= n_features: flag = True; break
                    gate_cls = ROTATION_GATES[name]
                    self.add_gate(gate_cls(input_names[idx], q))
                    idx += 1
                if flag: break

            for q in range(n):                  # trainable layer
                for name in trainable_rotations:
                    gate_cls = TRAINABLE_ROTATION_GATES[name]
                    self.add_gate(gate_cls(next(param_iter), q))

            if entangle and n > 1:              # ring entangler
                for q in range(n):
                    self.add_gate(CX(q, (q + 1) % n))
            self.add_gate(Barrier())

        if features is not None:
            self.set_input_values(features)


    # ------------------------------------------------------------
    def set_input_values(self, input_values) -> None:
        values = np.asarray(input_values, dtype=float).flatten()
        super().set_input_values(self.scale * values)


    # ------------------------------------------------------------
    @staticmethod
    def _check_validation(data, n_layers: int, n_features: Optional[int]):
        if n_layers < 1:
            raise ValueError("'n_layers' must be at least 1, got {}.".format(n_layers))

        if data is not None:
            features = validate_features(data)
            if n_features is not None and n_features != features.size:
                raise ValueError("'n_features' ({}) does not match len(data) ({}).".format(n_features, features.size))
            n_features = features.size
        else:
            features = None
            if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
                raise ValueError("Give either 'data' or a positive int 'n_features'.")

        return features, n_features
