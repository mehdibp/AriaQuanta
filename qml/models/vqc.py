from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.algorithms.eigen_solver import Hamiltonian, find_expectation_value
from AriaQuanta.qml.gradients.parameter_shift import parameter_shift_gradient
from AriaQuanta.qml.training.loss import Loss
from AriaQuanta.qml.training.optimizer import Optimizer


# -------------------------------------------------------------------------------------------
# Variational Quantum Classifier/Regressor -- the piece that ties every earlier qml.*
# module together into something end-to-end trainable:
#
#   feature_map(x)  -- AriaQuanta.qml.encoding / qml.feature_map: embeds one classical
#                       sample as fixed state-preparation gates (not trained)
#   + ansatz         -- AriaQuanta.qml.ansatz / aqc.ansatz: the trainable circuit
#   -> observable    -- AriaQuanta.algorithms.eigen_solver.Hamiltonian, measured via
#                       find_expectation_value to get a raw scalar prediction
#   -> output_map    -- rescales the raw expectation value onto the scale a given Loss
#                       expects (identity for MSE/MAE/Hinge; a probability map for the
#                       cross-entropy losses)
#   -> loss          -- AriaQuanta.qml.training.loss.Loss
#   -> optimizer     -- AriaQuanta.qml.training.optimizer.Optimizer, stepped using the
#                       CHAIN-RULE gradient below.
#
# The gradient VQC.fit() uses, for one sample (x, y_true), is:
#
#   d(raw)/d(theta)    = parameter_shift_gradient(ansatz, raw_prediction_fn)   -- [1]
#   d(output)/d(raw)   = output_map.derivative(raw)                            -- [2]
#   d(loss)/d(output)  = loss.gradient(output, y_true)                         -- [3]
#   d(loss)/d(theta)   = [3] * [2] * [1]                          (chain rule, elementwise)
#
# [1] is exact via parameter-shift *only* because raw_prediction_fn returns the raw
# observable expectation value (sinusoidal in each ansatz parameter) -- never the loss
# value itself. Composing evaluate_fn = loss(prediction(theta)) and parameter-shifting
# *that* directly is mathematically wrong, not merely imprecise: it silently returns a
# different, unrelated quantity (see AriaQuanta.qml.gradients.parameter_shift's docstring
# for the analytic proof). This is exactly why every Loss exposes gradient() as a
# first-class method, and why this file -- not qml.gradients or qml.training -- is where
# that composition is allowed to happen.
# -------------------------------------------------------------------------------------------


class OutputMap(ABC):
    """
    Base class for rescaling a raw observable expectation value onto the scale a Loss
    expects. Always a simple, differentiable, elementwise classical function -- forward()
    and derivative() must agree (derivative() is exactly d(forward(raw))/d(raw)), since
    VQC.fit() multiplies by derivative() as part of the chain rule above.
    """
    name: str = ''

    @abstractmethod
    def forward(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityOutputMap(OutputMap):
    """Passes the raw expectation value through unchanged -- the right choice for MSE/MAE/
    Hinge, which all operate directly on a raw <Z>-style (or any Hamiltonian) expectation."""
    name = 'identity'

    def forward(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, dtype=float)

    def derivative(self, raw: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(raw, dtype=float))


class ZToProbabilityOutputMap(OutputMap):
    """
    Maps a <Z>-style expectation value in [-1, 1] to a probability in [0, 1] via
    p = (1 + z) / 2 -- the natural affine map for pairing a single-Pauli-Z observable
    readout with BinaryCrossEntropyLoss (which requires an actual probability, not a raw
    expectation value).
    """
    name = 'z_to_probability'

    def forward(self, raw: np.ndarray) -> np.ndarray:
        return (1.0 + np.asarray(raw, dtype=float)) / 2.0

    def derivative(self, raw: np.ndarray) -> np.ndarray:
        return 0.5 * np.ones_like(np.asarray(raw, dtype=float))


# -------------------------------------------------------------------------------------------
class VQC:
    """
    Variational Quantum Classifier/Regressor.

    :param feature_map: Callable(x) -> Circuit embedding one classical sample x as fixed
                    state-preparation gates, e.g. `lambda x: angle_encoding(x, num_of_qubits=n)`
                    or `lambda x: zz_feature_map(x, reps=2)`. Wrap whichever
                    AriaQuanta.qml.encoding / qml.feature_map function with whatever kwargs
                    are needed -- VQC only ever calls it as feature_map(x), with x a single
                    sample (one row of X).
    :param ansatz: The trainable circuit (an AriaQuanta.aqc.ansatz.Ansatz, e.g.
                    HardwareEfficientAnsatz), on the same number of qubits as feature_map(x).
    :param observable: Hamiltonian whose expectation value is the model's raw prediction,
                    e.g. Hamiltonian([('Z0', 1.0)]) for a single-qubit readout.
    :param loss: A Loss instance. Must implement gradient() to be used with fit() --
                    predict()/evaluate() work with any Loss.
    :param optimizer: An Optimizer instance driving the parameter updates.
    :param num_of_iter_measure: Measurement shots per expectation-value estimate.
    :param output_map: Rescales the raw observable expectation onto the loss's expected
                    scale. Defaults to IdentityOutputMap; use ZToProbabilityOutputMap with
                    a cross-entropy loss.
    :param initial_params: Starting parameter values for the ansatz. Defaults to a
                    uniform-random draw in [-pi, pi) (seeded by random_state) -- an
                    ansatz's gates are symbolic (unbound) until set_params_values() is
                    called at least once, so VQC always binds something at construction.
    :param random_state: Seeds the default random initial_params draw. Ignored if
                    initial_params is given explicitly.
    """

    def __init__(self, feature_map: Callable[[np.ndarray], Circuit], ansatz: Ansatz,
                 observable: Hamiltonian, loss: Loss, optimizer: Optimizer,
                 num_of_iter_measure: int = 200, output_map: Optional[OutputMap] = None,
                 initial_params: Optional[Union[List[float], np.ndarray]] = None,
                 random_state: Optional[int] = None) -> None:

        self._check_init(feature_map, ansatz, observable, loss, optimizer, num_of_iter_measure, output_map)

        self.feature_map = feature_map
        self.ansatz = ansatz
        self.observable = observable
        self.loss = loss
        self.optimizer = optimizer
        self.num_of_iter_measure = num_of_iter_measure
        self.output_map = output_map if output_map is not None else IdentityOutputMap()

        if initial_params is None:
            rng = np.random.default_rng(random_state)
            initial_params = rng.uniform(-np.pi, np.pi, size=len(ansatz.params_names))
        self.ansatz.set_params_values(initial_params)

        self.history: Dict[str, List[float]] = {'loss': []}

    # ------------------------------------------------------------
    def _build_circuit_for(self, x: np.ndarray, ansatz: Ansatz) -> Circuit:
        encoding_circuit = self.feature_map(x)
        if not isinstance(encoding_circuit, Circuit):
            raise TypeError("'feature_map' must return a Circuit, got {}.".format(type(encoding_circuit).__name__))
        if encoding_circuit.num_of_qubits != ansatz.num_of_qubits:
            raise ValueError(
                "feature_map(x) produced a {}-qubit circuit but the ansatz has {} qubits -- "
                "they must act on the same number of qubits.".format(encoding_circuit.num_of_qubits, ansatz.num_of_qubits)
            )
        circuit = Circuit(ansatz.num_of_qubits)
        circuit.gates = list(encoding_circuit.gates) + list(ansatz.gates)
        return circuit

    def _raw_expectation(self, x: np.ndarray, ansatz: Optional[Ansatz] = None) -> float:
        ansatz = ansatz if ansatz is not None else self.ansatz
        circuit = self._build_circuit_for(x, ansatz)
        _, energy = find_expectation_value(circuit, self.observable, self.num_of_iter_measure)
        return energy

    # ------------------------------------------------------------
    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw observable expectation value(s), before output_map. Shape (n_samples,)."""
        X = self._validate_X(X)
        return np.array([self._raw_expectation(x) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Model output(s) after output_map (e.g. a probability, with ZToProbabilityOutputMap). Shape (n_samples,)."""
        return self.output_map.forward(self.predict_raw(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean self.loss over (X, y). No training, no gradient computation."""
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=float).flatten()
        if y_pred.shape != y_true.shape:
            raise ValueError("'X' and 'y' must have the same number of samples, got {} and {}.".format(y_pred.shape[0], y_true.shape[0]))
        return self.loss(y_pred, y_true)

    # ------------------------------------------------------------
    def _gradient_for_sample(self, x: np.ndarray, y_true: float) -> Tuple[np.ndarray, float]:
        """The chain-rule gradient of self.loss for one sample (see the module docstring). Returns (gradient, loss_value)."""

        def raw_prediction_fn(a: Ansatz) -> float:
            return self._raw_expectation(x, ansatz=a)

        raw_pred = raw_prediction_fn(self.ansatz)
        d_raw_d_theta = parameter_shift_gradient(self.ansatz, raw_prediction_fn)

        output = self.output_map.forward(np.array([raw_pred]))
        d_output_d_raw = self.output_map.derivative(np.array([raw_pred]))[0]

        try:
            d_loss_d_output = self.loss.gradient(output, np.array([y_true]))[0]
        except NotImplementedError:
            raise NotImplementedError(
                "{} does not implement an analytic gradient(), so it cannot be used to train "
                "a VQC -- differentiating loss(prediction) directly via parameter-shift is "
                "mathematically incorrect (see AriaQuanta.qml.gradients.parameter_shift_"
                "gradient's docstring). Choose a loss that implements gradient(), or add one "
                "following the same pattern.".format(type(self.loss).__name__)
            )

        loss_value = float(self.loss(output, np.array([y_true])))
        gradient = d_loss_d_output * d_output_d_raw * d_raw_d_theta
        return gradient, loss_value

    # ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: Optional[int] = None,
            shuffle: bool = True, verbose: bool = False,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            random_state: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Trains the ansatz's parameters against (X, y) using self.loss/self.optimizer.

        Cost note: each gradient step re-evaluates the circuit O(batch_size * n_params)
        times (2 shifted evaluations per parameter per sample, via parameter-shift), each
        costing num_of_iter_measure shots -- keep the dataset/ansatz size modest, or swap
        in AriaQuanta.qml.gradients.spsa_gradient (2 evaluations total, regardless of
        n_params) paired with AriaQuanta.qml.training.optimizer.SPSA once that becomes the
        bottleneck (see this file's own "candidates for later" for how that swap would
        change _gradient_for_sample).

        :param X: (n_samples, n_features) array-like.
        :param y: (n_samples,) array-like of targets.
        :param epochs: Number of passes over the full dataset.
        :param batch_size: Samples per gradient step (the gradient is averaged across the
                    batch). Defaults to the full dataset (one step per epoch).
        :param shuffle: Reshuffle sample order every epoch.
        :param verbose: Print the epoch loss (and val_loss, if validation_data is given).
        :param validation_data: Optional (X_val, y_val), evaluated (no training) once per
                    epoch and recorded under history['val_loss'].
        :param random_state: Seeds the epoch shuffling, for reproducibility.
        :return: History dict: {'loss': [...]} (plus 'val_loss' if validation_data given).
                    Also stored on self.history. Calling fit() again continues training
                    from the current ansatz parameters and optimizer state (a warm start),
                    it does not reset either.
        """
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float).flatten()
        if X.shape[0] != y.shape[0]:
            raise ValueError("'X' and 'y' must have the same number of samples, got {} and {}.".format(X.shape[0], y.shape[0]))
        n_samples = X.shape[0]

        bs = batch_size if batch_size is not None else n_samples
        if not (isinstance(bs, int) and not isinstance(bs, bool) and 1 <= bs <= n_samples):
            raise ValueError("'batch_size' must be an int in [1, {}] (the dataset size), got {}.".format(n_samples, batch_size))
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError("'epochs' must be a positive int, got {}.".format(epochs))

        rng = np.random.default_rng(random_state)
        indices = np.arange(n_samples)

        history: Dict[str, List[float]] = {'loss': []}
        if validation_data is not None:
            history['val_loss'] = []

        for epoch in range(epochs):
            if shuffle:
                rng.shuffle(indices)

            epoch_losses: List[float] = []
            for start in range(0, n_samples, bs):
                batch_idx = indices[start:start + bs]
                grads, losses = [], []
                for i in batch_idx:
                    g, l = self._gradient_for_sample(X[i], y[i])
                    grads.append(g)
                    losses.append(l)

                batch_grad = np.mean(grads, axis=0)
                new_params = self.optimizer.step(self.ansatz.params_values, batch_grad)
                self.ansatz.set_params_values(new_params)
                epoch_losses.extend(losses)

            epoch_loss = float(np.mean(epoch_losses))
            history['loss'].append(epoch_loss)

            log_line = "epoch {}/{} - loss: {:.6f}".format(epoch + 1, epochs, epoch_loss)
            if validation_data is not None:
                val_loss = self.evaluate(*validation_data)
                history['val_loss'].append(val_loss)
                log_line += " - val_loss: {:.6f}".format(val_loss)
            if verbose:
                print(log_line)

        self.history = history
        return history

    # ------------------------------------------------------------
    @staticmethod
    def _validate_X(X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                "'X' must be 2-D with shape (n_samples, n_features), got shape {}. "
                "For a single sample, wrap it as [x] or x.reshape(1, -1); for a single "
                "feature, reshape as (n_samples, 1).".format(X.shape)
            )
        return X

    @staticmethod
    def _check_init(feature_map, ansatz, observable, loss, optimizer, num_of_iter_measure, output_map) -> None:
        if not callable(feature_map):
            raise TypeError("'feature_map' must be callable, got {}.".format(type(feature_map).__name__))
        if not isinstance(ansatz, Ansatz):
            raise TypeError("'ansatz' must be an Ansatz instance, got {}.".format(type(ansatz).__name__))
        if not isinstance(observable, Hamiltonian):
            raise TypeError("'observable' must be a Hamiltonian instance, got {}.".format(type(observable).__name__))
        if not isinstance(loss, Loss):
            raise TypeError("'loss' must be a Loss instance, got {}.".format(type(loss).__name__))
        if not isinstance(optimizer, Optimizer):
            raise TypeError("'optimizer' must be an Optimizer instance, got {}.".format(type(optimizer).__name__))
        if not isinstance(num_of_iter_measure, int) or isinstance(num_of_iter_measure, bool):
            raise TypeError("'num_of_iter_measure' must be an int, got {}.".format(type(num_of_iter_measure).__name__))
        if num_of_iter_measure < 1:
            raise ValueError("'num_of_iter_measure' must be at least 1, got {}.".format(num_of_iter_measure))
        if output_map is not None and not isinstance(output_map, OutputMap):
            raise TypeError("'output_map' must be an OutputMap instance (or None), got {}.".format(type(output_map).__name__))


# -------------------------------------------------------------------------------------------
# Candidates for later (none needed by anything shipped yet -- add when something actually
# needs them):
#   - Multi-observable / multi-class output: accept a list of Hamiltonians (one raw
#     prediction per class) plus a softmax-style OutputMap, extending the gradient
#     computation to a per-observable Jacobian (call parameter_shift_gradient once per
#     observable) -- CategoricalCrossEntropyLoss already supports the resulting shape.
#   - SPSA-based fit(): AriaQuanta.qml.gradients.spsa_gradient and
#     AriaQuanta.qml.training.optimizer.SPSA now both exist. Wiring them in means
#     _gradient_for_sample would call spsa_gradient directly on a full loss(prediction)
#     closure instead of parameter_shift_gradient on the raw prediction -- since
#     spsa_gradient doesn't need the chain-rule composition (it has no sinusoidal-structure
#     requirement, see its docstring), this actually *removes* the output_map.derivative()/
#     loss.gradient() steps for that path rather than just swapping one call. Likely shape:
#     a `gradient_method='parameter_shift'|'spsa'` constructor flag selecting which
#     _gradient_for_sample body runs; not added yet since fit() only has the one path today.
#   - partial_fit(): fit() already supports calling it more than once on the same VQC
#     instance (ansatz params and optimizer state both persist between calls, a warm
#     start) -- a partial_fit() alias is only a naming/ergonomics addition.
#   - Sampling-based (not observable-based) multi-class readout: read a full measured
#     bitstring distribution via Result.count() instead of a Hamiltonian expectation
#     value -- a fundamentally different readout mechanism from the one here, worth its
#     own OutputMap-like abstraction rather than bolting onto VQC as written.
#   - Quantum kernel methods / QSVM: a different model family entirely (a classical SVM
#     over a quantum-computed kernel matrix, from qml.feature_map's state overlaps) --
#     would be its own class here, not a VQC variant.
# -------------------------------------------------------------------------------------------
