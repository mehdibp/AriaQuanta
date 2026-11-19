from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.hamiltonian import Hamiltonian
from AriaQuanta.qml.gradients.spsa import spsa_gradient, spsa_perturbation_schedule
from AriaQuanta.qml.training.loss import Loss
from AriaQuanta.qml.training.optimizer import Optimizer
from AriaQuanta.qml.models.qnn import QNN
from AriaQuanta.qml.models.persistence import save_params, load_params, read_params
from AriaQuanta.qml._shared import validate_X, validate_y


# -------------------------------------------------------------------------------------------
# Variational Quantum Classifier/Regressor -- a QNN (qnn.py: feature_map + ansatz + observables
# -> raw expectation values and their Jacobian) plus everything needed to train it:
#
#   qnn              -- AriaQuanta.qml.models.qnn.QNN: raw prediction, a scalar or a vector of K
#                       expectation values, and d(raw)/d(theta)
#   -> output_map    -- maps the raw prediction onto the scale a given Loss expects (identity
#                       for MSE/MAE/Hinge; a probability map for binary cross-entropy;
#                       softmax for multi-class categorical cross-entropy)
#   -> loss          -- AriaQuanta.qml.training.loss.Loss
#   -> optimizer     -- AriaQuanta.qml.training.optimizer.Optimizer, stepped using the
#                       CHAIN-RULE gradient below.
#
# The gradient VQC.fit() uses, for one sample (x, y_true), with K outputs and P parameters:
#
#   D = d(raw)/d(theta)    = qnn.backward_sample(x)                                (K, P)   -- [1]
#   J = d(output)/d(raw)   = output_map.jacobian(raw)                              (K, K)   -- [2]
#   g = d(loss)/d(output)  = loss.gradient(output, y_true)                         (K,)     -- [3]
#   d(loss)/d(theta)       = g @ J @ D                                             (P,)     (chain rule)
#
# For a single observable K = 1 and everything collapses to the scalar chain rule (J is the 1x1
# matrix [output_map.derivative(raw)]). For an elementwise OutputMap J is diagonal; for softmax
# it is the full matrix  scale * (diag(s) - s s^T).
#
# [1] is exact via parameter-shift *only* because the QNN returns the raw observable expectation
# values (sinusoidal in each ansatz parameter) -- never the loss value itself. Composing
# loss(prediction(theta)) and parameter-shifting *that* directly is mathematically wrong, not
# merely imprecise: it silently returns a different, unrelated quantity (see
# AriaQuanta.qml.gradients.parameter_shift's docstring for the analytic proof). This is exactly
# why every Loss exposes gradient() as a first-class method, and why this file -- not
# qml.gradients, qml.training or the QNN -- is where that composition is allowed to happen.
#
# gradient_method='spsa' selects a second, structurally different path: spsa_gradient perturbs
# every parameter simultaneously and evaluates the *full* loss at theta +/- c_k*delta -- 2
# evaluations per sample regardless of parameter count (plus one more to report the loss).
# It does not rely on the sinusoidal structure of a raw expectation value, so there is no
# chain rule here: [1]-[3] above are not used at all, and a Loss without gradient() trains
# fine. The price is a noisy gradient estimate, which is why it is meant to be paired with the
# decaying-step-size AriaQuanta.qml.training.optimizer.SPSA.
# -------------------------------------------------------------------------------------------


class OutputMap(ABC):
    """
    Base class for mapping the raw observable expectation value(s) of ONE sample onto the scale
    a Loss expects. Always a simple, differentiable, classical function -- forward() and
    jacobian() must agree (jacobian() is exactly d(forward(raw))/d(raw)), since VQC.fit()
    multiplies by it as part of the chain rule above.

    Elementwise maps (identity, p = (1+z)/2, ...) only need forward() and derivative(); the
    default jacobian() is then diag(derivative(raw)). Maps that mix their inputs (softmax) override
    jacobian() instead and leave derivative() unimplemented.

    forward() acts along the LAST axis, so it also accepts a whole batch (n_samples, K); jacobian()
    takes the raw vector of one sample, shape (K,), and returns the (K, K) matrix J[i, j] = d out_i / d raw_j.
    """
    name: str = ''

    @abstractmethod
    def forward(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, raw: np.ndarray) -> np.ndarray:
        """Elementwise d(forward(raw))/d(raw). Only defined for elementwise maps."""
        raise NotImplementedError("{} is not an elementwise map -- use jacobian().".format(type(self).__name__))

    def jacobian(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float).reshape(-1)
        return np.diag(self.derivative(raw))

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

class SoftmaxOutputMap(OutputMap):
    """
    Softmax over the K observable expectation values of one sample:

        p_i = exp(scale * raw_i) / sum_j exp(scale * raw_j)

    The multi-class counterpart of ZToProbabilityOutputMap, meant for CategoricalCrossEntropyLoss
    (the K observables play the role of the K logits, as with EstimatorQNN + a softmax/
    cross-entropy head in Qiskit Machine Learning). Not elementwise, so it provides the full
    Jacobian  J[i, j] = d p_i / d raw_j = scale * p_i * (delta_ij - p_j).

    :param scale: Inverse temperature applied to the raw values before the softmax. Expectation
                  values of Pauli observables live in [-1, 1], so with scale = 1 the output can
                  never become confident (3 classes: the largest reachable probability is
                  e / (e + 2/e) ~ 0.79, a loss floor of ~0.24 even for a perfect model). A larger
                  scale (e.g. 3-5) removes that ceiling; scale = 1 is the plain softmax.
    """
    name = 'softmax'

    def __init__(self, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ValueError("'scale' must be positive, got {}.".format(scale))
        self.scale = float(scale)

    def forward(self, raw: np.ndarray) -> np.ndarray:
        z = self.scale * np.asarray(raw, dtype=float)
        z = z - np.max(z, axis=-1, keepdims=True)           # numerically stable softmax
        e = np.exp(z)
        return e / np.sum(e, axis=-1, keepdims=True)

    def jacobian(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if raw.ndim != 1:
            raise ValueError("jacobian() takes the raw vector of ONE sample (shape (K,)), got shape {}.".format(raw.shape))
        s = self.forward(raw)
        return self.scale * (np.diag(s) - np.outer(s, s))


# -------------------------------------------------------------------------------------------
class VQC:
    """
    Variational Quantum Classifier/Regressor: a QNN (see qnn.py) plus an output map, a loss and
    an optimizer. The QNN is built from feature_map / ansatz / observable / num_of_iter_measure /
    initial_params / random_state and is available as `self.qnn`; `self.ansatz`,
    `self.feature_map`, `self.observables`, ... are read-only shortcuts to it.

    :param feature_map: Callable(x) -> Circuit embedding one classical sample x as fixed
                    state-preparation gates, e.g. `lambda x: angle_encoding(x, num_of_qubits=n)`
                    or `lambda x: zz_feature_map(x, reps=2)`. Wrap whichever
                    AriaQuanta.qml.encoding / qml.feature_map function with whatever kwargs
                    are needed -- VQC only ever calls it as feature_map(x), with x a single
                    sample (one row of X). May be None for an ansatz carrying the data as
                    symbolic input parameters (e.g. DataReUploadingAnsatz).
    :param ansatz: The trainable circuit (an AriaQuanta.aqc.ansatz.Ansatz, e.g.
                    HardwareEfficientAnsatz), on the same number of qubits as feature_map(x).
    :param observable: Hamiltonian whose expectation value is the model's raw prediction,
                    e.g. Hamiltonian([('Z0', 1.0)]) for a single-qubit readout -- OR a list of
                    K Hamiltonians for a model with K outputs (one per class for a classifier,
                    one per target for a multi-output regressor). With a single Hamiltonian
                    predict()/predict_raw() return shape (n_samples,); with a list, (n_samples, K).
    :param loss: A Loss instance. Must implement gradient() to be used with fit() --
                    predict()/evaluate() work with any Loss.
    :param optimizer: An Optimizer instance driving the parameter updates.
    :param num_of_iter_measure: Measurement shots per expectation-value estimate.
    :param output_map: Maps the raw observable expectation(s) onto the loss's expected scale.
                    Defaults to IdentityOutputMap; use ZToProbabilityOutputMap with a binary
                    cross-entropy loss, SoftmaxOutputMap (K >= 2 observables) with
                    CategoricalCrossEntropyLoss.
    :param initial_params: Starting parameter values for the ansatz. Defaults to a
                    uniform-random draw in [-pi, pi) (seeded by random_state).
    :param random_state: Seeds the default random initial_params draw. Ignored if
                    initial_params is given explicitly.
    :param gradient_method: How fit() obtains d(loss)/d(theta):
                    'parameter_shift' (default) -- exact chain rule: QNN Jacobian x output_map.jacobian()
                        x loss.gradient(). 2*n_params + 1 circuit evaluations per sample (independent
                        of K); the loss must implement gradient().
                    'spsa' -- SPSA estimate of the full loss's gradient. 3 evaluations per
                        sample regardless of n_params; works with any Loss (gradient() not
                        needed). Noisy by design: pair it with qml.training.SPSA (decaying
                        step size) and use batch_size > 1 so the batch average smooths it.
    :param spsa_c: Starting perturbation magnitude for gradient_method='spsa' (decays across
                    optimizer steps as spsa_c / (k + 1) ** spsa_gamma -- Spall's schedule, see
                    qml.gradients.spsa_perturbation_schedule). Ignored otherwise.
    :param spsa_gamma: Decay exponent of that schedule (Spall's recommendation: 0.101).
    """
    def __init__(self, feature_map: Optional[Callable[[np.ndarray], Circuit]], ansatz: Ansatz,
                 observable: Union[Hamiltonian, Sequence[Hamiltonian]], loss: Loss, optimizer: Optimizer,
                 num_of_iter_measure: int = 200, output_map: Optional[OutputMap] = None,
                 initial_params: Optional[Union[List[float], np.ndarray]] = None,
                 random_state: Optional[int] = None, gradient_method: str = 'parameter_shift',
                 spsa_c: float = 0.2, spsa_gamma: float = 0.101) -> None:

        self._check_init(loss, optimizer, output_map)
        self._check_gradient_options(gradient_method, spsa_c, spsa_gamma)

        self.qnn = QNN(feature_map, ansatz, observable, num_of_iter_measure=num_of_iter_measure,
                       initial_params=initial_params, random_state=random_state)
        self.loss = loss
        self.optimizer = optimizer
        self.output_map = output_map if output_map is not None else IdentityOutputMap()
        self.gradient_method = gradient_method
        self.spsa_c = spsa_c
        self.spsa_gamma = spsa_gamma
        self._optimizer_steps = 0      # optimizer updates taken so far (all fit() calls) -- indexes the SPSA c_k schedule

        if isinstance(self.output_map, SoftmaxOutputMap) and self.num_of_outputs < 2:
            raise ValueError("SoftmaxOutputMap needs at least 2 observables (one per class), got {}.".format(self.num_of_outputs))

        self.history: Dict[str, List[float]] = {'loss': []}


    # ------------------------------------------------------------
    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Raw observable expectation value(s), before output_map.
        Shape (n_samples,) for a single observable, (n_samples, K) for a list of K observables.
        """
        raw = self.qnn.forward(X)
        return raw if self.multi_output else raw[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Model output(s) after output_map (e.g. a probability, with ZToProbabilityOutputMap; the K class
        probabilities, with SoftmaxOutputMap). Shape (n_samples,) or (n_samples, K), like predict_raw().
        """
        return self.output_map.forward(self.predict_raw(X))

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predicted class index per sample = argmax over the K outputs. Multi-observable models only. Shape (n_samples,)."""
        if not self.multi_output:
            raise ValueError("predict_classes() needs a list of observables (one per class); this VQC has a single observable.")
        return np.argmax(self.predict(X), axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean self.loss over (X, y). No training, no gradient computation."""
        y_pred = self.predict(X)
        y_true = validate_y(y, self.num_of_outputs, self.multi_output)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("'X' and 'y' must have the same number of samples, got {} and {}.".format(y_pred.shape[0], y_true.shape[0]))
        return self.loss(y_pred, y_true)

    # ------------------------------------------------------------
    def _gradient_for_sample(self, x: np.ndarray, y_true: np.ndarray, spsa_c: Optional[float] = None,
                              spsa_seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """d(loss)/d(theta) for one sample, via self.gradient_method (see module docstring). Returns (gradient, loss_value)"""
        y_true = np.atleast_1d(np.asarray(y_true, dtype=float))      # one sample's target: (1,) for a scalar/label, (K,) for one-hot/multi-output
        if self.gradient_method == 'spsa':
            return self._gradient_spsa(x, y_true, spsa_c if spsa_c is not None else self.spsa_c, spsa_seed)
        return self._gradient_parameter_shift(x, y_true)

    def _gradient_spsa(self, x: np.ndarray, y_true: np.ndarray, c: float, seed: Optional[int]) -> Tuple[np.ndarray, float]:
        """SPSA estimate of the full loss's gradient: no chain rule, so no loss.gradient()/output_map.jacobian()."""

        def loss_fn(_ansatz: Ansatz) -> float:        # spsa_gradient rebinds self.ansatz in place, which is what the QNN evaluates
            output = self.output_map.forward(self.qnn.forward_sample(x))
            return float(self.loss(output, y_true))

        gradient = spsa_gradient(self.ansatz, loss_fn, c=c, seed=seed)
        return gradient, loss_fn(self.ansatz)     # one extra evaluation, at the unperturbed point, for the reported loss

    def _gradient_parameter_shift(self, x: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, float]:
        """The exact chain-rule gradient of self.loss for one sample (see the module docstring)."""
        raw = self.qnn.forward_sample(x)                                        # (K,)
        d_raw_d_theta = self.qnn.backward_sample(x)                             # (K, P)

        output = self.output_map.forward(raw)                                   # (K,)
        d_output_d_raw = self.output_map.jacobian(raw)                          # (K, K)

        try:
            d_loss_d_output = np.asarray(self.loss.gradient(output, y_true), dtype=float).reshape(-1)   # (K,)
        except NotImplementedError:
            raise NotImplementedError(
                "{} does not implement an analytic gradient(), so it cannot be used to train "
                "a VQC -- differentiating loss(prediction) directly via parameter-shift is "
                "mathematically incorrect (see AriaQuanta.qml.gradients.parameter_shift_"
                "gradient's docstring). Choose a loss that implements gradient(), or add one "
                "following the same pattern.".format(type(self.loss).__name__)
            )

        loss_value = float(self.loss(output, y_true))
        gradient = d_loss_d_output @ d_output_d_raw @ d_raw_d_theta             # (K,) @ (K,K) @ (K,P) -> (P,)
        return gradient, loss_value

    # ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: Optional[int] = None,
            shuffle: bool = True, verbose: bool = False,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            random_state: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Trains the ansatz's parameters against (X, y) using self.loss/self.optimizer.

        Cost note: with gradient_method='parameter_shift' each gradient step re-evaluates the
        circuit O(batch_size * n_params) times (2 shifted evaluations per parameter per
        sample), each costing num_of_iter_measure shots -- independent of the number of observables.
        With gradient_method='spsa' that drops to 3 evaluations per sample regardless of n_params --
        the usual choice once the ansatz or dataset grows (pair it with qml.training.SPSA and batch_size > 1).

        :param X: (n_samples, n_features) array-like.
        :param y: Targets. Single observable: (n_samples,). List of K observables: either (n_samples,)
                    integer class labels in [0, K) (for CategoricalCrossEntropyLoss) or (n_samples, K)
                    one-hot / multi-output regression targets.
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
                    it does not reset either -- and, for gradient_method='spsa', the c_k
                    perturbation schedule keeps counting from where it left off too.
        """
        X = validate_X(X)
        y = validate_y(y, self.num_of_outputs, self.multi_output)
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
                if self.gradient_method == 'spsa':
                    c_k = spsa_perturbation_schedule(self._optimizer_steps, c=self.spsa_c, gamma=self.spsa_gamma)
                else:
                    c_k = None

                grads, losses = [], []
                for i in batch_idx:
                    # a reproducible perturbation direction per sample when random_state was given
                    seed = int(rng.integers(0, 2**31 - 1)) if self.gradient_method == 'spsa' else None
                    g, l = self._gradient_for_sample(X[i], y[i], spsa_c=c_k, spsa_seed=seed)
                    grads.append(g)
                    losses.append(l)

                batch_grad = np.mean(grads, axis=0)
                new_params = self.optimizer.step(self.ansatz.params_values, batch_grad)
                self.ansatz.set_params_values(new_params)
                self._optimizer_steps += 1
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
    def save(self, path: str) -> None:
        """Saves the trained ansatz parameters (and the training history) to a JSON file."""
        save_params(self.ansatz, path, extra={'history': self.history})

    def load(self, path: str, strict: bool = True) -> List[str]:
        """
        Loads saved parameters into self.ansatz (see persistence.load_params for strict) and
        restores the saved training history. Returns the names of the loaded parameters.
        """
        loaded = load_params(self.ansatz, path, strict=strict)
        self.history = read_params(path)['extra'].get('history', self.history)
        return loaded


    # shortcuts to the QNN ----------------------------------------
    @property
    def feature_map(self): 
        return self.qnn.feature_map
    @property
    def ansatz(self) -> Ansatz: 
        return self.qnn.ansatz
    @property
    def observable(self): 
        return self.qnn.observable
    @property
    def observables(self) -> List[Hamiltonian]: 
        return self.qnn.observables
    @property
    def multi_output(self) -> bool:
        return self.qnn.multi_output
    @property
    def num_of_outputs(self) -> int:
        return self.qnn.num_of_outputs
    @property
    def num_of_iter_measure(self) -> int:
        return self.qnn.num_of_iter_measure
    @num_of_iter_measure.setter
    def num_of_iter_measure(self, value: int) -> None:
        self.qnn.num_of_iter_measure = value


    # ------------------------------------------------------------
    @staticmethod
    def _check_gradient_options(gradient_method, spsa_c, spsa_gamma) -> None:
        if gradient_method not in ('parameter_shift', 'spsa'):
            raise ValueError("'gradient_method' must be 'parameter_shift' or 'spsa', got {!r}.".format(gradient_method))
        if spsa_c <= 0:
            raise ValueError("'spsa_c' must be positive, got {}.".format(spsa_c))
        if spsa_gamma <= 0:
            raise ValueError("'spsa_gamma' must be positive, got {}.".format(spsa_gamma))

    @staticmethod
    def _check_init(loss, optimizer, output_map) -> None:
        # the QNN validates feature_map / ansatz / observable / num_of_iter_measure itself
        if not isinstance(loss, Loss):
            raise TypeError("'loss' must be a Loss instance, got {}.".format(type(loss).__name__))
        if not isinstance(optimizer, Optimizer):
            raise TypeError("'optimizer' must be an Optimizer instance, got {}.".format(type(optimizer).__name__))
        if output_map is not None and not isinstance(output_map, OutputMap):
            raise TypeError("'output_map' must be an OutputMap instance (or None), got {}.".format(type(output_map).__name__))



# -------------------------------------------------------------------------------------------
# Candidates for later (none needed by anything shipped yet -- add when something actually
# needs them):
#   - partial_fit(): fit() already supports calling it more than once on the same VQC
#     instance (ansatz params and optimizer state both persist between calls, a warm
#     start) -- a partial_fit() alias is only a naming/ergonomics addition.
#   - Sampling-based (not observable-based) multi-class readout: read a full measured
#     bitstring distribution via Result.count() instead of a Hamiltonian expectation
#     value (this is what Qiskit's own VQC does: SamplerQNN + bitstring-to-class interpret
#     function) -- a fundamentally different readout mechanism from the one here, i.e. a
#     second QNN class (a SamplerQNN counterpart) rather than a change to VQC.
#   - Quantum kernel methods / QSVM: a different model family entirely (a classical SVM
#     over a quantum-computed kernel matrix, from qml.feature_map's state overlaps) --
#     would be its own class here, not a VQC variant.
# -------------------------------------------------------------------------------------------
