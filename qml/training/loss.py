from abc import ABC, abstractmethod
from typing import Dict, Type

from AriaQuanta._utils import np


# -------------------------------------------------------------------------------------------
# Loss functions for AriaQuanta.qml.training.
#
# Implemented in plain numpy (via AriaQuanta._utils.np, so these automatically follow the
# project's existing CPU/GPU dispatch -- Config.use_gpu already decides whether that's numpy
# or cupy) rather than pulled in from TensorFlow/PyTorch: every loss here is 3-10 lines of
# array arithmetic on small vectors, so a multi-hundred-MB deep-learning framework would add
# a heavy, GPU-driver-sensitive dependency for something that doesn't need one -- and would
# sidestep AriaQuanta's own array-module dispatch. If a genuinely elaborate loss shows up
# later (e.g. something needing autodiff through a huge classical head), that's the point to
# revisit this -- not before.
#
# Every loss is a Loss subclass (forward() is required; gradient() -- the analytic
# d(loss)/d(y_pred) -- is optional, but worth implementing where cheap: a future training
# loop can then use the chain rule, d(loss)/d(theta) = d(loss)/d(y_pred) * d(y_pred)/d(theta),
# combining this analytic gradient with a *single* parameter-shift pass over the model,
# rather than parameter-shifting the whole loss(model(theta)) composition -- fewer circuit
# evaluations per training step). Falling back to the latter is still always possible: any
# Loss instance can be dropped into `evaluate_fn = lambda a: my_loss(model_output(a), y)` and
# differentiated end-to-end with AriaQuanta.qml.gradients.parameter_shift_gradient, gradient()
# or not.
#
# Adding a new loss: subclass Loss, implement forward() (and gradient(), if convenient),
# decorate with @register_loss('name') to make it reachable via get_loss('name') and the
# LOSS_REGISTRY dict -- the same lookup pattern AriaQuanta.qml._shared.validation already
# uses for ROTATION_GATES. See the bottom of this file for candidates worth adding later
# (Huber, KL divergence, a fidelity-based loss for state-preparation/autoencoder tasks, an L2
# parameter-regularization wrapper) -- none of them are needed by anything shipped yet, so
# they're deliberately not stubbed out here.
# -------------------------------------------------------------------------------------------


class Loss(ABC):
    """
    Base class for every loss function here. A Loss instance is callable:
    `my_loss(y_pred, y_true) -> float`. Subclasses implement forward() and, optionally,
    gradient() (see the module docstring for why the split exists).
    """
    name: str = ''

    def __call__(self, y_pred, y_true) -> float:
        y_pred, y_true = self._validate(y_pred, y_true)
        return self.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Returns the scalar loss value, averaged over the batch (axis 0)."""
        raise NotImplementedError

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Analytic d(loss)/d(y_pred), same shape as y_pred. Not every loss needs to override
        this -- if it doesn't, differentiate loss(model(theta)) end-to-end instead, via
        AriaQuanta.qml.gradients.parameter_shift_gradient.
        """
        raise NotImplementedError(
            "{} does not implement an analytic gradient -- differentiate it end-to-end with "
            "AriaQuanta.qml.gradients.parameter_shift_gradient instead.".format(type(self).__name__)
        )

    @staticmethod
    def _validate(y_pred, y_true) -> tuple[np.ndarray, np.ndarray]:
        y_pred = np.asarray(y_pred, dtype=float).flatten()
        y_true = np.asarray(y_true, dtype=float).flatten()
        if y_pred.size == 0:
            raise ValueError("'y_pred' must contain at least one element.")
        if y_pred.shape != y_true.shape:
            raise ValueError("'y_pred' and 'y_true' must have the same shape, got {} and {}.".format(y_pred.shape, y_true.shape))
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("'y_pred' contains NaN or Inf values.")
        return y_pred, y_true

    @staticmethod
    def _validate_probabilities(y_pred: np.ndarray, name: str='y_pred', atol: float=1e-6) -> None:
        # catches the common mistake of feeding a raw <Z>-style expectation value in
        # [-1, 1] into a loss that expects an actual probability in [0, 1]
        if np.any(y_pred < -atol) or np.any(y_pred > 1.0 + atol):
            raise ValueError(
                "'{}' must contain probabilities in [0, 1] for this loss, got values in [{}, {}]. "
                "(A raw expectation-value output needs mapping to a probability first -- e.g. "
                "p = (1 + <Z>) / 2 -- before using a cross-entropy loss.)"
                .format(name, float(np.min(y_pred)), float(np.max(y_pred)))
            )


LOSS_REGISTRY: Dict[str, Type[Loss]] = {}

def register_loss(name: str):
    def decorator(cls: Type[Loss]) -> Type[Loss]:
        if name in LOSS_REGISTRY:
            raise ValueError("A loss named {!r} is already registered ({}).".format(name, LOSS_REGISTRY[name].__name__))
        LOSS_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator

def get_loss(name: str) -> Loss:
    """Look up a registered loss by name and return a fresh instance, e.g. get_loss('mse')."""
    if name not in LOSS_REGISTRY:
        raise ValueError("Unknown loss {!r}; available: {}.".format(name, sorted(LOSS_REGISTRY)))
    return LOSS_REGISTRY[name]()


_EPS = 1e-12   # clipping floor for anything that takes a log(), avoids log(0) = -inf


# -------------------------------------------------------------------------------------------
# Regression-style losses -- y_pred/y_true are real numbers (e.g. a <Z> expectation value),
# no [0, 1] restriction.
# -------------------------------------------------------------------------------------------

@register_loss('mse')
class MSELoss(Loss):
    """Mean squared error: mean((y_pred - y_true)^2)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2.0 * (y_pred - y_true) / y_pred.size


@register_loss('mae')
class MAELoss(Loss):
    """Mean absolute error: mean(|y_pred - y_true|)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred - y_true)))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # subgradient 0 at y_pred == y_true (a valid choice, matches most autodiff libraries)
        return np.sign(y_pred - y_true) / y_pred.size


@register_loss('hinge')
class HingeLoss(Loss):
    """
    Hinge loss: mean(max(0, 1 - y_true * y_pred)). y_true must be in {-1, +1}; y_pred is a
    raw margin/score (a <Z>-style expectation value in [-1, 1] is a natural fit -- no
    probability mapping needed, unlike the cross-entropy losses below).
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_pm_one(y_true)
        return float(np.mean(np.maximum(0.0, 1.0 - y_true * y_pred)))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self._validate_pm_one(y_true)
        margin = 1.0 - y_true * y_pred
        return np.where(margin > 0.0, -y_true, 0.0) / y_pred.size

    @staticmethod
    def _validate_pm_one(y_true: np.ndarray) -> None:
        if not np.all(np.isin(y_true, [-1.0, 1.0])):
            raise ValueError("HingeLoss expects 'y_true' values of -1 or +1, got values {}.".format(np.unique(y_true)))


# -------------------------------------------------------------------------------------------
# Classification losses -- y_pred must be actual probabilities.
# -------------------------------------------------------------------------------------------

@register_loss('binary_crossentropy')
class BinaryCrossEntropyLoss(Loss):
    """
    Binary cross-entropy: mean(-[y_true*log(p) + (1-y_true)*log(1-p)]). y_pred (p) must be a
    probability in [0, 1] (e.g. from Result.count()/probabilities(), or a <Z> expectation
    mapped via p = (1 + <Z>) / 2); y_true in {0, 1}.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._validate_probabilities(y_pred)
        p = np.clip(y_pred, _EPS, 1.0 - _EPS)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self._validate_probabilities(y_pred)
        p = np.clip(y_pred, _EPS, 1.0 - _EPS)
        return (-(y_true / p) + (1.0 - y_true) / (1.0 - p)) / y_pred.size


@register_loss('categorical_crossentropy')
class CategoricalCrossEntropyLoss(Loss):
    """
    Categorical cross-entropy: mean_over_samples(-sum_over_classes(y_true * log(p))).
    y_pred: shape (n_samples, n_classes) probability distributions (rows summing to ~1).
    y_true: either one-hot, same shape as y_pred, or a 1-D vector of integer class labels
            (one label per sample) -- converted to one-hot internally.
    """

    def __call__(self, y_pred, y_true) -> float:
        y_pred, y_true = self._validate_categorical(y_pred, y_true)
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        p = np.clip(y_pred, _EPS, 1.0)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=-1)))

    def gradient(self, y_pred, y_true) -> np.ndarray:
        y_pred, y_true = self._validate_categorical(y_pred, y_true)
        p = np.clip(y_pred, _EPS, 1.0)
        return (-y_true / p) / y_pred.shape[0]

    @staticmethod
    def _validate_categorical(y_pred, y_true) -> "tuple[np.ndarray, np.ndarray]":
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        if y_pred.ndim != 2:
            raise ValueError("'y_pred' must be 1-D (one sample) or 2-D (n_samples, n_classes), got shape {}.".format(y_pred.shape))
        Loss._validate_probabilities(y_pred)

        y_true = np.asarray(y_true)
        if y_true.ndim == 1 and y_true.shape[0] != y_pred.shape[1]:
            # a vector of integer class labels, one per sample -- expand to one-hot
            n_samples, n_classes = y_pred.shape
            if y_true.shape[0] != n_samples:
                raise ValueError("'y_true' must have one label per sample ({}), got {}.".format(n_samples, y_true.shape[0]))
            labels = y_true.astype(int)
            if np.any(labels < 0) or np.any(labels >= n_classes):
                raise ValueError("Integer labels in 'y_true' must be in [0, {}), got {}.".format(n_classes, labels))
            one_hot = np.zeros((n_samples, n_classes))
            one_hot[np.arange(n_samples), labels] = 1.0
            y_true = one_hot
        else:
            y_true = y_true.astype(float)
            if y_true.ndim == 1:
                y_true = y_true.reshape(1, -1)

        if y_true.shape != y_pred.shape:
            raise ValueError("'y_true' must have shape {} (one-hot) or ({},) (integer labels), got {}.".format(y_pred.shape, y_pred.shape[0], y_true.shape))

        return y_pred, y_true


# functional convenience access, e.g. mse_loss(y_pred, y_true) -- mirrors get_loss('mse')(...)
mse_loss                      = MSELoss()
mae_loss                      = MAELoss()
hinge_loss                    = HingeLoss()
binary_crossentropy_loss      = BinaryCrossEntropyLoss()
categorical_crossentropy_loss = CategoricalCrossEntropyLoss()


# -------------------------------------------------------------------------------------------
# Candidates for later (none needed by anything shipped yet -- add when something actually
# needs them, following the same Loss/@register_loss pattern above):
#   - HuberLoss              : MSE near 0, MAE far out -- robust regression, less sensitive
#                               to outlier shots/samples than plain MSE.
#   - KLDivergenceLoss       : mean(sum(y_true * log(y_true / y_pred))) -- distribution
#                               matching (e.g. training a feature map / QGAN-style generator
#                               against a target probability distribution, not a fixed label).
#   - FidelityLoss           : 1 - |<target|psi(theta)>|^2 -- for state-preparation /
#                               autoencoder-style tasks where the target is a quantum state,
#                               not a classical label; needs a target statevector rather than
#                               y_true, so it won't fit the Loss.forward(y_pred, y_true)
#                               signature as-is without a small adapter.
#   - L2Regularized (wrapper): wraps another Loss and adds lambda * sum(theta**2) -- needs the
#                               parameter vector itself, not just y_pred/y_true, so it's a
#                               decorator over a Loss + params rather than a plain Loss.
# -------------------------------------------------------------------------------------------
