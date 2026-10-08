from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

from AriaQuanta._utils import np


# -------------------------------------------------------------------------------------------
# Gradient-based parameter-update rules for AriaQuanta.qml.training.
#
# Implemented in plain numpy (via AriaQuanta._utils.np -- the project's existing CPU/GPU
# dispatch), not borrowed from TensorFlow/PyTorch: every rule here is a handful of lines of
# array arithmetic on a flat parameter vector, and framework optimizers (tf.keras.optimizers,
# torch.optim) are built around their own Variable/tape-based autodiff -- they don't take a
# plain (params, gradient) pair the way this one needs to, since AriaQuanta's gradients come
# from parameter-shift, not backprop through a tensor framework. So there's nothing to
# actually "borrow" here; a full framework dependency wouldn't even plug in naturally. Same
# reasoning as AriaQuanta.qml.training.loss.
#
# An Optimizer is a pure update rule: `new_params = optimizer.step(params, gradient)`. It
# knows nothing about Ansatz, Loss, or AriaQuanta.qml.gradients -- same independence Loss has
# from those. The piece that actually ties an Ansatz + a Loss + parameter_shift_gradient +
# an Optimizer into a training loop over a dataset belongs one level up, in
# AriaQuanta.qml.models (a future VQC.fit()) -- not here.
#
# Adding a new optimizer: subclass Optimizer, implement step() (using self._state_like(...)
# for any per-parameter state -- momentum, moving averages, ... -- so reset() and the
# shape-mismatch guard come for free), decorate with @register_optimizer('name') to make it
# reachable via get_optimizer('name', **kwargs) and the OPTIMIZER_REGISTRY dict -- the same
# lookup pattern already used by AriaQuanta.qml.training.loss's LOSS_REGISTRY. See the bottom
# of this file for candidates worth adding later (Nesterov momentum, AdamW, quantum natural
# gradient) -- none of them are needed by anything shipped yet, so they're deliberately not
# stubbed out here. SPSA (below) is implemented as a plain Optimizer -- it fits the
# step(params, gradient) contract just fine, since applying a decaying step size to a
# gradient doesn't care where that gradient came from. What genuinely doesn't fit here is
# *estimating* the SPSA-style gradient itself (that needs to call evaluate_fn), which is why
# it lives as its own function, AriaQuanta.qml.gradients.spsa_gradient, alongside
# parameter_shift_gradient -- not in this file.
# -------------------------------------------------------------------------------------------


class Optimizer(ABC):
    """
    Base class for every optimizer here. Stateful across calls (momentum, moving averages,
    step count), so one instance is meant to track one optimization run -- call reset()
    before reusing an instance for a different run (a different ansatz, or a restart).
    """
    name: str = ''

    def __init__(self, learning_rate: float = 0.01) -> None:
        if learning_rate <= 0:
            raise ValueError("'learning_rate' must be positive, got {}.".format(learning_rate))
        self.learning_rate = learning_rate
        self.reset()

    @abstractmethod
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Returns updated params, given the current params and their gradient."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clears all accumulated state (momentum, moving averages, step count)."""
        self._state: Dict[str, np.ndarray] = {}
        self._t: int = 0

    # ------------------------------------------------------------
    def _validate_step_inputs(self, params: np.ndarray, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=float)
        gradient = np.asarray(gradient, dtype=float)
        if params.shape != gradient.shape:
            raise ValueError("'params' and 'gradient' must have the same shape, got {} and {}.".format(params.shape, gradient.shape))
        if not np.all(np.isfinite(gradient)):
            raise ValueError("'gradient' contains NaN or Inf values.")
        return params, gradient

    def _state_like(self, name: str, params: np.ndarray) -> np.ndarray:
        # lazily creates a zero-initialized per-parameter state array on first use, and
        # checks its shape still matches params on every later call -- a mismatch almost
        # always means this instance is being reused for a different problem without reset()
        state = self._state.get(name)
        if state is None:
            state = np.zeros_like(params)
            self._state[name] = state
        elif state.shape != params.shape:
            raise ValueError(
                "This optimizer's {!r} state was built for parameter vectors of shape {}, got {}. "
                "Call reset() before reusing an optimizer instance for a different problem."
                .format(name, state.shape, params.shape)
            )
        return state


OPTIMIZER_REGISTRY: Dict[str, Type[Optimizer]] = {}

def register_optimizer(name: str):
    def decorator(cls: Type[Optimizer]) -> Type[Optimizer]:
        if name in OPTIMIZER_REGISTRY:
            raise ValueError("An optimizer named {!r} is already registered ({}).".format(name, OPTIMIZER_REGISTRY[name].__name__))
        OPTIMIZER_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator

def get_optimizer(name: str, **kwargs) -> Optimizer:
    """Look up a registered optimizer by name and construct it, e.g. get_optimizer('adam', learning_rate=0.05)."""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError("Unknown optimizer {!r}; available: {}.".format(name, sorted(OPTIMIZER_REGISTRY)))
    return OPTIMIZER_REGISTRY[name](**kwargs)


# -------------------------------------------------------------------------------------------
@register_optimizer('sgd')
class GradientDescent(Optimizer):
    """
    Vanilla gradient descent, with optional classical momentum:
        no momentum:   params <- params - lr * gradient
        momentum > 0:  velocity <- momentum * velocity + gradient
                       params   <- params - lr * velocity
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0) -> None:
        if not (0.0 <= momentum < 1.0):
            raise ValueError("'momentum' must be in [0, 1), got {}.".format(momentum))
        self.momentum = momentum
        super().__init__(learning_rate)

    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        params, gradient = self._validate_step_inputs(params, gradient)
        self._t += 1

        if self.momentum == 0.0:
            return params - self.learning_rate * gradient

        velocity = self._state_like('velocity', params)
        velocity = self.momentum * velocity + gradient
        self._state['velocity'] = velocity
        return params - self.learning_rate * velocity


# -------------------------------------------------------------------------------------------
@register_optimizer('adagrad')
class Adagrad(Optimizer):
    """
    Adagrad: accumulates the sum of squared gradients per parameter and scales the learning
    rate down for parameters that have consistently received large gradients.
        sum_sq  <- sum_sq + gradient**2
        params  <- params - lr * gradient / (sqrt(sum_sq) + epsilon)
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8) -> None:
        if epsilon <= 0:
            raise ValueError("'epsilon' must be positive, got {}.".format(epsilon))
        self.epsilon = epsilon
        super().__init__(learning_rate)

    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        params, gradient = self._validate_step_inputs(params, gradient)
        self._t += 1

        sum_sq = self._state_like('sum_sq', params)
        sum_sq = sum_sq + gradient ** 2
        self._state['sum_sq'] = sum_sq

        return params - self.learning_rate * gradient / (np.sqrt(sum_sq) + self.epsilon)


# -------------------------------------------------------------------------------------------
@register_optimizer('rmsprop')
class RMSProp(Optimizer):
    """
    RMSProp: like Adagrad, but the squared-gradient accumulator is an exponential moving
    average (decay_rate) instead of a running sum -- so, unlike Adagrad, the effective
    learning rate doesn't monotonically shrink to zero over a long run.
        avg_sq  <- decay_rate * avg_sq + (1 - decay_rate) * gradient**2
        params  <- params - lr * gradient / (sqrt(avg_sq) + epsilon)
    """

    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.9, epsilon: float = 1e-8) -> None:
        if not (0.0 <= decay_rate < 1.0):
            raise ValueError("'decay_rate' must be in [0, 1), got {}.".format(decay_rate))
        if epsilon <= 0:
            raise ValueError("'epsilon' must be positive, got {}.".format(epsilon))
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        super().__init__(learning_rate)

    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        params, gradient = self._validate_step_inputs(params, gradient)
        self._t += 1

        avg_sq = self._state_like('avg_sq', params)
        avg_sq = self.decay_rate * avg_sq + (1.0 - self.decay_rate) * gradient ** 2
        self._state['avg_sq'] = avg_sq

        return params - self.learning_rate * gradient / (np.sqrt(avg_sq) + self.epsilon)


# -------------------------------------------------------------------------------------------
@register_optimizer('adam')
class Adam(Optimizer):
    """
    Adam (Kingma & Ba, 2014): tracks an exponential moving average of the gradient (m, the
    "momentum" term) and of the squared gradient (v, the "adaptive learning rate" term), each
    bias-corrected for their warm-up at small step counts.
        m       <- beta1 * m + (1 - beta1) * gradient
        v       <- beta2 * v + (1 - beta2) * gradient**2
        m_hat   <- m / (1 - beta1**t)
        v_hat   <- v / (1 - beta2**t)
        params  <- params - lr * m_hat / (sqrt(v_hat) + epsilon)
    The default hyperparameters (beta1=0.9, beta2=0.999, epsilon=1e-8) are the ones from the
    original paper and the usual framework defaults.
    """

    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        if not (0.0 <= beta1 < 1.0):
            raise ValueError("'beta1' must be in [0, 1), got {}.".format(beta1))
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("'beta2' must be in [0, 1), got {}.".format(beta2))
        if epsilon <= 0:
            raise ValueError("'epsilon' must be positive, got {}.".format(epsilon))
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(learning_rate)

    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        params, gradient = self._validate_step_inputs(params, gradient)
        self._t += 1

        m = self._state_like('m', params)
        v = self._state_like('v', params)
        m = self.beta1 * m + (1.0 - self.beta1) * gradient
        v = self.beta2 * v + (1.0 - self.beta2) * gradient ** 2
        self._state['m'], self._state['v'] = m, v

        m_hat = m / (1.0 - self.beta1 ** self._t)
        v_hat = v / (1.0 - self.beta2 ** self._t)
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# -------------------------------------------------------------------------------------------
@register_optimizer('spsa')
class SPSA(Optimizer):
    """
    Decaying-step-size gradient descent, using Spall's classic SPSA gain-sequence schedule:
        a_k     <- a / (t + A) ** alpha            (t = 1, 2, 3, ... -- this instance's own step count)
        params  <- params - a_k * gradient

    This is a plain Optimizer -- same step(params, gradient) contract as every other class
    here, so it accepts a gradient from *any* source. It's specifically designed to be
    paired with AriaQuanta.qml.gradients.spsa_gradient's noisy, 2-evaluation-per-call
    gradient estimate: Spall's theory (the decaying a_k specifically) is what makes
    convergence provable despite that noise. Using spsa_gradient's estimate with a
    fixed-learning-rate optimizer (GradientDescent, Adam, ...) instead still runs, just
    without that guarantee; conversely, pairing SPSA (this class) with an exact gradient
    (e.g. from parameter_shift_gradient) is also valid -- it's just an unusually
    conservative, monotonically-shrinking learning rate schedule in that case.

    :param a: Numerator of the step-size gain sequence (plays the role of a learning rate).
    :param A: Stability constant damping the first few steps. Spall recommends ~10% of the
                    expected total iteration count, if known; 0 (the default) is the common
                    practical choice otherwise.
    :param alpha: Step-size decay exponent. 0.602 is Spall's standard recommendation.
    """

    def __init__(self, a: float = 0.1, A: float = 0.0, alpha: float = 0.602) -> None:
        if A < 0:
            raise ValueError("'A' must be non-negative, got {}.".format(A))
        if alpha <= 0:
            raise ValueError("'alpha' must be positive, got {}.".format(alpha))
        self.A = A
        self.alpha = alpha
        super().__init__(learning_rate=a)   # base class validates a > 0, and gives every Optimizer a .learning_rate attribute

    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        params, gradient = self._validate_step_inputs(params, gradient)
        self._t += 1

        a_k = self.learning_rate / (self._t + self.A) ** self.alpha
        return params - a_k * gradient


# -------------------------------------------------------------------------------------------
# Candidates for later (none needed by anything shipped yet -- add when something actually
# needs them, following the same Optimizer/@register_optimizer pattern above):
#   - Nesterov momentum : a look-ahead variant of GradientDescent's momentum term (evaluate
#                          the gradient at params - lr*momentum*velocity instead of at
#                          params) -- a small change to GradientDescent.step(), not a new
#                          state shape.
#   - AdamW              : Adam plus decoupled weight decay (params -= lr*weight_decay*params
#                          applied separately from the Adam update, not folded into the
#                          gradient like classic L2) -- relevant once a models/vqc.py wants
#                          regularization.
#   - Quantum natural gradient : preconditions the gradient by the inverse Fubini-Study
#                          metric tensor of the ansatz before applying it (i.e. gradient
#                          descent in the circuit's own state-space geometry rather than raw
#                          parameter space) -- directly relevant to the barren-plateau /
#                          expressivity material already studied. Needs the metric tensor
#                          (itself estimated via extra circuit evaluations), so it's a
#                          gradient *transform* that would sit between parameter_shift_
#                          gradient and an Optimizer.step() call, not a step() rule itself.
# -------------------------------------------------------------------------------------------
