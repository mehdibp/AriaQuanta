from typing import Callable, List, Optional, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.ansatz import Ansatz


# -------------------------------------------------------------------------------------------
# Simultaneous Perturbation Stochastic Approximation (SPSA; Spall, 1992) gradient estimator.
#
# Where parameter_shift_gradient needs 2 * len(ansatz.params_names) circuit evaluations for
# an *exact* gradient, spsa_gradient needs exactly 2 -- regardless of parameter count -- at
# the cost of the estimate being noisy (a single call is a single random-direction sample;
# only unbiased in expectation over many such calls). This module owns *estimating* a
# gradient; AriaQuanta.qml.training.optimizer.SPSA (a plain Optimizer subclass, taking this
# module's output or any other gradient) owns *applying* it with the decaying step-size
# schedule that makes convergence provable despite the estimator's noise -- the same
# "estimate vs. apply" split parameter_shift_gradient / Optimizer already have.
#
# A genuine bonus over parameter_shift_gradient: because this estimator doesn't rely on the
# raw-expectation-value/sinusoidal structure the parameter-shift rule needs (see
# parameter_shift_gradient's docstring), evaluate_fn here may be a full composed loss
# (`lambda a: mse_loss(model_output(a), y)`) directly -- no chain rule / Loss.gradient()
# required. Confirmed below (in the accompanying tests) to converge to the *correct*
# gradient of a nonlinear composed loss, unlike naively parameter-shifting one.
# -------------------------------------------------------------------------------------------


def spsa_gradient(ansatz: Ansatz, evaluate_fn: Callable[[Ansatz], float],
                   c: float = 0.1, seed: Optional[int] = None,
                   params_values: Optional[Union[List[float], np.ndarray]] = None) -> np.ndarray:
    """
    One SPSA gradient estimate: a single random simultaneous perturbation of *every*
    parameter (a Rademacher vector delta, each component +/-1), evaluated at
    theta +/- c*delta:

        gradient_i ~= ( f(theta + c*delta) - f(theta - c*delta) ) / (2 * c * delta_i)

    This is unbiased in expectation over the random draw of delta, not exact for any single
    call -- SPSA's convergence guarantees come from pairing many such noisy estimates with a
    *decaying* step-size schedule (see AriaQuanta.qml.training.optimizer.SPSA), not from
    trusting one call as if it were the true gradient.

    :param ansatz: Ansatz to differentiate. Its parameters are temporarily perturbed via
                    set_params_values() during evaluation, and restored to their original
                    values (params_values, or ansatz.params_values if not given) before this
                    function returns -- so the ansatz is left exactly as it was found.
    :param evaluate_fn: Callable(ansatz) -> float. Unlike parameter_shift_gradient, this may
                    be a raw prediction (an expectation value) OR a full loss already
                    composed with a prediction -- both are valid for this estimator (see the
                    module docstring for why).
    :param c: Perturbation magnitude. Spall's theory calls for this to shrink across
                    training iterations (see spsa_perturbation_schedule below) -- this
                    function itself only performs one fixed-c estimate; shrinking c across
                    calls is the caller's (e.g. a training loop's) responsibility.
    :param seed: Seeds the random perturbation direction, for reproducibility. None (the
                    default) draws a fresh direction every call, as an actual training loop
                    should.
    :param params_values: Point to differentiate at. Defaults to ansatz.params_values
                    (whatever was last bound via set_params_values).
    :return: Gradient estimate, same length/order as ansatz.params_names.
    """
    _check_validation(ansatz, evaluate_fn, c)
    base_values = _resolve_base_values(ansatz, params_values)
    n = len(ansatz.params_names)

    rng = np.random.default_rng(seed)
    delta = rng.choice(np.array([-1.0, 1.0]), size=n)   # Rademacher perturbation, as SPSA requires

    try:
        ansatz.set_params_values(base_values + c * delta)
        f_plus = float(evaluate_fn(ansatz))

        ansatz.set_params_values(base_values - c * delta)
        f_minus = float(evaluate_fn(ansatz))

        gradient = (f_plus - f_minus) / (2.0 * c * delta)
    finally:
        # leave the ansatz bound to the point the gradient was taken at, regardless of
        # whether evaluate_fn raised partway through the two evaluations above
        ansatz.set_params_values(base_values)

    return gradient


# -------------------------------------------------------------------------------------------
def spsa_perturbation_schedule(k: int, c: float = 0.1, gamma: float = 0.101) -> float:
    """
    Spall's standard decaying perturbation-size schedule: c_k = c / (k + 1) ** gamma.
    Pass the result as spsa_gradient's `c` argument, with k = the current training
    iteration (0-indexed), e.g.:
        spsa_gradient(ansatz, fn, c=spsa_perturbation_schedule(epoch))
    gamma=0.101 is Spall's standard recommendation; c is the schedule's starting magnitude
    (same role as spsa_gradient's own `c` default).
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("'k' must be a non-negative int, got {}.".format(k))
    if c <= 0:
        raise ValueError("'c' must be positive, got {}.".format(c))
    if gamma <= 0:
        raise ValueError("'gamma' must be positive, got {}.".format(gamma))
    return c / (k + 1) ** gamma


# validations / helpers ----------------------------------------------------------------------
def _check_validation(ansatz: Ansatz, evaluate_fn: Callable[[Ansatz], float], c: float) -> None:
    if not isinstance(ansatz, Ansatz):
        raise TypeError("'ansatz' must be an Ansatz instance, got {}.".format(type(ansatz).__name__))
    if not callable(evaluate_fn):
        raise TypeError("'evaluate_fn' must be callable, got {}.".format(type(evaluate_fn).__name__))
    if c <= 0:
        raise ValueError("'c' must be positive, got {}.".format(c))

def _resolve_base_values(ansatz: Ansatz, params_values: Optional[Union[List[float], np.ndarray]]) -> np.ndarray:
    if params_values is None:
        base_values = np.array(ansatz.params_values, dtype=float).copy()
    else:
        base_values = np.asarray(params_values, dtype=float).flatten().copy()

    if base_values.size != len(ansatz.params_names):
        raise ValueError(
            "'params_values' must have length {} (= number of ansatz parameters), got {}."
            .format(len(ansatz.params_names), base_values.size)
        )
    return base_values
