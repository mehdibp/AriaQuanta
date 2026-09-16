from typing import Callable, List, Optional, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.qml._shared import validate_parameter_shift, validate_parameter_shift_expectation


# -------------------------------------------------------------------------------------------
# Generic parameter-shift gradient estimator for any AriaQuanta Ansatz.
#
# This is deliberately independent of AriaQuanta.algorithms.eigen_solver.Hamiltonian /
# find_expectation_value: it only needs `evaluate_fn(ansatz) -> float`, so the same function
# differentiates a Hamiltonian expectation value (VQE/QAOA-style), a measured class
# probability (a future qml.models.VQC), or any other scalar readout built on top of an
# Ansatz -- without qml.gradients depending on qml.models or vice versa.
# -------------------------------------------------------------------------------------------


def parameter_shift_gradient(ansatz: Ansatz, evaluate_fn: Callable[[Ansatz], float],
                            params_values: Optional[Union[List[float], np.ndarray]]=None, shift: float=np.pi/2) -> np.ndarray:
    """
    Estimate d(evaluate_fn)/d(θ_i) for every trainable parameter in ansatz.params_names,
    using the parameter-shift rule: ( ### evaluate_fn := f(θ) ### )

        d<f>/dθ_i = ( f(θ_i + shift) - f(θ_i - shift) ) / (2 * sin(shift))

    This is *exact* (not a finite-difference approximation) for every rotation gate
    AriaQuanta ships -- RX/RY/RZ/P/CRX/CRY/CRZ/CNP/... (anything built on the
    `_AxisRotationGate` pattern) -- because each has a generator with eigenvalues +/-1.
    shift=π/2 (the default) is the standard, best-conditioned choice, since sin(π/2) = 1.

    :param ansatz: Ansatz to differentiate. Its parameters are repeatedly overwritten via
                    set_params_values() during evaluation, and restored to their original
                    values (params_values, or ansatz.params_values if not given) before
                    this function returns -- so the ansatz is left exactly as it was found.
    :param evaluate_fn: Callable that takes `ansatz` (with parameters already bound for this
                    call) and returns a single float, e.g.:
                        lambda a: find_expectation_value(a, hamiltonian, shots)[1]
                    Called twice per parameter (2 * len(ansatz.params_names) calls total).
    :param params_values: Point to differentiate at. Defaults to ansatz.params_values
                    (whatever was last bound via set_params_values / bind_parameters).
    :param shift: Shift amount in radians. Must not be a multiple of π (sin(shift) would be
                    zero, making the estimator undefined). Default π/2.
    :return: Gradient vector, same length and order as ansatz.params_names.
    """

    # _resolve_base_values
    if params_values is None: base_values = np.array(ansatz.params_values, dtype=float).copy()
    else: base_values = np.asarray(params_values, dtype=float).flatten().copy()

    validate_parameter_shift(ansatz, evaluate_fn, shift, base_values)

    n = len(ansatz.params_names)
    denom = 2.0 * np.sin(shift)

    gradient = np.zeros(n)
    try:
        for i in range(n):
            plus_values = base_values.copy()
            plus_values[i] += shift
            ansatz.set_params_values(plus_values)
            f_plus = float(evaluate_fn(ansatz))

            minus_values = base_values.copy()
            minus_values[i] -= shift
            ansatz.set_params_values(minus_values)
            f_minus = float(evaluate_fn(ansatz))

            gradient[i] = (f_plus - f_minus) / denom
    finally:
        # leave the ansatz bound to the point the gradient was taken at, regardless of
        # whether evaluate_fn raised partway through the loop above
        ansatz.set_params_values(base_values)

    return gradient


# -------------------------------------------------------------------------------------------
def parameter_shift_gradient_expectation(ansatz: Ansatz, hamiltonian, num_of_iter_measure: int,
                                         params_values: Optional[Union[List[float], np.ndarray]]=None,
                                         shift: float=np.pi/2) -> np.ndarray:
    """
    Convenience wrapper for the common case of differentiating a Hamiltonian expectation
    value: <ansatz(θ)|hamiltonian|ansatz(θ)>. Bridges to
    AriaQuanta.algorithms.eigen_solver.find_expectation_value (already used by VQE/QAOA) so
    this shape of gradient doesn't need to be re-derived at each call site.

    :param ansatz: Ansatz to differentiate (see parameter_shift_gradient).
    :param hamiltonian: AriaQuanta.algorithms.eigen_solver.Hamiltonian instance.
    :param num_of_iter_measure: Number of measurement shots per expectation-value estimate
                    (passed straight through to find_expectation_value -- called
                    2 * len(ansatz.params_names) times, so cost scales accordingly).
    :param params_values: See parameter_shift_gradient.
    :param shift: See parameter_shift_gradient.
    :return: Gradient of the total energy, same length/order as ansatz.params_names.
    """
    # imported lazily, here rather than at module level, so qml.gradients doesn't force a
    # circular import with AriaQuanta.algorithms merely by being imported itself
    from AriaQuanta.algorithms.eigen_solver import find_expectation_value

    validate_parameter_shift_expectation(hamiltonian, num_of_iter_measure)

    def evaluate_fn(a: Ansatz) -> float:
        _, total_energy = find_expectation_value(a, hamiltonian, num_of_iter_measure)
        return total_energy

    return parameter_shift_gradient(ansatz, evaluate_fn, params_values=params_values, shift=shift)


