import numpy as np
import pytest

from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.gatelibrary import RY
from AriaQuanta.qml.gradients.parameter_shift import parameter_shift_gradient, parameter_shift_jacobian
from AriaQuanta.qml.gradients.spsa import spsa_gradient, spsa_perturbation_schedule


@pytest.fixture
def two_qubit_ansatz():
    ansatz = Ansatz(2, ['a', 'b'])
    ansatz | RY('a', 0) | RY('b', 1)
    return ansatz


# parameter shift ---------------------------------------------------------------------------------------
def test_jacobian_matches_analytic_derivatives(two_qubit_ansatz, exact_expectations):
    # RY(a)|0> -> <Z> = cos(a), so d<Z0>/da = -sin(a); the two qubits are independent -> diagonal Jacobian
    theta = np.array([0.7, -1.3])
    two_qubit_ansatz.set_params_values(theta)
    J = parameter_shift_jacobian(two_qubit_ansatz, lambda a: exact_expectations(a, ['Z0', 'Z1']))
    assert J.shape == (2, 2)
    np.testing.assert_allclose(J, np.diag(-np.sin(theta)), atol=1e-12)

def test_jacobian_of_a_product_observable(two_qubit_ansatz, exact_expectations):
    # <Z0 Z1> = cos(a) cos(b)
    a, b = 0.4, 1.1
    two_qubit_ansatz.set_params_values([a, b])
    J = parameter_shift_jacobian(two_qubit_ansatz, lambda s: exact_expectations(s, ['Z0Z1']))
    np.testing.assert_allclose(J, [[-np.sin(a) * np.cos(b), -np.cos(a) * np.sin(b)]], atol=1e-12)

def test_gradient_is_the_scalar_special_case_of_the_jacobian(two_qubit_ansatz, exact_expectations):
    two_qubit_ansatz.set_params_values([0.4, 1.1])
    f = lambda s: exact_expectations(s, ['Z0Z1'])[0]
    g = parameter_shift_gradient(two_qubit_ansatz, f)
    J = parameter_shift_jacobian(two_qubit_ansatz, lambda s: [f(s)])
    assert g.shape == (2,)
    np.testing.assert_allclose(g, J[0])

def test_explicit_params_values_override_the_bound_ones_and_the_ansatz_is_left_at_that_point(two_qubit_ansatz, exact_expectations):
    two_qubit_ansatz.set_params_values([0.1, 0.2])
    point = np.array([0.9, -0.4])
    J = parameter_shift_jacobian(two_qubit_ansatz, lambda a: exact_expectations(a, ['Z0', 'Z1']), params_values=point)
    np.testing.assert_allclose(J, np.diag(-np.sin(point)), atol=1e-12)
    np.testing.assert_allclose(two_qubit_ansatz.params_values, point)               # left bound to the point differentiated at

def test_ansatz_is_restored_even_when_evaluate_fn_raises(two_qubit_ansatz):
    two_qubit_ansatz.set_params_values([0.1, 0.2])
    calls = {'n': 0}
    def failing(a):
        calls['n'] += 1
        if calls['n'] == 3:
            raise RuntimeError('boom')
        return [0.0]
    with pytest.raises(RuntimeError):
        parameter_shift_jacobian(two_qubit_ansatz, failing)
    np.testing.assert_allclose(two_qubit_ansatz.params_values, [0.1, 0.2])

def test_evaluate_fn_is_called_twice_per_parameter_regardless_of_output_size(two_qubit_ansatz):
    calls = {'n': 0}
    def f(a):
        calls['n'] += 1
        return np.zeros(5)
    parameter_shift_jacobian(two_qubit_ansatz, f)
    assert calls['n'] == 2 * 2

def test_inconsistent_output_size_raises(two_qubit_ansatz):
    calls = {'n': 0}
    def f(a):
        calls['n'] += 1
        return np.zeros(2 if calls['n'] % 2 else 3)
    with pytest.raises(ValueError):
        parameter_shift_jacobian(two_qubit_ansatz, f)

@pytest.mark.parametrize('shift', [0.0, np.pi, 2 * np.pi])
def test_shift_multiple_of_pi_is_rejected(two_qubit_ansatz, shift):
    with pytest.raises(ValueError):
        parameter_shift_gradient(two_qubit_ansatz, lambda a: 0.0, shift=shift)

def test_scalar_gradient_rejects_vector_valued_function(two_qubit_ansatz):
    with pytest.raises((TypeError, ValueError)):
        parameter_shift_gradient(two_qubit_ansatz, lambda a: np.zeros(2))

def test_wrong_length_params_values_raises(two_qubit_ansatz):
    with pytest.raises(ValueError):
        parameter_shift_jacobian(two_qubit_ansatz, lambda a: [0.0], params_values=[0.1, 0.2, 0.3])

def test_non_ansatz_raises():
    with pytest.raises(TypeError):
        parameter_shift_gradient('not an ansatz', lambda a: 0.0)


# SPSA -----------------------------------------------------------------------------------------------------
def test_spsa_is_unbiased_on_a_known_function(two_qubit_ansatz):
    # f(a, b) = cos(a) + a*b  ->  grad = (-sin(a) + b, a). Cheap to evaluate, so average many estimates.
    theta = np.array([0.6, -0.8])
    two_qubit_ansatz.set_params_values(theta)
    f = lambda s: float(np.cos(s.params_values[0]) + s.params_values[0] * s.params_values[1])
    estimates = np.array([spsa_gradient(two_qubit_ansatz, f, c=0.01, seed=s) for s in range(20000)])
    expected = np.array([-np.sin(theta[0]) + theta[1], theta[0]])
    np.testing.assert_allclose(estimates.mean(axis=0), expected, atol=0.05)

def test_spsa_is_reproducible_with_a_seed_and_restores_parameters(two_qubit_ansatz):
    two_qubit_ansatz.set_params_values([0.3, 0.9])
    f = lambda s: float(np.sum(np.sin(s.params_values)))
    g1 = spsa_gradient(two_qubit_ansatz, f, seed=7)
    g2 = spsa_gradient(two_qubit_ansatz, f, seed=7)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_allclose(two_qubit_ansatz.params_values, [0.3, 0.9])

def test_spsa_uses_exactly_two_evaluations(two_qubit_ansatz):
    calls = {'n': 0}
    def f(a):
        calls['n'] += 1
        return 0.0
    spsa_gradient(two_qubit_ansatz, f, seed=0)
    assert calls['n'] == 2

def test_spsa_validation(two_qubit_ansatz):
    with pytest.raises(ValueError):
        spsa_gradient(two_qubit_ansatz, lambda a: 0.0, c=0.0)
    with pytest.raises(TypeError):
        spsa_gradient(two_qubit_ansatz, 'not callable')

def test_spsa_perturbation_schedule():
    assert spsa_perturbation_schedule(0, c=0.2) == pytest.approx(0.2)
    assert spsa_perturbation_schedule(9, c=0.2, gamma=0.5) == pytest.approx(0.2 / 10 ** 0.5)
    for bad in [dict(k=-1), dict(k=1, c=0.0), dict(k=1, gamma=0.0), dict(k=1.5)]:
        with pytest.raises(ValueError):
            spsa_perturbation_schedule(**bad)
