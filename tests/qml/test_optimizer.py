import numpy as np
import pytest

from AriaQuanta.qml.training.optimizer import (
    Optimizer, GradientDescent, Adagrad, RMSProp, Adam, SPSA, OPTIMIZER_REGISTRY, get_optimizer,
)

P = np.array([1.0, -2.0, 0.5])
G = np.array([0.4, -0.1, 2.0])


# single-step formulas -------------------------------------------------------------------------
def test_gradient_descent_step():
    np.testing.assert_allclose(GradientDescent(learning_rate=0.1).step(P, G), P - 0.1 * G)

def test_gradient_descent_with_momentum_two_steps():
    opt = GradientDescent(learning_rate=0.1, momentum=0.9)
    g1, g2 = G, np.array([1.0, 1.0, 1.0])
    p1 = opt.step(P, g1)
    v1 = g1
    np.testing.assert_allclose(p1, P - 0.1 * v1)
    v2 = 0.9 * v1 + g2
    np.testing.assert_allclose(opt.step(p1, g2), p1 - 0.1 * v2)

def test_adagrad_first_step():
    np.testing.assert_allclose(Adagrad(learning_rate=0.5).step(P, G), P - 0.5 * G / (np.abs(G) + 1e-8))

def test_rmsprop_first_step():
    avg = 0.1 * G ** 2
    np.testing.assert_allclose(RMSProp(learning_rate=0.05, decay_rate=0.9).step(P, G), P - 0.05 * G / (np.sqrt(avg) + 1e-8))

def test_adam_first_step_is_a_sign_step():
    # with bias correction, step 1 gives m_hat = g and v_hat = g**2, i.e. p - lr * sign(g)
    np.testing.assert_allclose(Adam(learning_rate=0.1).step(P, G), P - 0.1 * np.sign(G), atol=1e-6)

def test_spsa_step_size_schedule():
    opt = SPSA(a=0.3, A=5.0, alpha=0.602)
    p1 = opt.step(P, G)
    np.testing.assert_allclose(p1, P - 0.3 / (1 + 5.0) ** 0.602 * G)
    np.testing.assert_allclose(opt.step(p1, G), p1 - 0.3 / (2 + 5.0) ** 0.602 * G)


# convergence on a convex quadratic f(p) = 0.5 * ||p - target||^2 ---------------------------------------
@pytest.mark.parametrize('make, steps, tol', [
    (lambda: GradientDescent(learning_rate=0.2), 100, 1e-6),
    (lambda: GradientDescent(learning_rate=0.05, momentum=0.9), 300, 1e-3),
    (lambda: Adagrad(learning_rate=0.5), 500, 1e-2),
    (lambda: RMSProp(learning_rate=0.02), 600, 5e-2),
    (lambda: Adam(learning_rate=0.1), 400, 1e-2),
    (lambda: SPSA(a=0.5, A=0.0, alpha=0.1), 300, 5e-2),
])
def test_converges_on_quadratic(make, steps, tol):
    target = np.array([1.0, -2.0, 0.5])
    params = np.zeros(3)
    opt = make()
    for _ in range(steps):
        params = opt.step(params, params - target)
    np.testing.assert_allclose(params, target, atol=tol)


# state handling -------------------------------------------------------------------------------------
def test_reset_restores_first_step_behaviour():
    opt = Adam(learning_rate=0.1)
    first = opt.step(P, G)
    opt.step(P, G)
    opt.reset()
    np.testing.assert_allclose(opt.step(P, G), first)

def test_reusing_an_optimizer_for_a_different_problem_without_reset_raises():
    opt = Adam()
    opt.step(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        opt.step(np.zeros(5), np.ones(5))
    opt.reset()
    opt.step(np.zeros(5), np.ones(5))      # fine after reset

def test_step_does_not_modify_its_inputs():
    p, g = P.copy(), G.copy()
    Adam().step(p, g)
    np.testing.assert_array_equal(p, P)
    np.testing.assert_array_equal(g, G)


# validation --------------------------------------------------------------------------------------------
def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        GradientDescent().step(np.zeros(3), np.zeros(2))

def test_non_finite_gradient_raises():
    with pytest.raises(ValueError):
        GradientDescent().step(np.zeros(2), np.array([1.0, np.nan]))

@pytest.mark.parametrize('make', [
    lambda: GradientDescent(learning_rate=0.0),
    lambda: GradientDescent(momentum=1.0),
    lambda: Adagrad(epsilon=0.0),
    lambda: RMSProp(decay_rate=1.0),
    lambda: Adam(beta1=1.0),
    lambda: Adam(beta2=-0.1),
    lambda: SPSA(A=-1.0),
    lambda: SPSA(alpha=0.0),
])
def test_invalid_hyperparameters_raise(make):
    with pytest.raises(ValueError):
        make()


# registry --------------------------------------------------------------------------------------------------
def test_get_optimizer_by_name():
    opt = get_optimizer('adam', learning_rate=0.05)
    assert isinstance(opt, Adam) and opt.learning_rate == 0.05

def test_get_optimizer_unknown_name_raises():
    with pytest.raises(ValueError):
        get_optimizer('nope')

def test_all_shipped_optimizers_are_registered_and_are_optimizers():
    assert {'sgd', 'adagrad', 'rmsprop', 'adam', 'spsa'} <= set(OPTIMIZER_REGISTRY)
    assert all(issubclass(cls, Optimizer) for cls in OPTIMIZER_REGISTRY.values())
