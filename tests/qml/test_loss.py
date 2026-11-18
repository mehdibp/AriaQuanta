import numpy as np
import pytest

from AriaQuanta.qml.training.loss import (
    Loss, MSELoss, MAELoss, HingeLoss, BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss,
    LOSS_REGISTRY, register_loss, get_loss, mse_loss,
)


def finite_difference(loss, y_pred, y_true, h=1e-6):
    y_pred = np.asarray(y_pred, dtype=float)
    grad = np.zeros_like(y_pred)
    for idx in np.ndindex(*y_pred.shape):
        plus, minus = y_pred.copy(), y_pred.copy()
        plus[idx] += h
        minus[idx] -= h
        grad[idx] = (loss(plus, y_true) - loss(minus, y_true)) / (2 * h)
    return grad


P3 = np.array([[0.7, 0.2, 0.1],
               [0.1, 0.8, 0.1],
               [0.2, 0.2, 0.6]])


# values -------------------------------------------------------------------------------------
def test_mse_value():
    assert MSELoss()([1, 2, 3], [1, 1, 1]) == pytest.approx(5 / 3)

def test_mae_value():
    assert MAELoss()([1, 2, 3], [1, 1, 1]) == pytest.approx(1.0)

def test_hinge_value():
    assert HingeLoss()([0.5, -0.5], [1, -1]) == pytest.approx(0.5)

def test_binary_crossentropy_value():
    expected = -(np.log(0.8) + np.log(0.7)) / 2
    assert BinaryCrossEntropyLoss()([0.8, 0.3], [1, 0]) == pytest.approx(expected)

def test_categorical_crossentropy_value_with_labels_and_one_hot():
    expected = -np.mean(np.log([0.7, 0.8, 0.6]))
    loss = CategoricalCrossEntropyLoss()
    assert loss(P3, [0, 1, 2]) == pytest.approx(expected)
    assert loss(P3, np.eye(3)) == pytest.approx(expected)

def test_functional_instances_match_classes():
    assert mse_loss([1, 2], [0, 0]) == MSELoss()([1, 2], [0, 0])


# gradients agree with finite differences ------------------------------------------------------
@pytest.mark.parametrize('loss, y_pred, y_true', [
    (MSELoss(),  [0.3, -0.8, 0.1], [1.0, 0.0, 0.5]),
    (MAELoss(),  [0.5, 2.0],       [1.0, 1.0]),                 # away from the kink at y_pred == y_true
    (HingeLoss(), [0.2, -0.4],     [1.0, -1.0]),                # both margins positive
    (HingeLoss(), [2.0, -3.0],     [1.0, -1.0]),                # both margins negative -> zero gradient
    (BinaryCrossEntropyLoss(), [0.8, 0.3, 0.55], [1.0, 0.0, 1.0]),
])
def test_gradient_matches_finite_difference(loss, y_pred, y_true):
    np.testing.assert_allclose(loss.gradient(np.array(y_pred), np.array(y_true)),
                               finite_difference(loss, y_pred, y_true), atol=1e-6)

def test_categorical_gradient_matches_finite_difference_batch():
    loss = CategoricalCrossEntropyLoss()
    np.testing.assert_allclose(loss.gradient(P3, [0, 1, 2]), finite_difference(loss, P3, [0, 1, 2]), atol=1e-6)

def test_categorical_gradient_single_sample():
    loss = CategoricalCrossEntropyLoss()
    p = np.array([0.5, 0.3, 0.2])
    grad = loss.gradient(p, np.array([1]))
    assert grad.shape == (1, 3)
    np.testing.assert_allclose(grad[0], finite_difference(lambda a, y: loss(a, y), p, np.array([1])), atol=1e-6)


# categorical: how y_true is interpreted -------------------------------------------------------
def test_categorical_labels_are_not_mistaken_for_one_hot_when_n_samples_equals_n_classes():
    # regression test: 3 samples with 3 classes used to be misread as a single one-hot vector
    loss = CategoricalCrossEntropyLoss()
    assert loss(P3, [0, 1, 2]) == pytest.approx(-np.mean(np.log([0.7, 0.8, 0.6])))
    assert loss(P3, [2, 2, 2]) == pytest.approx(-np.mean(np.log([0.1, 0.1, 0.6])))

def test_categorical_single_sample_label_scalar_and_one_hot_agree():
    loss = CategoricalCrossEntropyLoss()
    p = np.array([0.5, 0.3, 0.2])
    expected = -np.log(0.3)
    assert loss(p, [1]) == pytest.approx(expected)
    assert loss(p, 1) == pytest.approx(expected)
    assert loss(p, [0, 1, 0]) == pytest.approx(expected)

def test_categorical_float_labels_are_accepted():
    assert CategoricalCrossEntropyLoss()(P3, np.array([0.0, 1.0, 2.0])) == pytest.approx(-np.mean(np.log([0.7, 0.8, 0.6])))

@pytest.mark.parametrize('y_true', [[0, 1, 3], [-1, 0, 1], [0, 1]])
def test_categorical_rejects_bad_labels(y_true):
    with pytest.raises(ValueError):
        CategoricalCrossEntropyLoss()(P3, y_true)

def test_categorical_rejects_non_probabilities():
    with pytest.raises(ValueError):
        CategoricalCrossEntropyLoss()(np.array([[0.5, 1.7, -0.2]]), [0])

def test_categorical_rejects_wrong_one_hot_shape():
    with pytest.raises(ValueError):
        CategoricalCrossEntropyLoss()(P3, np.ones((3, 2)))


# input validation shared by the plain losses ---------------------------------------------------
def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        MSELoss()([1, 2, 3], [1, 2])

def test_empty_input_raises():
    with pytest.raises(ValueError):
        MSELoss()([], [])

def test_nan_prediction_raises():
    with pytest.raises(ValueError):
        MSELoss()([np.nan, 1.0], [0.0, 0.0])

def test_hinge_requires_plus_minus_one_labels():
    with pytest.raises(ValueError):
        HingeLoss()([0.1, 0.2], [0, 1])

def test_binary_crossentropy_rejects_raw_expectation_values():
    with pytest.raises(ValueError):
        BinaryCrossEntropyLoss()([-0.4, 0.2], [0, 1])       # a raw <Z> in [-1, 1], not a probability

def test_loss_without_gradient_raises_not_implemented():
    class NoGradient(Loss):
        def forward(self, y_pred, y_true):
            return 0.0
    with pytest.raises(NotImplementedError):
        NoGradient().gradient(np.zeros(2), np.zeros(2))


# registry -----------------------------------------------------------------------------------------
def test_get_loss_returns_fresh_instances():
    assert isinstance(get_loss('mse'), MSELoss)
    assert get_loss('mse') is not get_loss('mse')

def test_get_loss_unknown_name_raises():
    with pytest.raises(ValueError):
        get_loss('does_not_exist')

def test_register_loss_rejects_duplicate_names():
    with pytest.raises(ValueError):
        @register_loss('mse')
        class Another(Loss):
            def forward(self, y_pred, y_true):
                return 0.0

def test_all_shipped_losses_are_registered():
    assert {'mse', 'mae', 'hinge', 'binary_crossentropy', 'categorical_crossentropy'} <= set(LOSS_REGISTRY)
