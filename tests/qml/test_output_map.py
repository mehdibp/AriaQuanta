import numpy as np
import pytest

from AriaQuanta.qml.models.vqc import OutputMap, IdentityOutputMap, ZToProbabilityOutputMap, SoftmaxOutputMap


def jacobian_by_finite_difference(output_map, raw, h=1e-6):
    raw = np.asarray(raw, dtype=float)
    columns = [(output_map.forward(raw + h * e) - output_map.forward(raw - h * e)) / (2 * h) for e in np.eye(raw.size)]
    return np.stack(columns, axis=1)                       # J[i, j] = d out_i / d raw_j


def test_identity():
    m = IdentityOutputMap()
    raw = np.array([0.3, -0.7])
    np.testing.assert_allclose(m.forward(raw), raw)
    np.testing.assert_allclose(m.jacobian(raw), np.eye(2))

def test_z_to_probability():
    m = ZToProbabilityOutputMap()
    np.testing.assert_allclose(m.forward(np.array([-1.0, 0.0, 1.0])), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(m.jacobian(np.array([0.3])), [[0.5]])

def test_elementwise_map_gets_diagonal_jacobian_by_default():
    class Square(OutputMap):
        def forward(self, raw):
            return np.asarray(raw, dtype=float) ** 2
        def derivative(self, raw):
            return 2 * np.asarray(raw, dtype=float)
    np.testing.assert_allclose(Square().jacobian(np.array([1.0, 2.0, 3.0])), np.diag([2.0, 4.0, 6.0]))


# softmax ------------------------------------------------------------------------------------------
def test_softmax_rows_sum_to_one_and_are_positive():
    p = SoftmaxOutputMap().forward(np.array([[0.2, -0.5, 0.9], [1.0, 1.0, 1.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all(p > 0)
    np.testing.assert_allclose(p[1], 1 / 3)

def test_softmax_matches_definition():
    z = np.array([0.3, -0.2, 0.5])
    expected = np.exp(2.0 * z) / np.sum(np.exp(2.0 * z))
    np.testing.assert_allclose(SoftmaxOutputMap(scale=2.0).forward(z), expected)

def test_softmax_is_shift_invariant_and_stable_for_large_values():
    m = SoftmaxOutputMap()
    z = np.array([0.1, 0.4, -0.3])
    np.testing.assert_allclose(m.forward(z), m.forward(z + 1000.0))
    assert np.all(np.isfinite(m.forward(np.array([1e4, 0.0, -1e4]))))

def test_larger_scale_gives_more_confident_output():
    z = np.array([1.0, -1.0, -1.0])
    assert SoftmaxOutputMap(scale=1.0).forward(z)[0] == pytest.approx(np.e / (np.e + 2 / np.e))     # ~0.787: the unscaled ceiling
    assert SoftmaxOutputMap(scale=3.0).forward(z)[0] > 0.99

@pytest.mark.parametrize('scale', [1.0, 2.5])
def test_softmax_jacobian_matches_finite_difference(scale):
    m = SoftmaxOutputMap(scale=scale)
    raw = np.array([0.3, -0.2, 0.5, 0.1])
    np.testing.assert_allclose(m.jacobian(raw), jacobian_by_finite_difference(m, raw), atol=1e-8)

def test_softmax_jacobian_structure():
    J = SoftmaxOutputMap(scale=1.7).jacobian(np.array([0.3, -0.2, 0.5]))
    np.testing.assert_allclose(J, J.T)                          # symmetric
    np.testing.assert_allclose(J.sum(axis=1), 0.0, atol=1e-12)   # outputs always sum to 1 -> rows of J sum to 0

def test_softmax_jacobian_requires_a_single_sample():
    with pytest.raises(ValueError):
        SoftmaxOutputMap().jacobian(np.zeros((2, 3)))

def test_softmax_is_not_elementwise():
    with pytest.raises(NotImplementedError):
        SoftmaxOutputMap().derivative(np.zeros(3))

@pytest.mark.parametrize('scale', [0.0, -1.0])
def test_softmax_rejects_non_positive_scale(scale):
    with pytest.raises(ValueError):
        SoftmaxOutputMap(scale=scale)

def test_softmax_cross_entropy_gradient_wrt_raw_is_p_minus_y():
    # chain rule used by VQC: dL/draw = (dL/dp) @ J  must equal scale * (p - y)  (torch CrossEntropyLoss, up to scale)
    from AriaQuanta.qml.training.loss import CategoricalCrossEntropyLoss
    m = SoftmaxOutputMap(scale=1.0)
    raw = np.array([0.3, -0.2, 0.5])
    p = m.forward(raw)
    g = CategoricalCrossEntropyLoss().gradient(p, np.array([2])).reshape(-1)
    np.testing.assert_allclose(g @ m.jacobian(raw), p - np.eye(3)[2], atol=1e-12)
