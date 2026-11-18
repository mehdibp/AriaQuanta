import numpy as np
import pytest
from scipy.linalg import expm

from AriaQuanta.qml.feature_map import z_feature_map, zz_feature_map, iqp_feature_map, pauli_feature_map
from AriaQuanta.qml.feature_map.unit import default_data_map
from AriaQuanta.qml._shared.connectivity import resolve_qubit_subsets
from AriaQuanta.qml._shared.pauli import pauli_tensor_matrix, pauli_evolution_matrix


def final_state(circuit):
    return np.asarray(circuit.run(), dtype=complex).reshape(-1)


# feature maps against a direct numpy construction -----------------------------------------------------------
def test_zz_feature_map_two_qubits_one_rep():
    x0, x1 = 0.4, 1.1
    phi01 = (np.pi - x0) * (np.pi - x1)
    expected = np.zeros(4, dtype=complex)
    for b0 in (0, 1):
        for b1 in (0, 1):
            z0, z1 = (-1) ** b0, (-1) ** b1
            expected[2 * b0 + b1] = 0.5 * np.exp(-1j * (x0 * z0 + x1 * z1 + phi01 * z0 * z1))      # H, then diagonal phases
    np.testing.assert_allclose(final_state(zz_feature_map([x0, x1], reps=1)), expected, atol=1e-12)

def test_z_feature_map_is_a_product_state():
    x = [0.4, 1.1]
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    def qubit(a):
        rz = np.diag([np.exp(-1j * a), np.exp(1j * a)])
        return rz @ H @ rz @ H @ np.array([1.0, 0.0])            # reps = 2: (Rz H) applied twice
    np.testing.assert_allclose(final_state(z_feature_map(x, reps=2)), np.kron(qubit(x[0]), qubit(x[1])), atol=1e-12)

def test_iqp_feature_map_is_diagonal_after_the_hadamards():
    state = final_state(iqp_feature_map([0.3, 0.9, 1.4], degree=3, final_hadamard=False))
    np.testing.assert_allclose(np.abs(state), np.full(8, 1 / np.sqrt(8)), atol=1e-12)

@pytest.mark.parametrize('make', [
    lambda x: zz_feature_map(x, reps=2),
    lambda x: zz_feature_map(x, reps=1, entanglement='linear'),
    lambda x: pauli_feature_map(x, paulis=('X', 'YY', 'ZZ'), reps=2),
    lambda x: iqp_feature_map(x, degree=3),
])
def test_feature_maps_prepare_normalized_states(make):
    state = final_state(make([0.3, 0.9, 1.4]))
    assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

def test_feature_map_depends_on_the_data():
    a = final_state(zz_feature_map([0.3, 0.9]))
    b = final_state(zz_feature_map([0.3, 1.9]))
    assert not np.allclose(a, b)

def test_feature_map_validation():
    with pytest.raises(ValueError):
        zz_feature_map([0.1, 0.2], reps=0)
    with pytest.raises(ValueError):
        zz_feature_map([0.1, 0.2], num_of_qubits=3)            # one feature per qubit
    with pytest.raises(ValueError):
        pauli_feature_map([0.1, 0.2], paulis=('A',))
    with pytest.raises(ValueError):
        iqp_feature_map([0.1, 0.2], degree=0)


# data map ---------------------------------------------------------------------------------------------------------------
def test_default_data_map():
    features = np.array([0.4, 1.1, 2.0])
    assert default_data_map(features, (1,)) == pytest.approx(1.1)
    assert default_data_map(features, (0, 2)) == pytest.approx((np.pi - 0.4) * (np.pi - 2.0))


# connectivity -------------------------------------------------------------------------------------------------------------
def test_resolve_qubit_subsets():
    assert resolve_qubit_subsets(2, 3, 'full') == [(0, 1), (0, 2), (1, 2)]
    assert resolve_qubit_subsets(2, 3, 'linear') == [(0, 1), (1, 2)]
    assert resolve_qubit_subsets(2, 3, 'circular') == [(0, 1), (1, 2), (2, 0)]
    assert resolve_qubit_subsets(3, 4, 'linear') == [(0, 1, 2), (1, 2, 3)]
    assert resolve_qubit_subsets(1, 3, 'full') == [(0,), (1,), (2,)]            # block size 1 ignores the pattern
    assert resolve_qubit_subsets(2, 3, [(0, 2)]) == [(0, 2)]
    assert resolve_qubit_subsets(3, 2, 'circular') == []

@pytest.mark.parametrize('block, n, entanglement', [(2, 3, 'ring'), (2, 3, [(0, 1, 2)]), (2, 3, [(0, 5)])])
def test_resolve_qubit_subsets_rejects_bad_input(block, n, entanglement):
    with pytest.raises(ValueError):
        resolve_qubit_subsets(block, n, entanglement)


# pauli helpers ------------------------------------------------------------------------------------------------------------------
def test_pauli_evolution_matrix_equals_the_matrix_exponential():
    phi = 0.7
    P = pauli_tensor_matrix('XZ')
    U = pauli_evolution_matrix('XZ', phi)
    np.testing.assert_allclose(U, expm(-1j * phi * P), atol=1e-12)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(4), atol=1e-12)

def test_pauli_tensor_matrix_rejects_unknown_letters_and_empty_input():
    with pytest.raises(ValueError):
        pauli_tensor_matrix('XQ')
    with pytest.raises(ValueError):
        pauli_tensor_matrix('')
