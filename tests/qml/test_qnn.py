import re

import numpy as np
import pytest

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.hamiltonian import Hamiltonian
from AriaQuanta.aqc.gatelibrary import RY
from AriaQuanta.qml.ansatz import DataReUploadingAnsatz
from AriaQuanta.qml.models.qnn import QNN

N, REPS = 3, 2


@pytest.fixture
def theta0():
    return np.random.default_rng(1).uniform(-np.pi, np.pi, N * REPS)

@pytest.fixture
def X():
    return np.random.default_rng(2).uniform(0, np.pi, (4, N))

@pytest.fixture
def qnn(ry_cx_ansatz, ry_feature_map, three_observables, theta0, exact_measurements):
    return QNN(ry_feature_map(N), ry_cx_ansatz(N, REPS), three_observables, num_of_iter_measure=10, initial_params=theta0)


# Qiskit reference --------------------------------------------------------------------------------------
def qiskit_expectations(x, theta, observables):
    qi = pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector, SparsePauliOp
    circuit = qi.QuantumCircuit(N)
    for q in range(N):
        circuit.ry(float(x[q]), q)
    k = 0
    for _ in range(REPS):
        for q in range(N):
            circuit.ry(float(theta[k]), q)
            k += 1
        for q in range(N - 1):
            circuit.cx(q, q + 1)
    state = Statevector(circuit)

    def operator(h):
        terms = []
        for pauli, coef in zip(h.paulis, h.coefs):
            qubits = [int(q) for _, q in re.findall(r'([XYZI])(\d+)', pauli)]
            terms.append(('Z' * len(qubits), qubits, float(coef)))
        return SparsePauliOp.from_sparse_list(terms, num_qubits=N)

    return np.array([state.expectation_value(operator(h)).real for h in observables])


# forward ---------------------------------------------------------------------------------------------------
@pytest.mark.qiskit
def test_forward_matches_qiskit(qnn, three_observables, X, theta0):
    expected = np.array([qiskit_expectations(x, theta0, three_observables) for x in X])
    np.testing.assert_allclose(qnn.forward(X), expected, atol=1e-10)

def test_forward_shape_is_samples_by_outputs(qnn, X):
    assert qnn.forward(X).shape == (4, 3)
    assert qnn.forward_sample(X[0]).shape == (3,)

def test_single_hamiltonian_still_has_an_output_axis(ry_cx_ansatz, ry_feature_map, X, exact_measurements):
    qnn = QNN(ry_feature_map(N), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), num_of_iter_measure=10, random_state=0)
    assert qnn.num_of_outputs == 1 and not qnn.multi_output
    assert qnn.forward(X).shape == (4, 1)
    assert qnn.backward(X).shape == (4, 1, N * REPS)

def test_observables_sharing_pauli_strings_are_measured_once(qnn, X, exact_measurements):
    qnn.forward_sample(X[0])
    assert exact_measurements['calls'] == 1                       # 3 observables, 4 distinct Pauli strings -> one merged measurement

def test_forward_applies_the_observable_coefficients(qnn, X, exact_expectations):
    raw = qnn.forward_sample(X[0])
    circuit = qnn._build_circuit_for(X[0])
    z0, z1, z2, z12 = exact_expectations(circuit, ['Z0', 'Z1', 'Z2', 'Z1Z2'])
    np.testing.assert_allclose(raw, [z0 + 0.5 * z12, z1 - 0.3 * z0, z2 + 0.2 * z12], atol=1e-12)


# backward --------------------------------------------------------------------------------------------------
@pytest.mark.qiskit
def test_backward_matches_qiskit_finite_differences(qnn, three_observables, X, theta0):
    h = 1e-6
    jacobian = qnn.backward(X)
    assert jacobian.shape == (4, 3, theta0.size)
    for s, x in enumerate(X):
        fd = np.stack([(qiskit_expectations(x, theta0 + h * e, three_observables)
                        - qiskit_expectations(x, theta0 - h * e, three_observables)) / (2 * h) for e in np.eye(theta0.size)], axis=1)
        np.testing.assert_allclose(jacobian[s], fd, atol=1e-7)

def test_backward_of_a_single_rotation_is_minus_sine(exact_measurements):
    # RY(a)|0>: <Z0> = cos(a)  ->  d<Z0>/da = -sin(a), whatever the (ignored) sample is
    ansatz = Ansatz(1, ['a'])
    ansatz | RY('a', 0)
    qnn = QNN(lambda x: Circuit(1), ansatz, Hamiltonian([('Z0', 1.0)]), initial_params=[0.7])
    np.testing.assert_allclose(qnn.backward(np.zeros((2, 1))), -np.sin(0.7) * np.ones((2, 1, 1)), atol=1e-12)

def test_backward_leaves_the_weights_untouched(qnn, X, theta0):
    qnn.backward(X[:1])
    np.testing.assert_allclose(qnn.params_values, theta0)


# weights ------------------------------------------------------------------------------------------------------
def test_weights_live_in_the_ansatz(qnn, theta0):
    assert qnn.num_of_params == theta0.size == len(qnn.params_names)
    new = theta0 + 0.1
    qnn.set_params_values(new)
    np.testing.assert_allclose(qnn.ansatz.params_values, new)

def test_initial_params_default_is_reproducible_with_random_state(ry_cx_ansatz, ry_feature_map, three_observables):
    make = lambda seed: QNN(ry_feature_map(N), ry_cx_ansatz(N, REPS), three_observables, random_state=seed).params_values
    np.testing.assert_array_equal(make(3), make(3))
    assert not np.allclose(make(3), make(4))

def test_wrong_number_of_initial_params_raises(ry_cx_ansatz, ry_feature_map, three_observables):
    with pytest.raises(ValueError):
        QNN(ry_feature_map(N), ry_cx_ansatz(N, REPS), three_observables, initial_params=[0.1, 0.2])


# data carried by the ansatz (feature_map=None) --------------------------------------------------------------------------
def test_feature_map_none_with_a_data_reuploading_ansatz(exact_measurements, exact_expectations):
    ansatz = DataReUploadingAnsatz(n_features=3, reps=1)
    qnn = QNN(None, ansatz, [Hamiltonian([('Z0', 1.0)]), Hamiltonian([('Z1', 1.0)])], random_state=0)
    x = np.array([0.3, 1.1, 2.0])
    ansatz.set_input_values(x)
    expected = exact_expectations(ansatz, ['Z0', 'Z1'])
    np.testing.assert_allclose(qnn.forward_sample(x), expected, atol=1e-12)
    assert qnn.backward_sample(x).shape == (2, len(ansatz.params_names))


# validation -----------------------------------------------------------------------------------------------------------------
def test_ansatz_must_be_an_ansatz(ry_feature_map):
    with pytest.raises(TypeError):
        QNN(ry_feature_map(N), 'ansatz', Hamiltonian([('Z0', 1.0)]))

@pytest.mark.parametrize('bad', ['Z0', [], [Hamiltonian([('Z0', 1.0)]), 'Z1'], 3])
def test_observable_must_be_hamiltonian_or_list_of_hamiltonians(ry_cx_ansatz, ry_feature_map, bad):
    with pytest.raises(TypeError):
        QNN(ry_feature_map(N), ry_cx_ansatz(N, REPS), bad)

def test_feature_map_and_ansatz_must_have_the_same_width(ry_cx_ansatz, ry_feature_map, X, exact_measurements):
    qnn = QNN(ry_feature_map(2), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), random_state=0)
    with pytest.raises(ValueError):
        qnn.forward(X[:, :2])

def test_feature_map_none_needs_input_parameters(ry_cx_ansatz):
    with pytest.raises(ValueError):
        QNN(None, ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]))

def test_feature_map_and_input_parameters_are_mutually_exclusive(ry_feature_map):
    with pytest.raises(ValueError):
        QNN(ry_feature_map(3), DataReUploadingAnsatz(n_features=3, reps=1), Hamiltonian([('Z0', 1.0)]))

@pytest.mark.parametrize('shots, error', [(0, ValueError), (2.5, TypeError), (True, TypeError)])
def test_num_of_iter_measure_validation(ry_cx_ansatz, ry_feature_map, shots, error):
    with pytest.raises(error):
        QNN(ry_feature_map(N), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), num_of_iter_measure=shots)

def test_x_must_be_two_dimensional(qnn, X):
    with pytest.raises(ValueError):
        qnn.forward(X[0])
    with pytest.raises(ValueError):
        qnn.backward(X[0])
