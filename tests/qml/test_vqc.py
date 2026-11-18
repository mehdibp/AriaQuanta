import re

import numpy as np
import pytest

from AriaQuanta.aqc.hamiltonian import Hamiltonian
from AriaQuanta.qml.models.vqc import VQC, IdentityOutputMap, ZToProbabilityOutputMap, SoftmaxOutputMap
from AriaQuanta.qml.training.loss import CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss, MSELoss, HingeLoss
from AriaQuanta.qml.training.optimizer import Adam, GradientDescent

N, REPS, SCALE = 3, 2, 2.5


@pytest.fixture
def theta0():
    return np.random.default_rng(1).uniform(-np.pi, np.pi, N * REPS)

@pytest.fixture
def sample():
    return np.random.default_rng(2).uniform(0, np.pi, N)

@pytest.fixture
def multiclass_vqc(ry_cx_ansatz, ry_feature_map, three_observables, theta0, exact_measurements):
    return VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), three_observables, CategoricalCrossEntropyLoss(), Adam(learning_rate=0.15),
               output_map=SoftmaxOutputMap(scale=SCALE), initial_params=theta0, num_of_iter_measure=10)

@pytest.fixture
def toy_data():
    X = np.random.default_rng(3).uniform(0, np.pi, (12, N))
    return X, np.argmax(X, axis=1)               # class = which feature is largest


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

def qiskit_softmax_ce(x, theta, observables, label):
    z = SCALE * qiskit_expectations(x, theta, observables)
    p = np.exp(z - z.max())
    p /= p.sum()
    return -np.log(p[label])


# raw readout ---------------------------------------------------------------------------------------------
@pytest.mark.qiskit
def test_raw_expectations_match_qiskit(multiclass_vqc, three_observables, sample, theta0):
    np.testing.assert_allclose(multiclass_vqc._raw_expectations(sample), qiskit_expectations(sample, theta0, three_observables), atol=1e-10)

def test_observables_sharing_pauli_strings_are_measured_once(multiclass_vqc, sample, exact_measurements):
    multiclass_vqc._raw_expectations(sample)
    assert exact_measurements['calls'] == 1                       # 3 observables, 4 distinct Pauli strings -> one merged measurement

def test_raw_expectations_apply_the_observable_coefficients(multiclass_vqc, three_observables, sample, exact_expectations):
    raw = multiclass_vqc._raw_expectations(sample)
    circuit = multiclass_vqc._build_circuit_for(sample, multiclass_vqc.ansatz)
    z0, z1, z2, z12 = exact_expectations(circuit, ['Z0', 'Z1', 'Z2', 'Z1Z2'])
    np.testing.assert_allclose(raw, [z0 + 0.5 * z12, z1 - 0.3 * z0, z2 + 0.2 * z12], atol=1e-12)


# gradients ---------------------------------------------------------------------------------------------------
@pytest.mark.qiskit
def test_parameter_shift_chain_rule_matches_qiskit_finite_differences(multiclass_vqc, three_observables, sample, theta0):
    label = 2
    gradient, loss_value = multiclass_vqc._gradient_for_sample(sample, label)
    h = 1e-6
    fd = np.array([(qiskit_softmax_ce(sample, theta0 + h * e, three_observables, label)
                    - qiskit_softmax_ce(sample, theta0 - h * e, three_observables, label)) / (2 * h) for e in np.eye(theta0.size)])
    assert loss_value == pytest.approx(qiskit_softmax_ce(sample, theta0, three_observables, label), abs=1e-10)
    np.testing.assert_allclose(gradient, fd, atol=1e-7)

def test_integer_label_and_one_hot_target_give_the_same_gradient(multiclass_vqc, sample):
    g_label, l_label = multiclass_vqc._gradient_for_sample(sample, 1)
    g_onehot, l_onehot = multiclass_vqc._gradient_for_sample(sample, np.eye(3)[1])
    np.testing.assert_allclose(g_label, g_onehot, atol=1e-12)
    assert l_label == pytest.approx(l_onehot)

def test_gradient_leaves_the_parameters_untouched(multiclass_vqc, sample, theta0):
    multiclass_vqc._gradient_for_sample(sample, 0)
    np.testing.assert_allclose(multiclass_vqc.ansatz.params_values, theta0)

def test_spsa_path_reports_the_same_loss_and_restores_parameters(ry_cx_ansatz, ry_feature_map, three_observables, theta0, sample, exact_measurements):
    make = lambda method: VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), three_observables, CategoricalCrossEntropyLoss(), Adam(),
                              output_map=SoftmaxOutputMap(SCALE), initial_params=theta0, num_of_iter_measure=10, gradient_method=method)
    _, loss_ps = make('parameter_shift')._gradient_for_sample(sample, 2)
    spsa_model = make('spsa')
    g, loss_spsa = spsa_model._gradient_for_sample(sample, 2, spsa_seed=0)
    assert g.shape == theta0.shape
    assert loss_spsa == pytest.approx(loss_ps)
    np.testing.assert_allclose(spsa_model.ansatz.params_values, theta0)

def test_single_observable_gradient_matches_finite_difference(ry_cx_ansatz, ry_feature_map, sample, theta0, exact_measurements):
    # the K = 1 path must reduce to the scalar chain rule: BCE on p = (1 + <Z0>) / 2
    vqc = VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), BinaryCrossEntropyLoss(), Adam(),
              output_map=ZToProbabilityOutputMap(), initial_params=theta0, num_of_iter_measure=10)
    gradient, loss_value = vqc._gradient_for_sample(sample, 1.0)

    def loss_at(theta):
        vqc.ansatz.set_params_values(theta)
        return vqc._gradient_for_sample(sample, 1.0)[1]
    h = 1e-6
    fd = np.array([(loss_at(theta0 + h * e) - loss_at(theta0 - h * e)) / (2 * h) for e in np.eye(theta0.size)])
    np.testing.assert_allclose(gradient, fd, atol=1e-7)


# prediction shapes and semantics ---------------------------------------------------------------------------------
def test_multi_observable_prediction_shapes(multiclass_vqc, toy_data):
    X, _ = toy_data
    assert multiclass_vqc.predict_raw(X).shape == (12, 3)
    probabilities = multiclass_vqc.predict(X)
    assert probabilities.shape == (12, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(multiclass_vqc.predict_classes(X), np.argmax(probabilities, axis=1))

def test_single_observable_keeps_the_old_one_dimensional_shapes(ry_cx_ansatz, ry_feature_map, toy_data, exact_measurements):
    X, _ = toy_data
    vqc = VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), MSELoss(), Adam(), random_state=0, num_of_iter_measure=10)
    assert vqc.predict_raw(X).shape == (12,)
    assert vqc.predict(X).shape == (12,)
    with pytest.raises(ValueError):
        vqc.predict_classes(X)

def test_evaluate_works_when_n_samples_equals_n_classes(multiclass_vqc, toy_data):
    X, y = toy_data
    value = multiclass_vqc.evaluate(X[:3], y[:3])                 # regression test for the 3-samples / 3-classes ambiguity
    probabilities = multiclass_vqc.predict(X[:3])
    assert value == pytest.approx(-np.mean(np.log(probabilities[np.arange(3), y[:3]])))


# training ---------------------------------------------------------------------------------------------------------
def test_multiclass_fit_reduces_the_loss(multiclass_vqc, toy_data):
    X, y = toy_data
    history = multiclass_vqc.fit(X, y, epochs=15, batch_size=6, random_state=0, validation_data=(X[:3], y[:3]))
    assert len(history['loss']) == len(history['val_loss']) == 15
    assert history['loss'][-1] < 0.8 * history['loss'][0]

def test_multiclass_fit_accepts_one_hot_targets(multiclass_vqc, toy_data):
    X, y = toy_data
    history = multiclass_vqc.fit(X, np.eye(3)[y], epochs=3, batch_size=6, random_state=0)
    assert len(history['loss']) == 3

def test_multiclass_fit_with_spsa_runs(ry_cx_ansatz, ry_feature_map, three_observables, toy_data, exact_measurements):
    from AriaQuanta.qml.training.optimizer import SPSA
    X, y = toy_data
    vqc = VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), three_observables, CategoricalCrossEntropyLoss(), SPSA(a=0.3, A=5),
              output_map=SoftmaxOutputMap(SCALE), random_state=0, num_of_iter_measure=10, gradient_method='spsa')
    history = vqc.fit(X, y, epochs=3, batch_size=6, random_state=0)
    assert np.all(np.isfinite(history['loss']))

def test_binary_fit_with_single_observable(ry_cx_ansatz, ry_feature_map, toy_data, exact_measurements):
    X, _ = toy_data
    y = (X[:, 0] > X[:, 1]).astype(float)
    vqc = VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), BinaryCrossEntropyLoss(), Adam(learning_rate=0.1),
              output_map=ZToProbabilityOutputMap(), random_state=0, num_of_iter_measure=10)
    history = vqc.fit(X, y, epochs=10, batch_size=6, random_state=0)
    assert history['loss'][-1] < history['loss'][0]

def test_multi_output_regression_with_identity_map(ry_cx_ansatz, ry_feature_map, three_observables, toy_data, exact_measurements):
    X, _ = toy_data
    Y = np.stack([np.cos(X[:, 0]), np.cos(X[:, 1])], axis=1)
    vqc = VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), three_observables[:2], MSELoss(), Adam(learning_rate=0.1),
              random_state=0, num_of_iter_measure=10)
    history = vqc.fit(X, Y, epochs=10, batch_size=6, random_state=0)
    assert history['loss'][-1] < history['loss'][0]

def test_fit_is_reproducible_with_seeds(ry_cx_ansatz, ry_feature_map, three_observables, toy_data, exact_measurements):
    X, y = toy_data
    def run():
        vqc = VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), three_observables, CategoricalCrossEntropyLoss(), Adam(learning_rate=0.1),
                  output_map=SoftmaxOutputMap(SCALE), random_state=5, num_of_iter_measure=10)
        return vqc.fit(X, y, epochs=2, batch_size=6, random_state=5)['loss']
    np.testing.assert_allclose(run(), run())


# validation -------------------------------------------------------------------------------------------------------------
def test_softmax_needs_at_least_two_observables(ry_cx_ansatz, ry_feature_map):
    with pytest.raises(ValueError):
        VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), [Hamiltonian([('Z0', 1.0)])], CategoricalCrossEntropyLoss(), Adam(), output_map=SoftmaxOutputMap())

@pytest.mark.parametrize('bad', ['Z0', [], [Hamiltonian([('Z0', 1.0)]), 'Z1'], 3])
def test_observable_must_be_hamiltonian_or_list_of_hamiltonians(ry_cx_ansatz, ry_feature_map, bad):
    with pytest.raises(TypeError):
        VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), bad, MSELoss(), Adam())

def test_loss_without_gradient_cannot_train_with_parameter_shift(ry_cx_ansatz, ry_feature_map, sample, exact_measurements):
    class NoGradient(MSELoss):
        def gradient(self, y_pred, y_true):
            raise NotImplementedError
    vqc = VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), NoGradient(), Adam(), random_state=0, num_of_iter_measure=10)
    with pytest.raises(NotImplementedError):
        vqc._gradient_for_sample(sample, 0.0)

def test_feature_map_and_ansatz_must_have_the_same_width(ry_cx_ansatz, ry_feature_map, sample, exact_measurements):
    vqc = VQC(ry_feature_map(2), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), MSELoss(), Adam(), random_state=0, num_of_iter_measure=10)
    with pytest.raises(ValueError):
        vqc.predict_raw(sample.reshape(1, -1)[:, :2])

def test_input_shapes_are_validated(multiclass_vqc, toy_data):
    X, y = toy_data
    with pytest.raises(ValueError):
        multiclass_vqc.fit(X[0], y)                       # X must be 2-D
    with pytest.raises(ValueError):
        multiclass_vqc.fit(X, y[:-1])                     # different number of samples
    with pytest.raises(ValueError):
        multiclass_vqc.fit(X, np.ones((12, 2)))           # one-hot needs one column per observable
    with pytest.raises(ValueError):
        multiclass_vqc.fit(X, y, batch_size=0)
    with pytest.raises(ValueError):
        multiclass_vqc.fit(X, y, epochs=0)

def test_invalid_gradient_options(ry_cx_ansatz, ry_feature_map):
    for kwargs in (dict(gradient_method='adjoint'), dict(spsa_c=0.0), dict(spsa_gamma=-1.0)):
        with pytest.raises(ValueError):
            VQC(ry_feature_map(N), ry_cx_ansatz(N, REPS), Hamiltonian([('Z0', 1.0)]), MSELoss(), Adam(), **kwargs)
