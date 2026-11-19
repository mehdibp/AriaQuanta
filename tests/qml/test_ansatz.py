import numpy as np
import pytest

from AriaQuanta.qml.ansatz import HardwareEfficientAnsatz, EfficientSU2Ansatz, DataReUploadingAnsatz


def final_state(circuit):
    return np.asarray(circuit.run(), dtype=complex).reshape(-1)


def qiskit_state(num_of_qubits, build):
    qi = pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector
    circuit = qi.QuantumCircuit(num_of_qubits)
    build(circuit)
    return Statevector(circuit).reverse_qargs().data           # Qiskit is little-endian, AriaQuanta has qubit 0 as the leftmost bit


# Layer convention shared by both ansatzes: `reps` + 1 rotation layers (layers 0 .. reps).
# HardwareEfficientAnsatz puts an entangler block after every rotation layer except the last one.

# HardwareEfficientAnsatz -------------------------------------------------------------------------------------
def test_parameter_names_and_order():
    ansatz = HardwareEfficientAnsatz(3, rotation=('RY', 'RZ'), reps=2)
    assert len(ansatz.params_names) == (2 + 1) * 3 * 2                  # reps + 1 rotation layers
    assert ansatz.params_names[:3] == ['theta_l0_q0_RY', 'theta_l0_q0_RZ', 'theta_l0_q1_RY']
    assert ansatz.params_names[-1] == 'theta_l2_q2_RZ'
    assert len(set(ansatz.params_names)) == len(ansatz.params_names)

@pytest.mark.parametrize('reps', [0, 1, 3])
def test_reps_entangler_blocks_and_reps_plus_one_rotation_layers(reps):
    info = HardwareEfficientAnsatz(3, rotation='RY', reps=reps).gatesinfo
    assert info['RY'] == 3 * (reps + 1)
    assert info.get('CX', 0) == 2 * reps                                 # linear connectivity: 2 pairs per block

def test_gate_counts_per_entanglement_pattern():
    counts = lambda **kw: HardwareEfficientAnsatz(3, rotation='RY', reps=1, **kw).gatesinfo
    assert counts(entanglement='linear') == {'RY': 6, 'CX': 2}
    assert counts(entanglement='circular') == {'RY': 6, 'CX': 3}
    assert counts(entanglement='full') == {'RY': 6, 'CX': 3}
    assert counts(entanglement=[(0, 2)]) == {'RY': 6, 'CX': 1}
    assert counts(entangler='CZ') == {'RY': 6, 'CZ': 2}

def test_the_last_rotation_layer_has_no_trailing_entangler():
    ansatz = HardwareEfficientAnsatz(3, rotation='RY', reps=2)
    assert [g.name for g in ansatz.operations[-3:]] == ['RY', 'RY', 'RY']

@pytest.mark.qiskit
def test_statevector_matches_qiskit():
    n, reps = 3, 2
    ansatz = HardwareEfficientAnsatz(n, rotation='RY', reps=reps)
    theta = np.random.default_rng(0).uniform(-np.pi, np.pi, len(ansatz.params_names))
    ansatz.set_params_values(theta)

    def build(circuit):
        k = 0
        for layer in range(reps + 1):
            for q in range(n):
                circuit.ry(float(theta[k]), q)
                k += 1
            if layer < reps:
                for q in range(n - 1):
                    circuit.cx(q, q + 1)
    np.testing.assert_allclose(final_state(ansatz), qiskit_state(n, build), atol=1e-10)

# EfficientSU2Ansatz is a preset of HardwareEfficientAnsatz ----------------------------------------------------
def test_efficient_su2_is_the_ry_rz_cx_linear_preset():
    n, reps = 3, 2
    preset = EfficientSU2Ansatz(n, reps=reps)
    generic = HardwareEfficientAnsatz(n, rotation=('RY', 'RZ'), entangler='CX', entanglement='linear', reps=reps)
    assert isinstance(preset, HardwareEfficientAnsatz)
    assert preset.params_names == generic.params_names
    assert preset.gatesinfo == generic.gatesinfo == {'RY': 9, 'RZ': 9, 'CX': 4}
    theta = np.random.default_rng(4).uniform(-np.pi, np.pi, len(preset.params_names))
    preset.set_params_values(theta)
    generic.set_params_values(theta)
    np.testing.assert_allclose(final_state(preset), final_state(generic), atol=1e-12)

@pytest.mark.qiskit
def test_efficient_su2_matches_qiskit_up_to_parameter_ordering():
    pytest.importorskip('qiskit')
    try:
        from qiskit.circuit.library import efficient_su2
    except ImportError:
        pytest.skip('this Qiskit version has no efficient_su2() function')
    n, reps = 3, 2
    ansatz = EfficientSU2Ansatz(n, reps=reps)
    theta = np.random.default_rng(4).uniform(-np.pi, np.pi, len(ansatz.params_names))
    ansatz.set_params_values(theta)

    # Qiskit orders a layer's angles as all RY (qubit 0..n-1), then all RZ; ours is per qubit: RY, RZ
    theta_qiskit = np.zeros_like(theta)
    for layer in range(reps + 1):
        for q in range(n):
            for k in range(2):                                           # k = 0: RY, k = 1: RZ
                theta_qiskit[layer * 2 * n + k * n + q] = theta[layer * 2 * n + q * 2 + k]
    circuit = efficient_su2(n, entanglement='linear', reps=reps).assign_parameters(theta_qiskit)

    from qiskit.quantum_info import Statevector
    np.testing.assert_allclose(final_state(ansatz), Statevector(circuit).reverse_qargs().data, atol=1e-10)

def test_ansatz_validation():
    with pytest.raises(ValueError):
        HardwareEfficientAnsatz(0)
    with pytest.raises(ValueError):
        HardwareEfficientAnsatz(2, reps=-1)
    with pytest.raises(ValueError):
        HardwareEfficientAnsatz(2, entangler='CY')
    with pytest.raises(ValueError):
        HardwareEfficientAnsatz(2, rotation='P')                 # no trainable form


# DataReUploadingAnsatz --------------------------------------------------------------------------------------------
def test_data_reuploading_separates_inputs_from_weights():
    ansatz = DataReUploadingAnsatz(n_features=3, reps=2)
    assert ansatz.input_names == ['x0', 'x1', 'x2']
    assert len(ansatz.params_names) == (2 + 1) * 3                      # reps + 1 layers, 3 qubits, 1 trainable rotation
    assert not set(ansatz.input_names) & set(ansatz.params_names)

@pytest.mark.parametrize('reps, uploads', [(0, 1), (1, 2), (3, 4)])
def test_data_is_uploaded_reps_plus_one_times(reps, uploads):
    ansatz = DataReUploadingAnsatz(n_features=3, reps=reps)
    assert ansatz.gatesinfo['RY'] == uploads * 3 * 2                    # per layer: 3 upload rotations + 3 trainable rotations

@pytest.mark.qiskit
def test_data_reuploading_matches_qiskit():
    n, reps = 3, 2
    rng = np.random.default_rng(1)
    ansatz = DataReUploadingAnsatz(n_features=n, reps=reps)
    theta, x = rng.uniform(-np.pi, np.pi, len(ansatz.params_names)), rng.uniform(0, np.pi, n)
    ansatz.set_params_values(theta)
    ansatz.set_input_values(x)

    def build(circuit):
        k = 0
        for _ in range(reps + 1):
            for q in range(n):
                circuit.ry(float(x[q]), q)                       # data upload
            for q in range(n):
                circuit.ry(float(theta[k]), q)                   # trainable layer
                k += 1
            for q in range(n):                                   # ring entangler: currently after EVERY layer, the last one included
                circuit.cx(q, (q + 1) % n)
    np.testing.assert_allclose(final_state(ansatz), qiskit_state(n, build), atol=1e-10)

def test_data_reuploading_scale_multiplies_the_inputs():
    x = np.array([0.4, 0.9])
    scaled = DataReUploadingAnsatz(n_features=2, reps=1, scale=2.0)
    plain = DataReUploadingAnsatz(n_features=2, reps=1)
    theta = np.random.default_rng(5).uniform(-1, 1, len(plain.params_names))
    for model, value in ((scaled, x), (plain, 2.0 * x)):
        model.set_params_values(theta)
        model.set_input_values(value)
    np.testing.assert_allclose(final_state(scaled), final_state(plain), atol=1e-12)

def test_data_reuploading_with_data_binds_it_immediately():
    ansatz = DataReUploadingAnsatz(data=[0.1, 0.2, 0.3], reps=1)
    np.testing.assert_allclose(ansatz.input_values, [0.1, 0.2, 0.3])

def test_data_reuploading_reps_zero_is_a_single_layer():
    ansatz = DataReUploadingAnsatz(n_features=3, reps=0)
    assert len(ansatz.params_names) == 3
    assert ansatz.num_of_layers == 1

def test_data_reuploading_validation():
    with pytest.raises(ValueError):
        DataReUploadingAnsatz()                                  # neither data nor n_features
    with pytest.raises(ValueError):
        DataReUploadingAnsatz(n_features=2, reps=-1)
    with pytest.raises(ValueError):
        DataReUploadingAnsatz(data=[0.1, 0.2], n_features=3)
    with pytest.raises(ValueError):
        DataReUploadingAnsatz(n_features=4, num_of_qubits=2)

