import numpy as np
import pytest
from scipy.linalg import expm

from AriaQuanta.qml.encoding import (
    basis_encoding, amplitude_encoding, angle_encoding, hamiltonian_encoding, uniform_superposition, qram_encoding,
)


def final_state(circuit):
    return np.asarray(circuit.run(), dtype=complex).reshape(-1)


def ry(theta):
    return np.array([np.cos(theta / 2), np.sin(theta / 2)])


# basis ----------------------------------------------------------------------------------------------
def test_basis_encoding_prepares_the_basis_state():
    state = final_state(basis_encoding([1, 0, 1]))
    assert abs(state[0b101]) == pytest.approx(1.0)            # qubit 0 is the most significant bit
    assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

def test_basis_encoding_leaves_extra_qubits_in_zero():
    state = final_state(basis_encoding([1, 0, 1], num_of_qubits=4))
    assert abs(state[0b1010]) == pytest.approx(1.0)

@pytest.mark.parametrize('data', [[0, 2], [0.5, 1], []])
def test_basis_encoding_rejects_non_binary_data(data):
    with pytest.raises(ValueError):
        basis_encoding(data)

def test_basis_encoding_rejects_too_few_qubits():
    with pytest.raises(ValueError):
        basis_encoding([1, 0, 1], num_of_qubits=2)


# angle ------------------------------------------------------------------------------------------------
def test_angle_encoding_ry_pi_flips_the_qubit():
    state = final_state(angle_encoding([np.pi, 0.0]))
    assert abs(state[0b10]) == pytest.approx(1.0)

def test_angle_encoding_gives_the_expected_product_state():
    a, b = 0.7, 2.1
    np.testing.assert_allclose(final_state(angle_encoding([a, b])), np.kron(ry(a), ry(b)), atol=1e-12)

def test_angle_encoding_scale_multiplies_every_feature():
    np.testing.assert_allclose(final_state(angle_encoding([0.5], scale=2.0)), ry(1.0), atol=1e-12)

def test_dense_angle_encoding_packs_several_features_per_qubit():
    circuit = angle_encoding([0.1, 0.2, 0.3], rotation=['RY', 'RZ'])
    assert circuit.num_of_qubits == 2
    assert circuit.gatesinfo == {'RY': 2, 'RZ': 1}

def test_angle_encoding_rejects_too_few_qubits():
    with pytest.raises(ValueError):
        angle_encoding([0.1, 0.2, 0.3], num_of_qubits=2)

def test_angle_encoding_rejects_unknown_rotation():
    with pytest.raises(ValueError):
        angle_encoding([0.1], rotation='RQ')

def test_angle_encoding_rejects_non_finite_data():
    with pytest.raises(ValueError):
        angle_encoding([0.1, np.nan])


# superposition -------------------------------------------------------------------------------------------
def test_uniform_superposition():
    state = final_state(uniform_superposition(3))
    np.testing.assert_allclose(state, np.full(8, 1 / np.sqrt(8)), atol=1e-12)

@pytest.mark.parametrize('bad, error', [(0, ValueError), (2.5, TypeError), (True, TypeError)])
def test_uniform_superposition_validation(bad, error):
    with pytest.raises(error):
        uniform_superposition(bad)


# amplitude ---------------------------------------------------------------------------------------------------
def test_amplitude_encoding_normalizes_and_pads():
    circuit = amplitude_encoding([3.0, 4.0])
    np.testing.assert_allclose(np.asarray(circuit.initial_state).reshape(-1), [0.6, 0.8])
    circuit = amplitude_encoding([1.0, 1.0, 1.0])               # padded with a zero to 4 amplitudes
    np.testing.assert_allclose(np.asarray(circuit.initial_state).reshape(-1), [1 / np.sqrt(3)] * 3 + [0.0])

def test_amplitude_encoding_state_survives_run():
    np.testing.assert_allclose(final_state(amplitude_encoding([1, 1, 1, 1])), [0.5] * 4)

def test_amplitude_encoding_rejects_zero_vector_and_too_many_amplitudes():
    with pytest.raises(ValueError):
        amplitude_encoding([0.0, 0.0])
    with pytest.raises(ValueError):
        amplitude_encoding([1.0] * 5, num_of_qubits=2)


# qram ---------------------------------------------------------------------------------------------------------------
def test_qram_encoding_stores_each_word_next_to_its_address():
    circuit = qram_encoding([1, 2, 3, 0])                       # 2 address qubits + 2 data qubits
    assert circuit.num_of_qubits == 4
    state = final_state(circuit)
    expected = np.zeros(16)
    for address, word in enumerate([1, 2, 3, 0]):
        expected[address * 4 + word] = 0.5                      # |address>|word>, qubit 0 = most significant bit
    np.testing.assert_allclose(np.abs(state), expected, atol=1e-12)

def test_qram_encoding_without_superposition_loads_only_the_first_word():
    state = final_state(qram_encoding([2, 1], superposition=False))        # 1 address qubit, 2 data qubits
    assert abs(state[0b0010]) == pytest.approx(1.0)             # address 0, word 2

@pytest.mark.parametrize('data', [[1, -1], [0.5, 1]])
def test_qram_encoding_rejects_non_natural_numbers(data):
    with pytest.raises(ValueError):
        qram_encoding(data)

def test_qram_encoding_rejects_too_few_qubits():
    with pytest.raises(ValueError):
        qram_encoding([1, 2, 3, 4, 5], num_of_address_qubits=2)
    with pytest.raises(ValueError):
        qram_encoding([1, 7], num_of_data_qubits=2)


# hamiltonian ------------------------------------------------------------------------------------------------------------
def test_hamiltonian_encoding_single_qubit_phase():
    state = final_state(hamiltonian_encoding([0.3], ['Z0'], t=1.0, num_of_qubits=1))
    np.testing.assert_allclose(state, [np.exp(-0.3j), 0.0], atol=1e-12)

def test_hamiltonian_encoding_matches_matrix_exponential():
    from AriaQuanta.aqc.hamiltonian import Hamiltonian
    data, paulis, t = [0.4, -0.7, 0.25], ['Z0', 'X1', 'Z0X1'], 0.9
    state = final_state(hamiltonian_encoding(data, paulis, t=t, num_of_qubits=2))
    matrix = Hamiltonian(list(zip(paulis, data))).matrix(2)
    np.testing.assert_allclose(state, expm(-1j * t * matrix)[:, 0], atol=1e-10)

def test_hamiltonian_encoding_infers_the_number_of_qubits():
    # known bug when this suite was written: `hamiltonian.min_num_of_qubits()` is called but its result
    # is never assigned to num_of_qubits, so omitting num_of_qubits raises instead of using 2 qubits.
    circuit = hamiltonian_encoding([0.3, 0.5], ['Z0', 'X1'])
    assert circuit.num_of_qubits == 2

def test_hamiltonian_encoding_needs_one_value_per_pauli_string():
    with pytest.raises(ValueError):
        hamiltonian_encoding([0.3], ['Z0', 'X1'], num_of_qubits=2)
