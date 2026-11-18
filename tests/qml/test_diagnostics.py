import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.hamiltonian import Hamiltonian
from AriaQuanta.aqc.gatelibrary import RY, X
from AriaQuanta.qml.diagnostics import (
    exact_expectation, local_z_hamiltonian, global_z_hamiltonian, GradientStatistics, gradient_statistics,
    BarrenPlateauScan, barren_plateau_scan, plot_barren_plateau,
)

MIXED = Hamiltonian([('X0', 0.5), ('Y1Z2', -0.3), ('Z0X2', 0.7), ('I', 0.2)])


@pytest.fixture
def bound_ansatz(ry_cx_ansatz):
    ansatz = ry_cx_ansatz(3, 2)
    ansatz.set_params_values(np.random.default_rng(0).uniform(-np.pi, np.pi, 6))
    return ansatz


# exact_expectation -------------------------------------------------------------------------------------------------------------
def test_exact_expectation_on_a_basis_state():
    circuit = Circuit(2)
    circuit | X(0)                                                   # |10>
    assert exact_expectation(circuit, Hamiltonian([('Z0', 1.0)])) == pytest.approx(-1.0)
    assert exact_expectation(circuit, Hamiltonian([('Z1', 1.0)])) == pytest.approx(1.0)
    assert exact_expectation(circuit, Hamiltonian([('Z0Z1', 2.0)])) == pytest.approx(-2.0)

def test_exact_expectation_matches_the_dense_matrix(bound_ansatz):
    psi = np.asarray(bound_ansatz.run(), dtype=complex).reshape(-1)
    expected = np.real(np.vdot(psi, MIXED.matrix(3) @ psi))
    assert exact_expectation(bound_ansatz, MIXED) == pytest.approx(expected, abs=1e-12)

@pytest.mark.qiskit
def test_exact_expectation_matches_qiskit_including_x_and_y_terms(bound_ansatz):
    qi = pytest.importorskip('qiskit')
    from qiskit.quantum_info import Statevector, SparsePauliOp
    theta = bound_ansatz.params_values
    circuit = qi.QuantumCircuit(3)
    k = 0
    for _ in range(2):
        for q in range(3):
            circuit.ry(float(theta[k]), q)
            k += 1
        for q in range(2):
            circuit.cx(q, q + 1)
    operator = SparsePauliOp.from_sparse_list([('X', [0], 0.5), ('YZ', [1, 2], -0.3), ('ZX', [0, 2], 0.7), ('I', [0], 0.2)], num_qubits=3)
    expected = Statevector(circuit).expectation_value(operator).real
    assert exact_expectation(bound_ansatz, MIXED) == pytest.approx(expected, abs=1e-10)

def test_exact_expectation_type_checks():
    with pytest.raises(TypeError):
        exact_expectation('circuit', Hamiltonian([('Z0', 1.0)]))
    with pytest.raises(TypeError):
        exact_expectation(Circuit(1), 'hamiltonian')


# cost Hamiltonians --------------------------------------------------------------------------------------------------------------------
def test_local_and_global_hamiltonians():
    assert local_z_hamiltonian(3, qubit=1).paulis == ['Z1']
    assert global_z_hamiltonian(3).paulis == ['Z0Z1Z2']
    with pytest.raises(ValueError):
        local_z_hamiltonian(3, qubit=3)
    with pytest.raises(ValueError):
        global_z_hamiltonian(0)


# gradient statistics ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def one_qubit_ansatz():
    ansatz = Ansatz(1, ['a'])
    ansatz | RY('a', 0)
    return ansatz

def test_gradient_statistics_of_a_known_function(one_qubit_ansatz):
    # <Z> = cos(theta), so dC/dtheta = -sin(theta); with theta ~ U[0, 2pi) its variance is 1/2
    ansatz = one_qubit_ansatz
    ansatz.set_params_values([0.25])
    stats = gradient_statistics(ansatz, local_z_hamiltonian(1), num_of_samples=400, seed=0)
    assert stats.gradients.shape == (400, 1)
    np.testing.assert_allclose(stats.gradients[:, 0], -np.sin(stats.init_values[:, 0]), atol=1e-10)
    assert stats.mean_variance == pytest.approx(0.5, abs=0.15)
    np.testing.assert_allclose(ansatz.params_values, [0.25])                    # restored

def test_gradient_statistics_is_reproducible(one_qubit_ansatz):
    a = gradient_statistics(one_qubit_ansatz, local_z_hamiltonian(1), num_of_samples=5, seed=3)
    b = gradient_statistics(one_qubit_ansatz, local_z_hamiltonian(1), num_of_samples=5, seed=3)
    np.testing.assert_array_equal(a.gradients, b.gradients)

def test_gradient_statistics_validation(one_qubit_ansatz):
    with pytest.raises(ValueError):
        gradient_statistics(one_qubit_ansatz, local_z_hamiltonian(1), num_of_samples=1)
    with pytest.raises(ValueError):
        gradient_statistics(one_qubit_ansatz, local_z_hamiltonian(1), init_range=(1.0, 0.0))
    with pytest.raises(ValueError):
        gradient_statistics(one_qubit_ansatz, local_z_hamiltonian(1), param_indices=[5])
    with pytest.raises(TypeError):
        gradient_statistics('ansatz', local_z_hamiltonian(1))


# scans and the exponential fit --------------------------------------------------------------------------------------------------------------
def synthetic_scan(variance_of):
    qubits = [2, 3, 4, 5]
    stats = []
    for n in qubits:
        a = np.sqrt(variance_of(n) / 2)                    # two samples +a, -a  ->  sample variance 2 a^2
        stats.append(GradientStatistics(num_of_qubits=n, num_of_params=1, num_of_samples=2, param_indices=[0],
                                        gradients=np.array([[a], [-a]]), init_values=np.zeros((2, 1))))
    return BarrenPlateauScan(qubits, stats, label='synthetic')

def test_fit_recovers_an_exponential_decay():
    scan = synthetic_scan(lambda n: 0.5 * 2.0 ** (-n))
    fit = scan.fit_exponential_decay()
    assert fit['rate'] == pytest.approx(np.log(2))
    assert fit['base'] == pytest.approx(2.0)
    assert fit['prefactor'] == pytest.approx(0.5)
    assert fit['r_squared'] == pytest.approx(1.0)
    assert 'fit:' in scan.summary()

def test_fit_reports_no_decay_for_constant_variance():
    assert synthetic_scan(lambda n: 0.3).fit_exponential_decay()['rate'] == pytest.approx(0.0, abs=1e-12)

def test_fit_needs_two_nonzero_variances():
    scan = synthetic_scan(lambda n: 0.0)
    with pytest.raises(ValueError):
        scan.fit_exponential_decay()
    assert 'unavailable' in scan.summary()

def test_barren_plateau_scan_runs_end_to_end():
    def ansatz_factory(n):
        names = ['t{}'.format(q) for q in range(n)]
        ansatz = Ansatz(n, names)
        for q in range(n):
            ansatz | RY(names[q], q)
        return ansatz
    scan = barren_plateau_scan(ansatz_factory, local_z_hamiltonian, scan_values=[1, 2], num_of_samples=10, seed=0, label='local')
    assert [s.num_of_qubits for s in scan.stats] == [1, 2]
    assert np.all(scan.variances > 0)
    fig, ax = plot_barren_plateau(scan)
    assert ax.get_ylabel()

def test_barren_plateau_scan_validation():
    with pytest.raises(ValueError):
        barren_plateau_scan(lambda n: None, local_z_hamiltonian, scan_values=[])
    with pytest.raises(TypeError):
        barren_plateau_scan('not callable', local_z_hamiltonian, scan_values=[1])
