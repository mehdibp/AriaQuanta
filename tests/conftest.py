import numpy as np
import pytest

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.hamiltonian import Hamiltonian
from AriaQuanta.aqc.gatelibrary import RY, CX


# -------------------------------------------------------------------------------------------
# Shared helpers.  Everything here uses only the most basic API (Circuit, Ansatz, RY, CX,
# Hamiltonian.matrix) so a failure in a *higher-level* module (encoding, ansatz, VQC, ...) is
# never caused by the test scaffolding itself.
# -------------------------------------------------------------------------------------------

def _exact_values(circuit, pauli_strings):
    """Exact <psi|P|psi> for every Pauli string, from ONE run of the circuit (no sampling)."""
    n = circuit.num_of_qubits
    psi = np.asarray(circuit.run(), dtype=complex).reshape(-1)
    values = []
    for pauli in pauli_strings:
        matrix = Hamiltonian([(pauli, 1.0)]).matrix(n)
        values.append(float(np.real(np.vdot(psi, matrix @ psi))))
    return np.array(values)


@pytest.fixture
def exact_expectations():
    """Callable (circuit, ['Z0', 'Z1Z2', ...]) -> np.ndarray of exact expectation values."""
    return _exact_values


@pytest.fixture
def exact_measurements(monkeypatch):
    """
    Replaces the shot-based find_expectation_value used by VQC with an exact one, so tests can
    check gradients to ~1e-8 instead of ~1/sqrt(shots). Returns a dict whose 'calls' entry counts
    how many measurements were requested.
    """
    counter = {'calls': 0}

    def exact_find_expectation_value(circuit, hamiltonian, num_of_iter_measure):
        counter['calls'] += 1
        values = _exact_values(circuit, hamiltonian.paulis)
        per_term = {p: float(c) * v for p, c, v in zip(hamiltonian.paulis, hamiltonian.coefs, values)}
        return per_term, sum(per_term.values())

    monkeypatch.setattr('AriaQuanta.qml.models.vqc.find_expectation_value', exact_find_expectation_value)
    return counter


@pytest.fixture
def ry_cx_ansatz():
    """Factory: layers of RY(theta) on every qubit, each followed by a linear CX chain. Parameters t0, t1, ..."""
    def build(num_of_qubits=3, reps=2):
        names = ['t{}'.format(i) for i in range(num_of_qubits * reps)]
        ansatz = Ansatz(num_of_qubits, names)
        k = 0
        for _ in range(reps):
            for q in range(num_of_qubits):
                ansatz | RY(names[k], q)
                k += 1
            for q in range(num_of_qubits - 1):
                ansatz | CX(q, q + 1)
        return ansatz
    return build


@pytest.fixture
def ry_feature_map():
    """Factory: feature_map(x) putting RY(x_q) on qubit q."""
    def build(num_of_qubits=3):
        def feature_map(x):
            circuit = Circuit(num_of_qubits)
            for q in range(num_of_qubits):
                circuit | RY(float(x[q]), q)
            return circuit
        return feature_map
    return build


@pytest.fixture
def three_observables():
    """Three readouts that deliberately share Pauli strings (Z0, Z1Z2), to exercise the merged measurement."""
    return [
        Hamiltonian([('Z0', 1.0), ('Z1Z2', 0.5)]),
        Hamiltonian([('Z1', 1.0), ('Z0', -0.3)]),
        Hamiltonian([('Z2', 1.0), ('Z1Z2', 0.2)]),
    ]
