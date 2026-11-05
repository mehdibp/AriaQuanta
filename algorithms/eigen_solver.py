from typing import Dict, List, Tuple

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.backend.simulator import Simulator
from AriaQuanta.aqc.gatelibrary import H, Sdg
from AriaQuanta.aqc.hamiltonian import Hamiltonian, parse_pauli_string


# -------------------------------------------------------------------------------------------
def _term_parity(bits: str, active_qubits: List[int]) -> int:
    # +1 if the measured bits on the active (non-identity) qubits have even parity, else -1
    total = sum(int(bits[q]) for q in active_qubits)
    return (-1) ** (total % 2)


# -------------------------------------------------------------------------------------------
def pauli_transform_circuit(circuit: Circuit, pauli_string: str) -> Circuit:
    """
    Return a *copy* of circuit with the basis-change gates needed to measure pauli_string
    in the Z basis appended (H for X, Sdg+H for Y; Z/I need no rotation). The original
    circuit is left untouched.

    :param circuit: Circuit to base the rotated copy on.
    :param pauli_string: e.g. 'Z0X1'.
    :return: A new Circuit with the basis-change gates appended.
    """
    circuit_copy = circuit.copy()
    active_qubits = parse_pauli_string(pauli_string, circuit.num_of_qubits)

    for qubit, pauli_char in active_qubits:
        if pauli_char == 'X':
            circuit_copy | H(qubit)
        elif pauli_char == 'Y':
            circuit_copy | Sdg(qubit)
            circuit_copy | H(qubit)
        # 'Z' needs no basis change

    return circuit_copy

# -------------------------------------------------------------------------------------------
def find_expectation_value(circuit: Circuit, hamiltonian: Hamiltonian, num_of_iter_measure: int) -> Tuple[Dict[str, float], float]:
    """
    Estimate <psi|H|psi> for the state prepared by circuit, by sampling each Pauli
    term of hamiltonian num_of_iter_measure times.

    :param circuit: Circuit that prepares the state |psi> (not yet measured).
    :param hamiltonian: Hamiltonian whose expectation value is estimated.
    :param num_of_iter_measure: Number of measurement shots per Pauli term.
    :return: (per-term weighted expectation values, total energy)
    """
    if not isinstance(circuit, Circuit):
        raise TypeError("'circuit' must be a Circuit instance, got {}.".format(type(circuit).__name__))
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("'hamiltonian' must be a Hamiltonian instance, got {}.".format(type(hamiltonian).__name__))
    if not isinstance(num_of_iter_measure, int) or isinstance(num_of_iter_measure, bool):
        raise TypeError("'num_of_iter_measure' must be an int, got {}.".format(type(num_of_iter_measure).__name__))
    if num_of_iter_measure < 1:
        raise ValueError("'num_of_iter_measure' must be at least 1, got {}.".format(num_of_iter_measure))

    num_of_qubits = circuit.num_of_qubits
    paulis = hamiltonian.paulis
    coefs = hamiltonian.coefs

    # Run once in the Z basis; every term that only involves Z/I reuses this result.
    sim = Simulator()
    result_z = sim.simulate(circuit.copy(), num_of_iter_measure, 4, show_progress=False)
    _, probability_z = result_z.count()

    pauli_exp_value: Dict[str, float] = {}
    total_energy = 0.0

    for pauli_string, coef in zip(paulis, coefs):
        active_qubits = parse_pauli_string(pauli_string, num_of_qubits)   # [(qubit, 'X'|'Y'|'Z'), ...]

        if not active_qubits:
            # Every factor is identity -> expectation is trivially 1
            this_exp_value = 1.0
        else:
            needs_basis_change = any(pauli_char in ('X', 'Y') for _, pauli_char in active_qubits)
            if needs_basis_change:
                circuit_rotated = pauli_transform_circuit(circuit, pauli_string)
                result = sim.simulate(circuit_rotated, num_of_iter_measure, 4, show_progress=False)
                _, probability = result.count()
            else:
                probability = probability_z

            active_qubit_indices = [qubit for qubit, _ in active_qubits]
            this_exp_value = sum(
                prob * _term_parity(state.strip('|>'), active_qubit_indices)
                for state, prob in probability.items()
            )

        pauli_exp_value[pauli_string] = this_exp_value * coef
        total_energy += this_exp_value * coef

    # example:
    # {'I': 1.0, 'Z0': 0.0, 'Z1': 0.2, 'Z0Z1': 0.4, 'X0X1': -0.4}
    #  0.2
    return pauli_exp_value, total_energy



