import re
from typing import Dict, List, Optional, Tuple, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.backend.simulator import Simulator
from AriaQuanta.aqc.gatelibrary import H, Sdg


_PAULI_TOKEN_RE = re.compile(r'([XYZI])(\d*)')


# -------------------------------------------------------------------------------------------
class Hamiltonian:
    def __init__(self, terms: List[Tuple[str, float]]) -> None:
        """
        A Hamiltonian expressed as a weighted sum of Pauli strings.

        :param terms: List of (pauli_string, coefficient) tuples.
            Example: H = 0.5*Z1 + 0.3*X2*Z3  ->  terms = [("Z1", 0.5), ("X2Z3", 0.3)]
            Use "I" for a (global) identity term.
        """
        self._check_validation(terms)

        self.terms = list(terms)                        # [("Z1", 0.5), ("X2Z3", 0.3)]
        terms_dict = dict(self.terms)

        self.paulis = list(terms_dict.keys())             # paulis: ['Z1', 'X2Z3']
        self.coefs = np.array(list(terms_dict.values()))  # [0.5 0.3]

    # ------------------------------------------------------------
    @property
    def _check_validation(terms):
        if not isinstance(terms, (list, tuple)) or len(terms) == 0:
            raise ValueError("'terms' must be a non-empty list of (pauli_string, coefficient) tuples.")

        seen_paulis = set()
        for i, term in enumerate(terms):
            if not (isinstance(term, (list, tuple)) and len(term) == 2):
                raise ValueError("terms[{}] must be a (pauli_string, coefficient) pair, got {!r}.".format(i, term))
            pauli_string, coef = term
            _parse_pauli_string(pauli_string)     # validates format; raises ValueError if malformed
            if not isinstance(coef, (int, float, np.floating, np.integer)) or isinstance(coef, bool):
                raise TypeError( "Coefficient for '{}' must be a real number, got {}.".format(pauli_string, type(coef).__name__) )
            if pauli_string in seen_paulis:
                raise ValueError("Duplicate Pauli term '{}' in Hamiltonian terms.".format(pauli_string))
            seen_paulis.add(pauli_string)


# -------------------------------------------------------------------------------------------
def _parse_pauli_string(pauli_string: str, num_of_qubits: Optional[int] = None) -> List[Tuple[int, str]]:
    """
    Parse a Pauli string such as 'X0Z2' into its non-identity (qubit, pauli_char) pairs.
    Identity factors ('I<n>' or the bare global identity string 'I') are dropped, since
    they don't affect the expectation value or need a basis change.

    :param pauli_string: e.g. 'Z0X1', 'X2Z3', or 'I'.
    :param num_of_qubits: If given, every referenced qubit index is checked against it.
    :return: List of (qubit_index, 'X'|'Y'|'Z') pairs, in the order they appear.
    """
    if not isinstance(pauli_string, str) or not pauli_string:
        raise ValueError("pauli_string must be a non-empty string, e.g. 'Z0X1'.")

    if pauli_string == 'I':
        return []

    tokens = _PAULI_TOKEN_RE.findall(pauli_string)
    if ''.join(letter + digits for letter, digits in tokens) != pauli_string:
        raise ValueError(
            "'{}' is not a valid Pauli string: expected letters from {{X, Y, Z, I}}, "
            "each followed by a qubit index, e.g. 'Z0X1'.".format(pauli_string)
        )

    parsed: List[Tuple[int, str]] = []
    seen_qubits = set()
    for letter, digits in tokens:
        if not digits:
            raise ValueError(
                "'{}' in Pauli string '{}' is missing a qubit index (expected e.g. '{}0')."
                .format(letter, pauli_string, letter)
            )
        qubit = int(digits)
        if qubit in seen_qubits:
            raise ValueError("Qubit {} appears more than once in Pauli string '{}'.".format(qubit, pauli_string))
        seen_qubits.add(qubit)
        if num_of_qubits is not None and qubit >= num_of_qubits:
            raise ValueError(
                "Pauli string '{}' references qubit {}, but the circuit only has {} qubits."
                .format(pauli_string, qubit, num_of_qubits)
            )
        if letter != 'I':
            parsed.append((qubit, letter))

    return parsed

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
    active_qubits = _parse_pauli_string(pauli_string, circuit.num_of_qubits)

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
        active_qubits = _parse_pauli_string(pauli_string, num_of_qubits)   # [(qubit, 'X'|'Y'|'Z'), ...]

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

