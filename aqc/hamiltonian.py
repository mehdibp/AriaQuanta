import re
from typing import List, Optional, Sequence, Tuple

from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary import X, Y, Z


# -------------------------------------------------------------------------------------------
# Hamiltonians as weighted sums of Pauli strings. This module only describes the operator
# (terms, parsing, dense matrix) -- it knows nothing about Circuit / Simulator, so it can be used by
# algorithms (VQE/QAOA), qml (encoding, VQC) and anything else without pulling in the algorithms package.
# Estimating <psi|H|psi> from a circuit lives in algorithms.eigen_solver (find_expectation_value), not here.
# -------------------------------------------------------------------------------------------

_PAULI_TOKEN_RE = re.compile(r'([XYZI])(\d*)')

PAULI_MATRIX = {'X': X().matrix, 'Y': Y().matrix, 'Z': Z().matrix, 'I': np.eye(2, dtype=complex)}


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

        self.terms = list(terms)                          # [("Z1", 0.5), ("X2Z3", 0.3)]
        terms_dict = dict(self.terms)

        self.paulis = list(terms_dict.keys())             # paulis: ['Z1', 'X2Z3']
        self.coefs = np.array(list(terms_dict.values()))  # [0.5 0.3]

    # ------------------------------------------------------------
    def min_num_of_qubits(self) -> int:
        """1 + the highest qubit index referenced by any term (1 if every term is the bare identity)."""
        max_qubit = 0
        for pauli_string in self.paulis:
            parsed = parse_pauli_string(pauli_string)
            if parsed:
                max_qubit = max(max_qubit, max(q for q, _ in parsed))
        return max_qubit + 1

    # ------------------------------------------------------------
    def matrix(self, num_of_qubits: Optional[int] = None) -> np.ndarray:
        """
        Dense 2**n x 2**n matrix of H = sum_i coef_i * P_i (qubit 0 is the leftmost
        Kronecker factor, as everywhere in AriaQuanta).

        :param num_of_qubits: Defaults to min_num_of_qubits(); may be larger (extra qubits get identity).
        """
        minimum = self.min_num_of_qubits()
        n = minimum if num_of_qubits is None else num_of_qubits
        if n < minimum:
            raise ValueError("This Hamiltonian references qubit {}, so it needs at least {} qubit(s), got {}.".format(minimum-1, minimum, n))
        return hamiltonian_matrix(self.paulis, self.coefs, n)

    # ------------------------------------------------------------
    @staticmethod
    def _check_validation(terms: List[Tuple[str, float]]):
        if not isinstance(terms, (list, tuple)) or len(terms) == 0:
            raise ValueError("'terms' must be a non-empty list of (pauli_string, coefficient) tuples.")

        seen_paulis = set()
        for i, term in enumerate(terms):
            if not (isinstance(term, (list, tuple)) and len(term) == 2):
                raise ValueError("terms[{}] must be a (pauli_string, coefficient) pair, got {!r}.".format(i, term))
            pauli_string, coef = term
            parse_pauli_string(pauli_string)     # validates format; raises ValueError if malformed
            if not isinstance(coef, (int, float, np.floating, np.integer)) or isinstance(coef, bool):
                raise TypeError("Coefficient for '{}' must be a real number, got {}.".format(pauli_string, type(coef).__name__))
            if pauli_string in seen_paulis:
                raise ValueError("Duplicate Pauli term '{}' in Hamiltonian terms.".format(pauli_string))
            seen_paulis.add(pauli_string)


# -------------------------------------------------------------------------------------------
def parse_pauli_string(pauli_string: str, num_of_qubits: Optional[int] = None) -> List[Tuple[int, str]]:
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
def hamiltonian_matrix(pauli_strings: Sequence[str], coefficients: Sequence[float], num_of_qubits: int) -> np.ndarray:
    """Builds the full 2**n x 2**n matrix of H = sum_i coefficients[i] * pauli_strings[i]."""
    if len(pauli_strings) != len(coefficients):
        raise ValueError(
            "pauli_strings and coefficients must have the same length, got {} and {}."
            .format(len(pauli_strings), len(coefficients))
        )

    dim = 2 ** num_of_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for pauli_string, coef in zip(pauli_strings, coefficients):
        matrix = matrix + coef * _term_matrix(pauli_string, num_of_qubits)
    return matrix


# -------------------------------------------------------------------------------------------
def _term_matrix(pauli_string: str, num_of_qubits: int) -> np.ndarray:
    active = dict(parse_pauli_string(pauli_string, num_of_qubits))   # {qubit: 'X'|'Y'|'Z'}
    matrix = np.array([[1.0]], dtype=complex)
    for q in range(num_of_qubits):
        block = PAULI_MATRIX[active[q]] if q in active else np.eye(2, dtype=complex)
        matrix = np.kron(matrix, block)
    return matrix

