from typing import Literal

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import X, H, CX
from AriaQuanta.aqc.measure import MeasureQubit


# -------------------------------------------------------------------------------------------
OracleType = Literal["constant_zero", "constant_one", "balanced"]

def dj(n_qubits: int, oracle: OracleType="constant_zero") -> Circuit:
    """
    Implements the Deutsch-Jozsa algorithm.
    :param n_qubits: Number of input qubits (excluding the oracle qubit).
    :param oracle: Type of oracle:
        - "constant_zero": f(x) = 0
        - "constant_one":  f(x) = 1
        - "balanced":      f(x) = parity(x)
    :return: Circuit implementing the Deutsch-Jozsa algorithm.
    """
    _check_validation(n_qubits, oracle)

    # Initialize circuit with n_qubits + 1 (oracle) qubit
    qc = Circuit(n_qubits + 1, num_of_ancilla=1)
    
    # Put the oracle qubit in |-> = (|0> - |1>)/sqrt(2)
    qc | X(n_qubits) | H(n_qubits)
    
    # Create uniform superposition over all input states.
    for qubit in range(n_qubits):
        qc | H(qubit)

    # Apply the oracle
    if oracle=="constant_zero":         # Constant oracle: f(x)=0 for all x --> U_f = I, nothing to apply
        pass
    elif oracle=="constant_one":        # Constant oracle: f(x)=1 for all x --> U_f = I^{⊗n} ⊗ X
        qc | X(n_qubits)
    elif oracle=="balanced":            # Balanced oracle: f(x)=parity(x) --> CNOT each input qubit onto the oracle qubit
        for qubit in range(n_qubits):
            qc | CX(qubit, n_qubits)
    
    else: raise TypeError(f"oracle must be in one of three states: constant_zero, constant_one, balanced")

    # Apply H^(⊗n) for interference
    for qubit in range(n_qubits):
        qc | H(qubit) | MeasureQubit([qubit])
    
    return qc


# -------------------------------------------------------------------------------------------
def _check_validation(n_qubits: int, oracle: OracleType):
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        raise TypeError(f"n_qubits must be an int, got {type(n_qubits).__name__}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")

    valid_oracles = {"constant_zero", "constant_one", "balanced"}
    if oracle not in valid_oracles:
        raise ValueError(f"oracle must be one of: {', '.join(sorted(valid_oracles))} - got {oracle!r}")
