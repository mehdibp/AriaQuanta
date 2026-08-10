from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import X, H, CX


# -------------------------------------------------------------------------------------------
def dj(n_qubits: int, is_constant: bool=True) -> Circuit:
    """
    Implements the Deutsch-Jozsa algorithm for a given number of qubits.
    :param n_qubits: Number of qubits in the input (not counting the oracle qubit)
    :param is_constant: Boolean flag to decide whether the oracle represents a constant or balanced function
    :return: Measurement result indicating if the function is constant or balanced
    """

    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool):
        raise TypeError(f"n_qubits must be an int, got {type(n_qubits).__name__}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits}")
    if not isinstance(is_constant, bool):
        raise TypeError(f"is_constant must be a bool, got {type(is_constant).__name__}")

    # Initialize circuit with n_qubits + 1 (oracle) qubit
    qc = Circuit(n_qubits + 1, num_of_ancilla=1)
    
    # Put the oracle qubit in |-> = (|0> - |1>)/sqrt(2)
    qc | X(n_qubits) | H(n_qubits)
    
    # Apply Hadamard gate to all input qubits
    for qubit in range(n_qubits):
        qc | H(qubit)

    # Apply the oracle
    if is_constant:     # Constant oracle: f(x) = 0 for all x -> identity, nothing to apply
        pass
    else:               # Balanced oracle: f(x) = parity(x) -> CNOT each input qubit onto the oracle qubit
        for qubit in range(n_qubits):
            qc | CX(qubit, n_qubits)

    # Apply Hadamard gate again to all input qubits
    for qubit in range(n_qubits):
        qc | H(qubit)
    
    return qc


