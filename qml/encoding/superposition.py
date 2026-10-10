from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import H


# -------------------------------------------------------------------------------------------
def uniform_superposition(num_of_qubits: int) -> Circuit:
    """
    Uniform superposition: a Hadamard on every qubit, preparing
    |s> = H^(x)n |0...0> = (1 / sqrt(2**n)) * sum_x |x>  -- every computational basis state
    with the same amplitude.

    Unlike the other encodings, this one takes NO classical data -- it's the data-independent
    starting state many algorithms begin from (Grover, QAOA, the address register of a QRAM
    lookup -- see qram_encoding). It lives next to the encodings because it is the simplest
    possible "state preparation" circuit and is typically the first block of a pipeline that
    then loads data on top of it.

    :param num_of_qubits: Number of qubits.
    :return: A new Circuit with H applied on every qubit (not yet run).
    """
    if not isinstance(num_of_qubits, int) or isinstance(num_of_qubits, bool):
        raise TypeError("'num_of_qubits' must be an int, got {}.".format(type(num_of_qubits).__name__))
    if num_of_qubits < 1:
        raise ValueError("'num_of_qubits' must be at least 1, got {}.".format(num_of_qubits))

    qc = Circuit(num_of_qubits)
    for q in range(num_of_qubits):
        qc | H(q)
    return qc
