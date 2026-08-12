import math
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import H, X, Z, CZ, CNZ


# -------------------------------------------------------------------------------------------
def oracle(qc: Circuit, target_state: str) -> None:
    """
    Oracle for Grover's Algorithm: flips the phase of |target_state>.

    :param qc: Circuit to apply the oracle to (modified in place).
    :param target_state: Binary string of the target state, e.g. '101'.
    """
    n = len(target_state)

    # Apply X gates on qubits where target_state is '0'
    for i, bit in enumerate(target_state):
        if bit == '0':
            qc | X(i)

    # Apply multi-controlled Z gate across all n qubits
    if   n == 1: qc | Z(0)
    elif n == 2: qc | CZ(0, 1)
    else:        qc | CNZ(n, n-1)

    # Undo the X gates
    for i, bit in enumerate(target_state):
        if bit == '0':
            qc | X(i)

# -------------------------------------------------------------------------------------------
def diffusion_operator(qc: Circuit, n: int) -> None:
    """
    Implements the Grover diffusion operator (inversion about the mean) on all n qubits.

    :param qc: Circuit to apply the diffusion operator to (modified in place).
    :param n: Number of qubits.
    """
    # Apply Hadamard gates to all qubits
    for i in range(n):
        qc | H(i)

    # Apply X gates to all qubits
    for i in range(n):
        qc | X(i)

    # Apply multi-controlled Z gate across all n qubits
    if   n == 1: qc | Z(0)
    elif n == 2: qc | CZ(0, 1)
    else:        qc | CNZ(n, n - 1)

    # Apply X gates to all qubits
    for i in range(n):
        qc | X(i)

    # Apply Hadamard gates to all qubits
    for i in range(n):
        qc | H(i)

# -------------------------------------------------------------------------------------------
def grover(n: int, target_state: str) -> Circuit:
    """
    Builds Grover's Algorithm circuit that amplifies the amplitude of |target_state>.

    Note: for n == 1 (a 2-state search space), no integer number of Grover iterations
    reaches certainty -- this is an inherent limitation of the algorithm for N = 2,
    not specific to this implementation.

    :param n: Number of qubits.
    :param target_state: Binary string of the target state (must have length n), e.g. '101'.
    :return: The built Circuit (not yet run).
    """
    _check_validation(n, target_state)
    
    # Step 1: Initialization
    qc = Circuit(n)

    # Apply Hadamard to all qubits
    for i in range(n):
        qc | H(i)

    # Calculate the number of iterations (pi/4 * sqrt(N))
    iterations = int(math.pi / 4 * math.sqrt(2**n))

    for _ in range(iterations):
        # Step 2: Oracle
        oracle(qc, target_state)

        # Step 3: Diffusion Operator
        diffusion_operator(qc, n)

    # Step 4: Measurement
    # qc.run()
    # measurement, measurement_index, probabilities = qc.measure_all()
    
    return qc #, measurement, measurement_index, probabilities


# -------------------------------------------------------------------------------------------
def _check_validation(n: int, target_state: str):
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")
    if not isinstance(target_state, str) or not target_state:
        raise TypeError(f"target_state must be a non-empty string, got {type(target_state).__name__}")
    if len(target_state) != n:
        raise ValueError(f"target_state must have length n ({n}), got length {len(target_state)} ('{target_state}')")
    if any(bit not in ('0', '1') for bit in target_state):
        raise ValueError(f"target_state must only contain '0'/'1' characters, got '{target_state}'")
