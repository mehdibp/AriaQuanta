import math
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import H, X, Z, CZ, CNZ


# -------------------------------------------------------------------------------------------
def oracle(qc, target_state):
    """
    Oracle for Grover's Algorithm.
    
    Parameters:
        circuit: QuantumCircuit object.
        target_state: Binary string of target state.
    """
    n = len(target_state)
    
    # Apply X gates on qubits where target_state is '0'
    for i, bit in enumerate(target_state):  #reversed(target_state)):
        if bit == '0':
            qc | X(i)
    
    # Apply multi-controlled Z gate
    if n == 1:
        qc | Z(0)
    elif n == 2:
        qc | CZ([0], [1])
    else:
        qc | CNZ(n, n-1)
    #    circuit.ccz(*range(n))

    #controls = list(range(n-1))
    #target = n
    #controlled_n_z(qc, controls, target)

    # Undo the X gates
    for i, bit in enumerate(target_state): #reversed(target_state)):
        if bit == '0':
            qc | X(i)


# -------------------------------------------------------------------------------------------
def diffusion_operator(qc, n):
    """
    Implements the Grover Diffusion Operator.
    
    Parameters:
        circuit: QuantumCircuit object.
        n: Number of qubits.
    """
    # Apply Hadamard gates to all qubits
    for i in range(n):
        qc | H(i)
    
    # Apply X gates to all qubits
    for i in range(n):
        qc | X(i)
    
    if n == 1:
        qc | Z(0)
    elif n == 2:
        qc | CZ([0], [1])
    else:
        qc | CNZ(n, n-1)

    # Apply multi-controlled Z gate
    #controls = list(range(n-1))
    #target = n
    #controlled_n_z(qc, controls, target)

    # Apply X gates to all qubits
    for i in range(n):
        qc | X(i)
    
    # Apply Hadamard gates to all qubits
    for i in range(n):
        qc | H(i)

# -------------------------------------------------------------------------------------------
def grover(n, target_state):
    """
    Implements Grover's Algorithm.
    
    Parameters:
        circuit: QuantumCircuit object.
        oracle: Function implementing the oracle.
        n: Number of qubits.
        target_state: String representing the target state (e.g., "101").
    """
    # Step 1: Initialization
    qc = Circuit(n)

    # Apply Hadamard to all qubits
    for i in range(n):
        qc | H(i)
    
    # Calculate the number of iterations (π/4 * √N)
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
