from AriaQuanta._utils import np


# -------------------------------------------------------------------------------------------
class GateControlN:
    def __init__(self, name, matrix, target_qubits):

        self.name = name
        matrix = np.asarray(matrix)
        self.matrix = matrix

        target_qubits = [target_qubits]
        target_qubits = np.asarray(target_qubits, dtype=int).flatten()
        self.target_qubits = target_qubits

        self.qubits = target_qubits.tolist()

    def apply(self, num_of_qubits, multistate):
        
        target_qubits = self.target_qubits
        matrix = self.matrix

        #-----------------------------------------------------
        num_of_target_qubits = np.shape(target_qubits)[0]

        multistate_swaped = multistate
        #for k1 in range(0, num_of_target_qubits):
        #    multistate_swaped = swap_qubits(k1, target_qubits[k1], num_of_qubits, multistate_swaped)
        #    print("1, 2 = ", multistate_swaped)

        #if (num_of_qubits - num_of_target_qubits) > 0:
        #    dim = 2 ** (num_of_qubits - num_of_target_qubits)
        #    I2 = np.identity(dim, dtype=complex)
        #else:
        #    I2 = 1         
    
        full_matrix = matrix # np.kron(matrix, I2)
        
        multistate_swaped = np.dot(full_matrix, multistate_swaped)
        #print("3 = ",multistate_swaped)

        #for k1 in reversed(range(0, num_of_target_qubits)):
        #    multistate_swaped = swap_qubits(target_qubits[k1], k1, num_of_qubits, multistate_swaped)
        #    print("4, 5 = ",multistate_swaped)

        multistate = multistate_swaped

        return multistate
    
    """
    def apply_density(self, num_of_qubits, density_matrix):
        
        target_qubits = self.target_qubits
        matrix = self.matrix
        
        #-----------------------------------------------------
        num_of_target_qubits = np.shape(target_qubits)[0]

        density_matrix_swaped = density_matrix
        for k1 in range(0, num_of_target_qubits):
            density_matrix_swaped = swap_qubits_density(k1, target_qubits[k1], num_of_qubits, density_matrix_swaped)

        if (num_of_qubits - num_of_target_qubits) > 0:
            dim = 2 ** (num_of_qubits - num_of_target_qubits)
            I2 = np.identity(dim, dtype=complex)
        else:
            I2 = 1         
    
        full_matrix = np.kron(matrix, I2)
        
        density_matrix_swaped = full_matrix @ density_matrix_swaped @ np.conj(full_matrix.T)

        for k1 in reversed(range(0, num_of_target_qubits)):
            density_matrix_swaped = swap_qubits_density(target_qubits[k1], k1, num_of_qubits, density_matrix_swaped)

        density_matrix = density_matrix_swaped

        return density_matrix  
    """    
    
# -------------------------------------------------------------------------------------------
class CNZ(GateControlN):
    def __init__(self, num_of_qubits, target_qubits=0):
        size = 2 ** num_of_qubits
        matrix = np.identity(size, dtype=complex)

        if target_qubits != (num_of_qubits-1):
            raise("Error! target_qubit has to be the last qubit")
        target_qubits = num_of_qubits-1
        
        # ***for CNZ target_qubits has to be only 1 qubit***
        # *** note! now works only for the target to be the last qubit***
        matrix[size-1, size-1] = -1

        super().__init__(name='CNZ', matrix=matrix, target_qubits=target_qubits)
