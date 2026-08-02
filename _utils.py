from typing import TYPE_CHECKING
from AriaQuanta.config import Config, get_array_module


# -------------------------------------------------------------------------------------------
if TYPE_CHECKING: import numpy as np
else: np = get_array_module(Config.use_gpu)

# -------------------------------------------------------------------------------------------
def is_unitary(matrix):
    return np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]), atol=1e-10)

# -------------------------------------------------------------------------------------------
def swap_qubits(idx1_, idx2_, num_of_qubits, multistate):

    idx1 = min(idx1_, idx2_)
    idx2 = max(idx1_, idx2_)

    indices_swaped = []
    size = 2 ** num_of_qubits
    
    # ------------------------------------------------------------
    for i in range(size):
        state_str = format(i, '0{}b'.format(num_of_qubits))

        new_state_str = state_str[:idx1] + state_str[idx2] + state_str[idx1+1:]
        new_state_str = new_state_str[:idx2] + state_str[idx1] + new_state_str[idx2+1:]
        
        index_swaped = int(new_state_str, 2)
        indices_swaped.append(index_swaped)

    multistate_swaped = multistate[indices_swaped]
    return multistate_swaped

# -------------------------------------------------------------------------------------------
def swap_qubits_density(idx1, idx2, num_of_qubits, density_matrix):

    indices_swaped = []
    size = 2 ** num_of_qubits
    
    # ------------------------------------------------------------
    for i in range(size):

        state_str = format(i, '0{}b'.format(num_of_qubits))

        new_state_str = state_str[:idx1] + state_str[idx2] + state_str[idx1+1:]
        new_state_str = new_state_str[:idx2] + state_str[idx1] + new_state_str[idx2+1:]
        
        index_swaped = int(new_state_str, 2)
        indices_swaped.append(index_swaped)

    density_matrix_swaped = density_matrix
    density_matrix_swaped = density_matrix_swaped[:,indices_swaped]
    density_matrix_swaped = density_matrix_swaped[indices_swaped,:]

    return density_matrix_swaped

# -------------------------------------------------------------------------------------------
def reorder_state(state):
    num_of_states = np.shape(state)[0]
    num_of_qubits = int(np.log2(num_of_states)) 

    bin_format = '#0' + str(num_of_qubits + 2) + 'b'                        # #05b (b stands for binay)
    all_states = [format(x, bin_format)[2:] for x in range(num_of_states)]  # Convert to binary format
    all_states = [x[::-1] for x in all_states]
    new_indices = [int(x, 2) for x in all_states]                           # binary to decimal
    reordered_state = state[new_indices]
    return reordered_state

# -------------------------------------------------------------------------------------------
def sv_to_density_matrix(statevector):
    return statevector @ statevector.conj().T


