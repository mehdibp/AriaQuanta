
'''
print(" __________________________________")
print("|                                  |")
print("|            AriaQuanta            |")
print("|                                  |")
print(" __________________________________")
'''
#-----------------------
# density matrix:
# 1. gate: apply_gate
# 2. circuit: self.density = True
#             self.density_matrix = True
# 3. job: apply vs apply_density

import numpy as np
np.set_printoptions(suppress=True)      

# from ._utils import(
#    is_unitary,
#    swap_qubits,
#    swap_qubits_density,
#    reorder_state
#)

from ._version import *

#from .config import(
#    Config,
#    get_array_module
#)