
from AriaQuanta.backend.simulator import Simulator
from AriaQuanta.aqc.gatelibrary import H, Sdg
import numpy as np
import re

#------------------------------------------------------------------------------------
class Hamiltonian:
    def __init__(self, terms):
        """
        terms: List of (coefficient, Pauli operator)
        Example: H = 0.5 * Z1 + 0.3 * X2 * Z3
        terms = [("Z1", 0.5), ("X2Z3", 0.3)]
        """
        self.terms = terms   # [("Z1", 0.5), ("X2Z3", 0.3)]
        terms_dict = dict(self.terms)  
        
        self.paulis = list(terms_dict.keys()) # paulis:  ['Z1', 'X2Z3']
        self.coefs = np.array(list(terms_dict.values()))  # [0.5 0.3]

#------------------------------------------------------------------------------------
def pauli_transform_circuit(circuit, pauli_string):
    # example of pauli string: "Z0X1"

    circuit_copy = circuit.copy()

    numbers = re.findall(r'(?<=[XYZ])\d+', pauli_string)
    xyzi = re.findall(r"[XYZI]", pauli_string)

    for n in range (len(xyzi)):
        if xyzi[n] == 'X':
            circuit | H(numbers[n])    
        elif xyzi[n] == 'Y':
            circuit | Sdg(numbers[n])
            circuit | H(numbers[n])            
        elif (xyzi[n] == 'Z') or (xyzi[n] == 'I'):
            continue
        else:
            raise Exception('{} is not a puali matrix'.format(xyzi[n]))
           
    return circuit_copy

#------------------------------------------------------------------------------------
def find_expectation_value(circuit, hamiltonian, num_of_iter_measure):
        
    pauli_exp_value = {}
    total_energy = 0

    paulis = hamiltonian.paulis
    coefs = hamiltonian.coefs

    #------------------------------------------
    # parity
    num_of_qubits = circuit.num_of_qubits
    num_of_states = 2**num_of_qubits
    bin_format = '#0' + str(num_of_qubits + 2) + 'b' # #05b
    all_states = [format(x, bin_format)[2:] for x in range(num_of_states)]

    parity = []
    for this_state in all_states:
        this_parity = np.sum(np.array([int(item) for item in this_state]))
        this_parity = (-1) ** (this_parity % 2)
        parity.append(this_parity)
    parity = np.array(parity)    
    # print(parity)

    #------------------------------------------
    # run circuit once for only 'Z'

    sim = Simulator()
    circuit_update = circuit.copy()
    result = sim.simulate(circuit_update, num_of_iter_measure, 4)
    counts_z, probability_z = result.count()  
        
    #------------------------------------------
    idx = 0
    for pauli_string in paulis:
        if pauli_string == 'I':
            this_exp_value = 1
        else:  
            if ('X' in pauli_string) or ('Y' in pauli_string): 
                circuit_update = pauli_transform_circuit(circuit, pauli_string)
            
                # run circuit for num_of_iter_measure times
                sim = Simulator()
                result = sim.simulate(circuit_update, num_of_iter_measure, 4)
                counts, probability = result.count()    

                probability_values = np.array(list(probability.values()))
                this_exp_value = np.sum(probability_values*parity)
            else:    
                probability_values = np.array(list(probability_z.values()))
                this_exp_value = np.sum(probability_values*parity)

        pauli_exp_value[pauli_string] = this_exp_value*coefs[idx]
        total_energy += this_exp_value*coefs[idx]
        idx += 1

    # example: 
    # {'I': np.float64(0.0), 'Z0': np.float64(0.0), 'Z1': np.float64(0.2), 'Z0Z1': np.float64(0.4), 'X0X1': np.float64(-0.4)}
    #  0.2
    return pauli_exp_value, total_energy



"""
#---------------------------------------
# <p> = sum(p_b * (-1)**(parity))
# find parity 
probability_keys = [x[1:-1] for x in probability.keys()]
parity = []
for item in probability_keys:
    x = sum(int(digit) for digit in item)
    parity.append(x%2)
parity = np.array(parity)  
# find <p>
probability_values = np.fromiter(probability.values(), dtype=float)
#probability_ave = sum(probability_values * (-1)**(parity))
expectation_value = sum(probability_values * (-1)**(parity))
#print("probability_ave = ", probability_ave)

return expectation_value, counts

# Compute expectation value
#expectation_value = 0
#for bitstring, prob in probability.items():
#    parity = sum(int(digit) for digit in bitstring[1:-1]) % 2
#    expectation_value += prob * (-1) ** parity
#print("expectation_value =", expectation_value)
#return expectation_value
"""