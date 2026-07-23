

from AriaQuanta.algorithms.eigen_solver import find_expectation_value
from scipy.optimize import minimize
import numpy as np

#------------------------------------------------------------------------------------
class VQE(): #, threshold, params_dict):
    def __init__(self, ansatz, hamiltonian, num_of_iter_measure, initial_values, optimizer='COBYLA'):

        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.num_of_iter_measure = num_of_iter_measure
        self.initial_values = initial_values
        self.optimizer = optimizer

        self.params_all = []
        self.final_params = None

        self.energy_all = []
        self.final_energy = None

    #--------------------------------------------
    def cost_function(self, params_values):

        ansatz = self.ansatz
        hamiltonian = self.hamiltonian
        num_of_iter_measure = self.num_of_iter_measure

        ansatz.set_params_values(params_values)

        pauli_exp_value, total_energy = find_expectation_value(ansatz, hamiltonian, num_of_iter_measure)

        self.params_all.append(params_values)
        self.energy_all.append(total_energy)

        self.final_params = params_values
        self.final_energy = total_energy

        # print("energy= ", total_energy, ", params= ", params_values) 

        return total_energy
        
    def find_minimize(self):
        result = minimize(self.cost_function, self.initial_values, method=self.optimizer)
        return result

#-------------------------------------------

# optimization with scipy
#from scipy.optimize import minimize
#fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
#result = minimize(fun, [0,1], method='COBYLA')
#print(result.x)   