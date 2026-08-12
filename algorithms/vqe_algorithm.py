from typing import List, Optional, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.algorithms.eigen_solver import Hamiltonian, find_expectation_value
from scipy.optimize import minimize, OptimizeResult


# -------------------------------------------------------------------------------------------
class VQE(): #, threshold, params_dict):
    def __init__(self, ansatz: Ansatz, hamiltonian: Hamiltonian, num_of_iter_measure: int,
                initial_values: Union[List[float], np.ndarray], optimizer: str='COBYLA') -> None:
        """
        Variational Quantum Eigensolver: finds the parameter values that minimize
        <ansatz(params)|hamiltonian|ansatz(params)>, using a classical optimizer.

        :param ansatz: Parametrized Ansatz whose parameters will be optimized.
        :param hamiltonian: Hamiltonian whose expectation value is minimized.
        :param num_of_iter_measure: Number of measurement shots per cost-function evaluation.
        :param initial_values: Starting parameter values, in the order of ansatz.params_names.
        :param optimizer: scipy.optimize.minimize method name (default 'COBYLA').
        """
        
        self._check_validation(ansatz, hamiltonian, num_of_iter_measure, optimizer)

        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.num_of_iter_measure = num_of_iter_measure
        self.initial_values = initial_values
        self.optimizer = optimizer

        self.params_all: List[np.ndarray] = []
        self.final_params: Optional[np.ndarray] = None

        self.energy_all: List[float] = []
        self.final_energy: Optional[float] = None

        self.result: Optional[OptimizeResult] = None


    # ------------------------------------------------------------
    def cost_function(self, params_values: Union[List[float], np.ndarray]) -> float:
        """
        Evaluates <ansatz(params_values)|hamiltonian|ansatz(params_values)>, and records
        the (params, energy) pair in self.params_all / self.energy_all for later inspection.

        :param params_values: Parameter values to bind to the ansatz for this evaluation.
        :return: The estimated total energy.
        """
        self.ansatz.set_params_values(params_values)

        _, total_energy = find_expectation_value(self.ansatz, self.hamiltonian, self.num_of_iter_measure)

        # Copy before storing: the optimizer may reuse/mutate the same array across calls.
        self.params_all.append(np.array(params_values, dtype=float))
        self.energy_all.append(total_energy)
        
        # self.final_params = params_values
        # self.final_energy = total_energy

        return total_energy
    
    # ------------------------------------------------------------
    def find_minimize(self) -> OptimizeResult:
        """
        Runs the classical optimizer to minimize cost_function starting from initial_values.
        Sets self.result, self.final_params (= result.x) and self.final_energy (= result.fun).

        :return: The scipy OptimizeResult.
        """
        result = minimize(self.cost_function, self.initial_values, method=self.optimizer)

        self.result = result
        # Use the optimizer's own best point/value, not just whatever cost_function was last
        # called with -- for several methods (e.g. COBYLA, the default) the last evaluated
        # point during the search is not guaranteed to be the best one found.
        self.final_params = result.x
        self.final_energy = result.fun

        return result
    

    # ------------------------------------------------------------
    @property
    def _check_validation(ansatz, hamiltonian, num_of_iter_measure, optimizer):
        if not isinstance(ansatz, Ansatz):
            raise TypeError("'ansatz' must be an Ansatz instance, got {}.".format(type(ansatz).__name__))
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("'hamiltonian' must be a Hamiltonian instance, got {}.".format(type(hamiltonian).__name__))
        if not isinstance(num_of_iter_measure, int) or isinstance(num_of_iter_measure, bool):
            raise TypeError("'num_of_iter_measure' must be an int, got {}.".format(type(num_of_iter_measure).__name__))
        if num_of_iter_measure < 1:
            raise ValueError("'num_of_iter_measure' must be at least 1, got {}.".format(num_of_iter_measure))

        initial_values = np.asarray(initial_values, dtype=float).flatten()
        if initial_values.size != len(ansatz.params_names):
            raise ValueError( "'initial_values' must have length {} (= number of ansatz parameters), got {}.".format(len(ansatz.params_names), initial_values.size) )
        if not isinstance(optimizer, str) or not optimizer:
            raise TypeError("'optimizer' must be a non-empty string, got {}.".format(type(optimizer).__name__))
