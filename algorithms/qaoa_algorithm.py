from typing import Any, Dict, List, Tuple, Union

from AriaQuanta._utils import np
from AriaQuanta.algorithms import VQE, Hamiltonian
from AriaQuanta.aqc.gatelibrary import H, RX, CX, RZ
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.backend import Simulator


# -------------------------------------------------------------------------------------------
class QAOA(VQE):
    def __init__(self, graph: Any, n_layers: int, num_of_iter_measure: int,
            initial_values: Union[List[float], np.ndarray], optimizer: str='COBYLA') -> None:

        self._check_validation(graph, n_layers, num_of_iter_measure)
         
        self.graph = graph
        self.n_layers = n_layers
        self.hamiltonian = GraphHamiltonian(self.graph)
        self.ansatz = GraphAnsatz(self.graph, self.n_layers)
        self.num_of_iter_measure = num_of_iter_measure
        self.initial_values = initial_values
        self.optimizer = optimizer

        super().__init__(ansatz=self.ansatz, hamiltonian=self.hamiltonian, 
                         num_of_iter_measure=self.num_of_iter_measure, 
                         initial_values=self.initial_values, optimizer=self.optimizer)


    # ------------------------------------------------------------
    def max_cut(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.find_minimize()
        final_params = self.final_params
        self.ansatz.set_params_values(final_params)

        sim = Simulator()
        result = sim.simulate(self.ansatz, self.num_of_iter_measure, 4, show_progress=False)
        counts, probability = result.count()

        max_key = max(probability, key=probability.get)  # Key with the maximum value
        max_value = probability[max_key]                 # Maximum value

        result = {max_key: max_value}
        return result, counts

    # ------------------------------------------------------------
    @property
    def _check_validation(graph: Any, n_layers: int, num_of_iter_measure: int):
        if not isinstance(n_layers, int) or isinstance(n_layers, bool):
            raise TypeError("'n_layers' must be an int, got {}.".format(type(n_layers).__name__))
        if n_layers < 1:
            raise ValueError("'n_layers' must be at least 1, got {}.".format(n_layers))
        if not isinstance(num_of_iter_measure, int) or isinstance(num_of_iter_measure, bool):
            raise TypeError("'num_of_iter_measure' must be an int, got {}.".format(type(num_of_iter_measure).__name__))
        if num_of_iter_measure < 1:
            raise ValueError("'num_of_iter_measure' must be at least 1, got {}.".format(num_of_iter_measure))
        if graph.num_nodes() < 1:
            raise ValueError("'graph' must have at least one node.")

        initial_values = np.asarray(initial_values, dtype=float).flatten()
        if initial_values.size != 2 * n_layers:
            raise ValueError( "'initial_values' must have length 2 * n_layers ({}), got {}.".format(2 * n_layers, initial_values.size) )
    

# -------------------------------------------------------------------------------------------
def GraphHamiltonian(graph: Any) -> Hamiltonian:
    """
    Builds the MaxCut cost Hamiltonian H = sum_{(u,v) in E} w_uv * Z_u * Z_v for a graph.

    Minimizing <H> maximizes the total weight of cut edges, since Z_u*Z_v = -1 when u and v
    are assigned to different sides of the cut, and +1 when they're on the same side.

    :param graph: rustworkx.PyGraph-style graph (see QAOA docstring).
    :return: The corresponding Hamiltonian.
    """
    pauli_list = []
    for u, v in graph.edge_list():
        pauli_string = "Z{}Z{}".format(u, v)
        weight = graph.get_edge_data(u, v)
        pauli_list.append((pauli_string, weight))

    return Hamiltonian(pauli_list)

# -------------------------------------------------------------------------------------------
def cost_hamiltonian(graph: Any, ansatz: Ansatz, gamma: str) -> None:
    """Applies cost unitary U_C based on the problem graph.

    :param graph: rustworkx.PyGraph-style graph (see QAOA docstring).
    :param ansatz: Ansatz to append gates to.
    :param gamma: Name of the registered parameter to use for every edge's rotation.
    """
    qubits = list(range(ansatz.num_of_qubits))
    for u, v in graph.edge_list():
        ansatz | CX(qubits[u], qubits[v])
        ansatz | RZ(gamma, qubits[v])
        ansatz | CX(qubits[u], qubits[v])

# -------------------------------------------------------------------------------------------
def mixer_hamiltonian(ansatz: Ansatz, beta: str) -> None:
    """
    Appends the mixer unitary U_B(beta) for one QAOA layer to ansatz (in place).

    :param ansatz: Ansatz to append gates to.
    :param beta: Name of the registered parameter to use for every qubit's rotation.
    """
    qubits = list(range(ansatz.num_of_qubits))
    for q in qubits:
        ansatz | RX(beta, q)

# -------------------------------------------------------------------------------------------
def GraphAnsatz(graph: Any, n_layers: int) -> Ansatz:
    """
    Builds the standard QAOA ansatz for a graph: an equal superposition, followed by
    n_layers alternations of the cost unitary and the mixer unitary.

    Parameters are named 'beta0'..'beta{n_layers-1}', 'gamma0'..'gamma{n_layers-1}',
    in that order -- matching the order expected by QAOA.initial_values.

    :param graph: rustworkx.PyGraph-style graph (see QAOA docstring).
    :param n_layers: Number of QAOA layers.
    :return: The parametrized Ansatz (not yet bound to numeric parameter values).
    """
    num_qubits  = graph.num_nodes()
    beta_names  = ['beta'  + str(i) for i in range(n_layers)]
    gamma_names = ['gamma' + str(i) for i in range(n_layers)]

    params_names = beta_names + gamma_names
    ansatz = Ansatz(num_qubits, params_names)

    # Initialize in equal superposition
    qubits = list(range(ansatz.num_of_qubits))
    for q in qubits:
        ansatz | H(q)

    # Apply alternating cost/mixer unitaries
    for i in range(n_layers):
        cost_hamiltonian(graph, ansatz, gamma_names[i])
        mixer_hamiltonian(ansatz, beta_names[i])

    return ansatz
