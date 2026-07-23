
import numpy as np
from AriaQuanta.algorithms import VQE, Hamiltonian, find_expectation_value
from AriaQuanta.aqc.gatelibrary import CP, H, RX, CX, RZ
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.backend import Simulator

#------------------------------------------------------------------------------------
class QAOA(VQE):
    def __init__(self, graph, n_layers, num_of_iter_measure, initial_values, optimizer='COBYLA'):
        
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
        
    def max_cut(self):
        self.find_minimize()
        final_energy = self.final_energy
        final_params = self.final_params
        self.ansatz.set_params_values(final_params)
        
        sim = Simulator()
        result = sim.simulate(self.ansatz, self.num_of_iter_measure, 4)
        counts, probability = result.count() 

        max_key = max(probability, key=probability.get)  # Key with the maximum value
        max_value = probability[max_key]                 # Maximum value

        result = {max_key: max_value}
        return result, counts


#------------------------------------------------------------------------
def GraphHamiltonian(graph):  
    graph = graph
    pauli_list = []
    for edge in list(graph.edge_list()):
        #paulis = ["I"] * len(graph)
        #paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
        pauli1 = "Z" + str(edge[0])
        pauli2 = "Z" + str(edge[1])
        paulis = [pauli1, pauli2]
        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis), weight))
    hamiltonian = Hamiltonian(pauli_list)        
    return(hamiltonian)   

#------------------------------------------------------------------------
def cost_hamiltonian(graph, ansatz, gamma):
    """Applies cost unitary U_C based on the problem graph."""
    qubits = list(range(ansatz.num_of_qubits))
    for u, v in graph.edge_list():
            #ansatz | CP(gamma, qubits[u], qubits[v])
            ansatz | CX(qubits[u], qubits[v])
            ansatz | RZ(gamma, qubits[v])            
            ansatz | CX(qubits[u], qubits[v])

#------------------------------------------------------------------------
def mixer_hamiltonian(ansatz, beta):
    """Applies mixer unitary U_B."""
    qubits = list(range(ansatz.num_of_qubits))
    for q in qubits:
        ansatz | RX(beta, q)   #(q, 2 * beta) 
             
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
def GraphAnsatz(graph, n_layers):
    num_qubits = graph.num_nodes()
    beta_names = []
    gamma_names = []
    for i in range(n_layers):
        beta_names.append('beta'+str(i))
        gamma_names.append('gamma'+str(i))  

    params_names = beta_names + gamma_names
    # params = np.random.uniform(low=0, high=np.pi, size=(p*2,))
    ansatz = Ansatz(num_qubits, params_names)

    """Constructs full QAOA circuit with given parameters."""
    # beta = params[:p]
    # gamma = params[p:]
            
    # Initialize in equal superposition
    qubits = list(range(ansatz.num_of_qubits))
    for q in qubits:
        ansatz | H(q)
    
    # Apply alternating Hamiltonians
    for i in range(n_layers):
        cost_hamiltonian(graph, ansatz, gamma_names[i])
        mixer_hamiltonian(ansatz, beta_names[i])  

    return ansatz              