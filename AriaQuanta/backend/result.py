from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from AriaQuanta._utils import np, reorder_state, sv_to_density_matrix


# -------------------------------------------------------------------------------------------
class Result():
    def __init__(self, statevector_all: List[np.ndarray], num_of_qubits: int, num_of_ancilla: int, measurequbit_values_all: List[Dict[str, str]]) -> None:

        if not statevector_all:
            raise ValueError("'statevector_all' must contain at least one statevector.")
        if num_of_qubits < 1:
            raise ValueError("'num_of_qubits' must be at least 1, got {}.".format(num_of_qubits))
        if not (0 <= num_of_ancilla <= num_of_qubits):
            raise ValueError("'num_of_ancilla' must be between 0 and num_of_qubits ({}), got {}.".format(num_of_qubits, num_of_ancilla))
        if len(measurequbit_values_all) != len(statevector_all):
            raise ValueError(
                "'measurequbit_values_all' must have one entry per shot: expected {}, got {}."
                .format(len(statevector_all), len(measurequbit_values_all))
            )

        self.statevector_all = statevector_all
        self.num_of_qubits   = num_of_qubits
        self.num_of_ancilla  = num_of_ancilla
        self.measurequbit_values_all = measurequbit_values_all


    # ------------------------------------------------------------
    def count(self, clbits: Optional[List[str]]=None) -> Tuple[Dict[str, int], Dict[str, float]]:
        if clbits is None:
            keys = list(self.measurequbit_values_all[0].keys()) or ['q' + str(i) for i in range(self.num_of_qubits)]
        else:
            keys = clbits
        select_idx = sorted(int(k[1:]) for k in keys)        # 'qN' -> N

        bin_format = '#0' + str(self.num_of_qubits + 2) + 'b'
        num_of_iterations = len(self.statevector_all)

        measurement_all: List[str] = []
        for i in range(num_of_iterations):
            recorded = self.measurequbit_values_all[i]

            if all(recorded.get('q' + str(q), '') != '' for q in select_idx):
                bits = ''.join(recorded['q' + str(q)] for q in select_idx)
            else:
                probabilities = np.abs(self.statevector_all[i].flatten()) ** 2
                probabilities = probabilities / probabilities.sum()
                outcome = np.random.choice(len(probabilities), p=probabilities)
                bitstring = format(outcome, bin_format)[2:]
                bits = ''.join(bitstring[q] for q in select_idx)

            measurement_all.append('|' + bits + '>')

        counts = Counter(measurement_all)

        # fill in every possible outcome of the selected bits, even the ones that never occurred
        measure_size_qubit = len(select_idx)
        select_bin_format = '#0' + str(measure_size_qubit + 2) + 'b'
        all_states = ['|' + format(x, select_bin_format)[2:] + '>' for x in range(2 ** measure_size_qubit)]
        for state_i in all_states:
            counts.setdefault(state_i, 0)

        sorted_states = sorted(all_states)                   # computed once, reused for both dicts
        counts_sorted = {key: counts[key] for key in sorted_states}
        probability = {key: counts[key] / num_of_iterations for key in sorted_states}

        return counts_sorted, probability

    # ------------------------------------------------------------
    def statevector_all_measured(self) -> List[np.ndarray]:
        measurequbit_values_all = self.measurequbit_values_all

        bin_format = '#0' + str(self.num_of_qubits + 2) + 'b'
        all_states = [format(x, bin_format)[2:] for x in range(2 ** self.num_of_qubits)]

        output_statevector_all_reduce: List[np.ndarray] = []
        for statevector_i, measurequbit_values_i in zip(self.statevector_all, measurequbit_values_all):
            result_indices = [idx for idx, s in enumerate(all_states) if all(s[int(k[1:])] == str(v) for k, v in measurequbit_values_i.items())]
            output_statevector_all_reduce.append(statevector_i[result_indices])

        return output_statevector_all_reduce


    # ------------------------------------------------------------
    @property
    def statevector_all_no_ancilla(self) -> List[np.ndarray]:
        num_of_ancilla = self.num_of_ancilla

        state_remaining_all: List[np.ndarray] = []
        for statevector_i in self.statevector_all:
            state_reorder = reorder_state(statevector_i)

            for _ in range(num_of_ancilla):                 # was `for i in ...`, shadowing the outer loop
                size_state_reorder = int(np.shape(state_reorder)[0] / 2)
                state_reorder = state_reorder[:size_state_reorder]

            state_remaining = reorder_state(state_reorder)
            state_remaining_all.append(state_remaining)

        return state_remaining_all

    @property
    def density_matrix_all(self) -> List[np.ndarray]:
        return [sv_to_density_matrix(sv) for sv in self.statevector_all]



# -------------------------------------------------------------------------------------------
class ResultDensity:
    def __init__(self, density_matrix_all: List[np.ndarray]) -> None:
        if not density_matrix_all:
            raise ValueError("'density_matrix_all' must contain at least one density matrix.")

        dim = density_matrix_all[0].shape[0]
        for i, dm in enumerate(density_matrix_all):
            if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
                raise ValueError("density_matrix_all[{}] must be a square matrix, got shape {}.".format(i, dm.shape))
            if dm.shape[0] != dim:
                raise ValueError("All density matrices must share the same dimension.")
        if dim & (dim - 1) != 0:
            raise ValueError("Density matrix dimension must be a power of two, got {}.".format(dim))

        self.density_matrix_all = density_matrix_all
        self.num_of_qubits = int(np.log2(dim))

    # ------------------------------------------------------------
    def purity(self, index: int=0) -> float:
        # Tr(rho^2): 1.0 for a pure state, < 1.0 the more mixed the state is
        dm = self.density_matrix_all[index]
        return float(np.real(np.trace(dm @ dm)))

    # ------------------------------------------------------------
    def probabilities(self, index: int=0) -> Dict[str, float]:
        diag = np.real(np.diag(self.density_matrix_all[index]))
        bin_format = '#0' + str(self.num_of_qubits + 2) + 'b'
        all_states = ['|' + format(x, bin_format)[2:] + '>' for x in range(2 ** self.num_of_qubits)]
        return {state: float(p) for state, p in zip(all_states, diag)}

    # ------------------------------------------------------------
    @property
    def mean_density_matrix(self) -> np.ndarray:
        return sum(self.density_matrix_all) / len(self.density_matrix_all)



# -------------------------------------------------------------------------------------------
def plot_histogram(counter: Dict[str, Any], 
                   ax: Optional[Axes]=None, title: Optional[str]=None, 
                   color: str='tab:blue', figsize: Tuple[float, float]=(8, 5)) -> Tuple[Figure, Axes]:

    if not counter:
        raise ValueError("'counter' must contain at least one entry.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 14, 'axes.linewidth': 1.5}):
        ax.bar(list(counter.keys()), list(counter.values()), color=color)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Count')
        if title:
            ax.set_title(title)

    return fig, ax


