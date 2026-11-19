from typing import Callable, List, Optional, Sequence, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.aqc.hamiltonian import Hamiltonian
from AriaQuanta.algorithms.eigen_solver import find_expectation_value
from AriaQuanta.qml.gradients.parameter_shift import parameter_shift_jacobian
from AriaQuanta.qml._shared import validate_X


# -------------------------------------------------------------------------------------------
# Quantum Neural Network -- the "model" half of AriaQuanta.qml.models, with no notion of a
# loss, an optimizer or training. Same role (and same layering) as Qiskit Machine Learning's
# EstimatorQNN versus its NeuralNetworkClassifier / VQC:
#
#   QNN  (this file)  -- a function  (x, weights) -> K expectation values, plus its exact
#                        Jacobian d(output)/d(weights) by the parameter-shift rule.
#   VQC  (vqc.py)     -- wraps a QNN with an OutputMap, a Loss and an Optimizer, and owns
#                        fit / predict / evaluate / save / load.
#
#   feature_map(x) + ansatz  -> circuit  -> K observables  -> raw outputs  (K,)
#                                                         \-> backward():  d raw / d theta  (K, P)
#
# The weights live in the ansatz (ansatz.params_values), as everywhere in AriaQuanta -- unlike
# Qiskit, where weights are passed to forward()/backward() on every call. QNN.set_params_values()
# binds them.
#
# Cost of K observables: all K are estimated from the SAME circuit run -- the union of their
# Pauli strings is measured once (every Z-type string shares one Z-basis simulation), so a
# multi-class model with Z0, Z1, Z2 readouts costs the same number of circuit evaluations as a
# single-observable one.
#
# backward() is exact via parameter-shift *only* because forward() returns raw observable
# expectation values (sinusoidal in each ansatz parameter) -- never a loss composed with one.
# See AriaQuanta.qml.gradients.parameter_shift for the details; the chain rule through an
# OutputMap and a Loss is VQC's job.
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
class QNN:
    """
    Quantum neural network: classical sample(s) in, K observable expectation values out.

    :param feature_map: Callable(x) -> Circuit embedding one classical sample x as fixed
                    state-preparation gates, e.g. `lambda x: angle_encoding(x, num_of_qubits=n)`
                    or `lambda x: zz_feature_map(x, reps=2)`. QNN only ever calls it as
                    feature_map(x), with x a single sample (one row of X). May be None for an
                    ansatz carrying the data as symbolic input parameters (e.g.
                    DataReUploadingAnsatz).
    :param ansatz: The trainable circuit (an AriaQuanta.aqc.ansatz.Ansatz, e.g.
                    HardwareEfficientAnsatz), on the same number of qubits as feature_map(x).
    :param observable: Hamiltonian whose expectation value is the output, e.g.
                    Hamiltonian([('Z0', 1.0)]) -- OR a list of K Hamiltonians for K outputs.
                    forward()/backward() always return the K axis (K = 1 for a single Hamiltonian).
    :param num_of_iter_measure: Measurement shots per expectation-value estimate.
    :param initial_params: Starting weights for the ansatz. Defaults to a uniform-random draw in
                    [-pi, pi) (seeded by random_state) -- an ansatz's gates are symbolic (unbound)
                    until set_params_values() is called at least once, so QNN always binds
                    something at construction.
    :param random_state: Seeds the default random initial_params draw. Ignored if
                    initial_params is given explicitly.
    """
    def __init__(self, feature_map: Optional[Callable[[np.ndarray], Circuit]], ansatz: Ansatz,
                 observable: Union[Hamiltonian, Sequence[Hamiltonian]], num_of_iter_measure: int = 200,
                 initial_params: Optional[Union[List[float], np.ndarray]] = None,
                 random_state: Optional[int] = None) -> None:

        self._check_init(feature_map, ansatz, observable, num_of_iter_measure)

        self.feature_map = feature_map
        self.ansatz = ansatz
        self.observable = observable
        self.multi_output = not isinstance(observable, Hamiltonian)
        self.observables: List[Hamiltonian] = [observable] if not self.multi_output else list(observable)
        self.num_of_outputs = len(self.observables)
        self.num_of_iter_measure = num_of_iter_measure

        # One merged readout: every distinct Pauli string of every observable is measured once
        # (coefficient 1.0), then observable k is rebuilt as  raw_k = sum_t C[k, t] * <P_t>.
        self._readout_paulis: List[str] = list(dict.fromkeys(p for h in self.observables for p in h.paulis))
        self._readout_hamiltonian = Hamiltonian([(p, 1.0) for p in self._readout_paulis])
        self._readout_coefs = np.zeros((self.num_of_outputs, len(self._readout_paulis)))
        for k, h in enumerate(self.observables):
            for pauli, coef in zip(h.paulis, h.coefs):
                self._readout_coefs[k, self._readout_paulis.index(pauli)] = float(coef)

        if initial_params is None:
            rng = np.random.default_rng(random_state)
            initial_params = rng.uniform(-np.pi, np.pi, size=len(ansatz.params_names))
        self.ansatz.set_params_values(initial_params)


    # ------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Raw observable expectation values for every sample. Shape (n_samples, K)."""
        X = validate_X(X)
        return np.array([self.forward_sample(x) for x in X])

    def backward(self, X: np.ndarray) -> np.ndarray:
        """
        Exact Jacobian of forward() with respect to the weights, by parameter-shift. Shape
        (n_samples, K, P), P = number of ansatz parameters. 2 * P circuit evaluations per sample.
        """
        X = validate_X(X)
        return np.array([self.backward_sample(x) for x in X])


    # ------------------------------------------------------------
    def forward_sample(self, x: np.ndarray) -> np.ndarray:
        """The K raw observable expectation values of ONE sample, shape (K,) -- from a single merged measurement."""
        circuit = self._build_circuit_for(x)
        per_term, _ = find_expectation_value(circuit, self._readout_hamiltonian, self.num_of_iter_measure)
        term_values = np.array([per_term[p] for p in self._readout_paulis], dtype=float)
        return self._readout_coefs @ term_values

    def backward_sample(self, x: np.ndarray) -> np.ndarray:
        """Jacobian d(forward_sample(x))/d(weights) of ONE sample, shape (K, P). The ansatz is left bound to its current weights."""
        return parameter_shift_jacobian(self.ansatz, lambda _ansatz: self.forward_sample(x))


    # ------------------------------------------------------------
    def _build_circuit_for(self, x: np.ndarray) -> Circuit:
        ansatz = self.ansatz
        if self.feature_map is None:
            # data lives inside the ansatz as symbolic inputs: bind this sample and run the ansatz as is
            ansatz.set_input_values(x)
            return ansatz

        encoding_circuit = self.feature_map(x)
        if not isinstance(encoding_circuit, Circuit):
            raise TypeError("'feature_map' must return a Circuit, got {}.".format(type(encoding_circuit).__name__))
        if encoding_circuit.num_of_qubits != ansatz.num_of_qubits:
            raise ValueError(
                "feature_map(x) produced a {}-qubit circuit but the ansatz has {} qubits -- "
                "they must act on the same number of qubits.".format(encoding_circuit.num_of_qubits, ansatz.num_of_qubits)
            )
        circuit = Circuit(ansatz.num_of_qubits)
        circuit.gates = list(encoding_circuit.gates) + list(ansatz.gates)
        # amplitude_encoding sets initial_state instead of adding gates -- without this the
        # model would silently see |0...0> instead of the data
        circuit.initial_state = encoding_circuit.initial_state
        circuit.statevector = encoding_circuit.initial_state

        return circuit


    # ------------------------------------------------------------
    @property
    def params_names(self) -> List[str]:
        return self.ansatz.params_names
    @property
    def params_values(self) -> np.ndarray:
        return self.ansatz.params_values
    @property
    def num_of_params(self) -> int:
        return len(self.ansatz.params_names)
    @property
    def num_of_qubits(self) -> int:
        return self.ansatz.num_of_qubits
    
    def set_params_values(self, params_values: Union[List[float], np.ndarray]) -> None:
        self.ansatz.set_params_values(params_values)


    # ------------------------------------------------------------
    @staticmethod
    def _check_init(feature_map, ansatz, observable, num_of_iter_measure) -> None:
        if not isinstance(ansatz, Ansatz):
            raise TypeError("'ansatz' must be an Ansatz instance, got {}.".format(type(ansatz).__name__))

        if feature_map is None:
            if not ansatz.input_names:
                raise ValueError(
                    "feature_map=None requires an ansatz with symbolic input parameters "
                    "(ansatz.input_names, e.g. DataReUploadingAnsatz); this ansatz has none, "
                    "so the data would never enter the circuit."
                )
        else:
            if not callable(feature_map):
                raise TypeError("'feature_map' must be callable or None, got {}.".format(type(feature_map).__name__))
            if ansatz.input_names:
                raise ValueError(
                    "This ansatz already carries the data through input parameters {} -- pass feature_map=None "
                    "(using both would encode the sample twice).".format(ansatz.input_names)
                )

        if isinstance(observable, Hamiltonian):
            pass
        elif isinstance(observable, (list, tuple)) and len(observable) > 0 and all(isinstance(h, Hamiltonian) for h in observable):
            pass
        else:
            raise TypeError("'observable' must be a Hamiltonian or a non-empty list of Hamiltonians, got {}.".format(type(observable).__name__))
        if not isinstance(num_of_iter_measure, int) or isinstance(num_of_iter_measure, bool):
            raise TypeError("'num_of_iter_measure' must be an int, got {}.".format(type(num_of_iter_measure).__name__))
        if num_of_iter_measure < 1:
            raise ValueError("'num_of_iter_measure' must be at least 1, got {}.".format(num_of_iter_measure))

