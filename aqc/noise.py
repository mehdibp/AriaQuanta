from typing import List, Optional, Union

from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary import X, Y, Z, GateCustom


# -------------------------------------------------------------------------------------------
class NoiseClass:
    def __init__(self, name: str, kraus_operators: List[np.ndarray], target_qubits: Optional[Union[int, List[int]]]=None) -> None:
            kraus_operators = [np.asarray(k, dtype=complex) for k in kraus_operators]
            n_kraus_qubits = self._validate_kraus_operators(kraus_operators)
    
            if target_qubits is None:
                if n_kraus_qubits != 1:
                    raise ValueError("target_qubits must be given explicitly for a {}-qubit noise channel.".format(n_kraus_qubits))
                
                self.target_qubits: Optional[List[int]] = None
                self.qubits: List[int] = []

            else:
                target_qubits_arr = np.atleast_1d(np.asarray(target_qubits, dtype=int)).flatten()
                if target_qubits_arr.size != n_kraus_qubits:
                    raise ValueError( "This {}-qubit noise channel needs {} target qubit(s), got {}.".format(n_kraus_qubits, n_kraus_qubits, target_qubits_arr.size) )
                if len(set(target_qubits_arr.tolist())) != target_qubits_arr.size:
                    raise ValueError("target_qubits must not contain duplicates, got {}.".format(target_qubits_arr.tolist()))

                self.target_qubits = target_qubits_arr.tolist()
                self.qubits = target_qubits_arr.tolist()
    
            self.name = name
            self.kraus_operators = kraus_operators


    # ------------------------------------------------------------
    @staticmethod
    def _validate_probability(value: float, name: str = "probability") -> None:
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("'{}' must be between 0 and 1, got {}.".format(name, value))
        
    # ------------------------------------------------------------
    @staticmethod
    def _validate_kraus_operators(kraus_operators: List[np.ndarray]) -> int:
        if not kraus_operators:
            raise ValueError("kraus_operators must contain at least one operator.")

        dim = None
        completeness = None
        for k in kraus_operators:
            if k.ndim != 2 or k.shape[0] != k.shape[1]:
                raise ValueError("Each Kraus operator must be a square matrix, got shape {}.".format(k.shape))
            if not np.all(np.isfinite(k)):
                raise ValueError("A Kraus operator contains NaN or Inf values.")
            if dim is None:
                dim = k.shape[0]
                if dim & (dim - 1) != 0:
                    raise ValueError("Kraus operator dimension must be a power of two, got {}.".format(dim))
            elif k.shape[0] != dim:
                raise ValueError("All Kraus operators must share the same dimension.")

            contribution = k.conj().T @ k                           # E_k^\dagger E_k
            completeness = contribution if completeness is None else completeness + contribution

        if not np.allclose(completeness, np.eye(dim), atol=1e-6):   # $ \sum_k E_k^\dagger E_k = I $
            raise ValueError( "Kraus operators must satisfy the completeness relation sum(K^dagger K) = I; got sum(K^dagger K) =\n{}".format(completeness) )
        return int(np.log2(dim))

    # ------------------------------------------------------------
    def _target_groups(self, num_of_qubits: int) -> List[List[int]]:
        # one group of qubits per independent application of the channel this "shot"
        if self.target_qubits is None:
            return [[q] for q in range(num_of_qubits)]
        return [self.target_qubits]

    # ------------------------------------------------------------
    def _apply_trajectory(self, target_qubits: List[int], num_of_qubits: int, state: np.ndarray) -> np.ndarray:
        candidates = []
        probabilities = []
        for k in self.kraus_operators:
            gate = GateCustom(name=self.name, matrix=k, target_qubits=target_qubits)
            candidate = gate.apply(num_of_qubits, state)
            p = float(np.real(np.vdot(candidate, candidate)))
            candidates.append(candidate)
            probabilities.append(max(p, 0.0))

        probabilities = np.array(probabilities)
        total = probabilities.sum()
        if total <= 0:
            raise ValueError("Noise channel '{}' produced a zero-norm state; check its Kraus operators.".format(self.name))
        probabilities = probabilities / total

        choice = int(np.random.choice(len(self.kraus_operators), p=probabilities))
        return candidates[choice] / np.sqrt(probabilities[choice])


    # ------------------------------------------------------------
    def apply(self, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
        for target_qubits in self._target_groups(num_of_qubits):
            multistate = self._apply_trajectory(target_qubits, num_of_qubits, multistate)
        return multistate
    
    # ------------------------------------------------------------
    def apply_density(self, num_of_qubits: int, density_matrix: np.ndarray) -> np.ndarray:
        for target_qubits in self._target_groups(num_of_qubits):
            updated = np.zeros_like(density_matrix)
            for k in self.kraus_operators:
                gate = GateCustom(name=self.name, matrix=k, target_qubits=target_qubits)
                full_k = gate._full_matrix(num_of_qubits)
                updated = updated + full_k @ density_matrix @ np.conj(full_k.T)     # E_k * pho * E_k^dag
            density_matrix = updated
        return density_matrix


    # ------------------------------------------------------------
    def __repr__(self) -> str:
        target = 'all qubits' if self.target_qubits is None else self.target_qubits
        return "{}(target_qubits={})".format(self.name, target)



# -------------------------------------------------------------------------------------------
class BitFlipNoise(NoiseClass):
    # With probability p, applies X (bit flip); otherwise leaves the qubit alone.

    def __init__(self, probability: float=1., target_qubits: Optional[Union[int, List[int]]]=None) -> None:
        self._validate_probability(probability)
        x = X(0).matrix
        k0 = np.sqrt(1.0 - probability) * np.eye(2, dtype=complex)
        k1 = np.sqrt(probability) * x
        super().__init__(name='BitFlip', kraus_operators=[k0, k1], target_qubits=target_qubits)
        self.probability = probability

# -------------------------------------------------------------------------------------------
class PhaseFlipNoise(NoiseClass):
    # With probability p, applies Z (phase flip); otherwise leaves the qubit alone.

    def __init__(self, probability: float = 1.0, target_qubits: Optional[Union[int, List[int]]] = None) -> None:
        self._validate_probability(probability)
        z = Z(0).matrix
        k0 = np.sqrt(1.0 - probability) * np.eye(2, dtype=complex)
        k1 = np.sqrt(probability) * z
        super().__init__(name='PhaseFlip', kraus_operators=[k0, k1], target_qubits=target_qubits)
        self.probability = probability

# -------------------------------------------------------------------------------------------
class DepolarizingNoise(NoiseClass):
    # Kraus operators:  K0 = sqrt(1 - 3p/4) I,  K1 = sqrt(p/4) X,  K2 = sqrt(p/4) Y,  K3 = sqrt(p/4) Z

    def __init__(self, probability: float = 1.0, target_qubits: Optional[Union[int, List[int]]] = None) -> None:
        self._validate_probability(probability)
        x, y, z = X(0).matrix, Y(0).matrix, Z(0).matrix
        k0 = np.sqrt(1.0 - 3.0 * probability / 4.0) * np.eye(2, dtype=complex)
        k1 = np.sqrt(probability / 4.0) * x
        k2 = np.sqrt(probability / 4.0) * y
        k3 = np.sqrt(probability / 4.0) * z
        super().__init__(name='Depolarizing', kraus_operators=[k0, k1, k2, k3], target_qubits=target_qubits)
        self.probability = probability

# -------------------------------------------------------------------------------------------
class AmplitudeDampingNoise(NoiseClass):
    # K0 = [[1, 0], [0, sqrt(1 - gamma)]],  K1 = [[0, sqrt(gamma)], [0, 0]]
    
    def __init__(self, gamma: float = 1.0, target_qubits: Optional[Union[int, List[int]]] = None) -> None:
        self._validate_probability(gamma, "gamma")
        k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
        k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
        super().__init__(name='AmplitudeDamping', kraus_operators=[k0, k1], target_qubits=target_qubits)
        self.gamma = gamma

# -------------------------------------------------------------------------------------------
class PhaseDampingNoise(NoiseClass):
    # K0 = [[1, 0], [0, sqrt(1 - lambda)]],  K1 = [[0, 0], [0, sqrt(lambda)]]

    def __init__(self, lam: float = 1.0, target_qubits: Optional[Union[int, List[int]]] = None) -> None:
        self._validate_probability(lam, "lam")
        k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - lam)]], dtype=complex)
        k1 = np.array([[0.0, 0.0], [0.0, np.sqrt(lam)]], dtype=complex)
        super().__init__(name='PhaseDamping', kraus_operators=[k0, k1], target_qubits=target_qubits)
        self.lam = lam

# -------------------------------------------------------------------------------------------
class CorrelatedBitFlipNoise(NoiseClass):
    def __init__(self, probability: float=1., target_qubits: Optional[List[int]]=None) -> None:
        self._validate_probability(probability)
        if target_qubits is None or len(np.atleast_1d(target_qubits)) != 2:
            raise ValueError("CorrelatedBitFlipNoise requires exactly two target qubits, e.g. target_qubits=[0, 1].")

        xx = np.kron(X(0).matrix, X(0).matrix)
        k0 = np.sqrt(1.0 - probability) * np.eye(4, dtype=complex)
        k1 = np.sqrt(probability) * xx
        super().__init__(name='CorrelatedBitFlip', kraus_operators=[k0, k1], target_qubits=target_qubits)
        self.probability = probability


# -------------------------------------------------------------------------------------------
class ThermalRelaxationNoise(NoiseClass):
    def __init__(self, t1: float, t2: float, gate_time: float, target_qubits: Optional[Union[int, List[int]]]=None) -> None:
        if t1 <= 0:
            raise ValueError("'t1' must be positive, got {}.".format(t1))
        if t2 <= 0:
            raise ValueError("'t2' must be positive, got {}.".format(t2))
        if gate_time < 0:
            raise ValueError("'gate_time' must be non-negative, got {}.".format(gate_time))
        if t2 > 2.0 * t1 * (1.0 + 1e-9):
            raise ValueError("Physically, t2 must be <= 2 * t1, got t1={}, t2={}.".format(t1, t2))

        gamma1 = 1.0 - np.exp(-gate_time / t1)
        if t2 < 2.0 * t1 * (1.0 - 1e-9):
            t_phi = 1.0 / (1.0 / t2 - 1.0 / (2.0 * t1))
            gamma2 = 1.0 - np.exp(-gate_time / t_phi)
        else:
            gamma2 = 0.0        # t2 == 2*t1: purely T1-limited, no extra pure dephasing

        a0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma1)]], dtype=complex)
        a1 = np.array([[0.0, np.sqrt(gamma1)], [0.0, 0.0]], dtype=complex)
        p0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma2)]], dtype=complex)
        p1 = np.array([[0.0, 0.0], [0.0, np.sqrt(gamma2)]], dtype=complex)

        # compose: amplitude damping happens first, then phase damping
        kraus_operators = [p_i @ a_j for p_i in (p0, p1) for a_j in (a0, a1)]
        super().__init__(name='ThermalRelaxation', kraus_operators=kraus_operators, target_qubits=target_qubits)
        self.t1, self.t2, self.gate_time = t1, t2, gate_time

# -------------------------------------------------------------------------------------------
class GeneralKrausChannel(NoiseClass):
    def __init__(self, kraus_operators: List[np.ndarray], target_qubits: Optional[Union[int, List[int]]]=None, name: str='GeneralKraus') -> None:
        super().__init__(name=name, kraus_operators=kraus_operators, target_qubits=target_qubits)



# -------------------------------------------------------------------------------------------
class ReadoutError:
    def __init__(self, 
                 probabilities: Optional[np.ndarray] = None,
                 p1given0: Optional[float] = None, 
                 p0given1: Optional[float] = None,
                 target_qubits: Optional[int] = None) -> None:

        if probabilities is not None:
            matrix = np.asarray(probabilities, dtype=float)
            if matrix.shape != (2, 2):
                raise ValueError("'probabilities' must be a 2x2 confusion matrix, got shape {}.".format(matrix.shape))
            if not np.allclose(matrix.sum(axis=0), 1.0):
                raise ValueError("Each column of the confusion matrix must sum to 1 (a probability distribution over measured outcomes given the true outcome).")
        else:
            if p1given0 is None or p0given1 is None:
                raise ValueError("Provide either 'probabilities' (a full 2x2 confusion matrix) or both 'p1given0' and 'p0given1'.")
            self._validate_probability(p1given0, "p1given0")
            self._validate_probability(p0given1, "p0given1")
            matrix = np.array([[1.0 - p1given0, p0given1],
                                [p1given0, 1.0 - p0given1]])

        self.confusion_matrix = matrix
        self.target_qubits = target_qubits

    # ------------------------------------------------------------
    def apply_to_bit(self, true_bit: str) -> str:
        if true_bit not in ('0', '1'):
            raise ValueError("true_bit must be '0' or '1', got {!r}.".format(true_bit))
        column = 0 if true_bit == '0' else 1
        p_measure_1 = self.confusion_matrix[1, column]
        return '1' if np.random.rand() < p_measure_1 else '0'

    # ------------------------------------------------------------
    @staticmethod
    def _validate_probability(value: float, name: str) -> None:
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("'{}' must be between 0 and 1, got {}.".format(name, value))



# -------------------------------------------------------------------------------------------
class NoiseModel:
    def __init__(self, noises: Optional[List[NoiseClass]]=None) -> None:
        self.noises: List[NoiseClass] = []
        for noise in (noises or []):
            self.add_noise(noise)

    # ------------------------------------------------------------
    def add_noise(self, noise: NoiseClass) -> "NoiseModel":
        if not isinstance(noise, NoiseClass):
            raise TypeError("NoiseModel only accepts NoiseClass instances, got {}.".format(type(noise).__name__))
        self.noises.append(noise)
        return self

    # ------------------------------------------------------------
    def apply(self, num_of_qubits: int, multistate: np.ndarray) -> np.ndarray:
        for noise in self.noises:
            multistate = noise.apply(num_of_qubits, multistate)
        return multistate

    # ------------------------------------------------------------
    def apply_density(self, num_of_qubits: int, density_matrix: np.ndarray) -> np.ndarray:
        for noise in self.noises:
            density_matrix = noise.apply_density(num_of_qubits, density_matrix)
        return density_matrix

    # ------------------------------------------------------------
    def __repr__(self) -> str:
        return "NoiseModel({} channel(s): {})".format(len(self.noises), [n.name for n in self.noises])
