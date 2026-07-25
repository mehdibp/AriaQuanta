from typing import List, Optional, Tuple, Union
from AriaQuanta._utils import np


# -------------------------------------------------------------------------------------------
class Qubit:
    def __init__(self, name: str='', state: Optional[np.ndarray]=None):
        self.name  = name

        if state is None:
            self.state = np.array([[1], [0]], dtype=complex)    # |0>
        else:
            state = np.asarray(state, dtype=complex)
            if state.size != 2:
                raise ValueError(
                    "A Qubit state must have exactly 2 amplitudes (one for |0> and one for |1>), got {} element(s) with shape {}."
                    "For example, use np.array([[1], [0]]) for |0>, not a 4-element vector.".format(state.size, state.shape)
                )
            else:
                self.state = state.reshape(2, 1)


    # ------------------------------------------------------------
    def normalize(self) -> "Qubit":
        norm = np.linalg.norm(self.state)
        if norm == 0: raise ValueError("Cannot normalize a zero state vector (a = b = 0).")

        self.state = self.state / norm
        return self

    # ------------------------------------------------------------
    def density_matrix(self) -> np.ndarray:
        state = np.asarray(self.state, dtype=complex).reshape(2, 1)
        norm  = np.linalg.norm(state)
        if norm == 0: raise ValueError("Cannot compute the density matrix of a zero state vector.")

        state = state / norm
        return state @ state.conj().T

    # ------------------------------------------------------------
    def bloch_vector(self) -> Tuple[float, float, float]:
        state = np.asarray(self.state, dtype=complex).flatten()
        norm = np.linalg.norm(state)
        if norm == 0: raise ValueError("Cannot compute the Bloch vector of a zero state vector.")

        a, b = state[0] / norm, state[1] / norm
        inner = np.conj(a) * b

        x = float(2 * inner.real)
        y = float(2 * inner.imag)
        z = float((np.abs(a) ** 2 - np.abs(b) ** 2))
        return (x, y, z)

    # ------------------------------------------------------------
    def __repr__(self) -> str:
        flat = np.asarray(self.state).flatten()
        amps = np.round(flat, 4).tolist()
        return f"Qubit(name={self.name!r}, state={amps})"


    # ------------------------------------------------------------
    # ready-made states, as factory methods (the six cardinal states on the Bloch sphere) ---
    @classmethod
    def zero(cls, name: str = '') -> "Qubit":
        return cls(name=name, state=np.array([[1], [0]], dtype=complex))            # |0>

    @classmethod
    def one(cls, name: str = '') -> "Qubit":
        return cls(name=name, state=np.array([[0], [1]], dtype=complex))            # |1>

    @classmethod
    def plus(cls, name: str = '') -> "Qubit":
        amp = 1 / np.sqrt(2)
        return cls(name=name, state=np.array([[amp], [amp]], dtype=complex))        # |+> = (|0> + |1>) / sqrt(2)

    @classmethod
    def minus(cls, name: str = '') -> "Qubit":
        amp = 1 / np.sqrt(2)
        return cls(name=name, state=np.array([[amp], [-amp]], dtype=complex))       # |-> = (|0> - |1>) / sqrt(2)

    @classmethod
    def plus_i(cls, name: str = '') -> "Qubit":
        amp = 1 / np.sqrt(2)
        return cls(name=name, state=np.array([[amp], [1j * amp]], dtype=complex))   # |+i> = (|0> + i|1>) / sqrt(2)

    @classmethod
    def minus_i(cls, name: str = '') -> "Qubit":
        amp = 1 / np.sqrt(2)
        return cls(name=name, state=np.array([[amp], [-1j * amp]], dtype=complex))  # |-i> = (|0> - i|1>) / sqrt(2)


# -------------------------------------------------------------------------------------------
class MultiQubit:
    def __init__(self, num_of_qubits: int, list_of_qubits: Optional[List[Qubit]]=None):
        qubits: List[Qubit] = []
 
        if list_of_qubits is None: qubit_0 = Qubit()
        else: qubit_0 = list_of_qubits[0]

        qubits.append(qubit_0)
        multistate = qubit_0.state

        for i in range(1, num_of_qubits):
            if list_of_qubits is None: qubit_i = Qubit()
            else: qubit_i = list_of_qubits[i]

            state_i = qubit_i.state
            qubits.append(qubit_i)
            multistate = np.kron(multistate, state_i)

        self.num_of_qubits = num_of_qubits
        self.multistate = multistate
        self.qubits = qubits


    # ------------------------------------------------------------
    def __repr__(self) -> str:
        qubit_names = [q.name if q.name else '?' for q in self.qubits]
        return (
            f"MultiQubit(num_of_qubits={self.num_of_qubits}, "
            f"qubits={qubit_names}, multistate.shape={tuple(np.shape(self.multistate))})"
        )


# -------------------------------------------------------------------------------------------
def create_state(name: str, a: Union[complex, float], b: Optional[Union[complex, float]]=None, normalize: bool=True) -> Qubit:
    zero = np.array([[1], [0]], dtype=complex)
    one  = np.array([[0], [1]], dtype=complex)

    # Constructing the complex value b from the complex value a ----------
    a = complex(a)
    if b is not None: 
        b = complex(b)
    else:
        if abs(a.imag) > 1e-12:
            raise ValueError(
                "b must be given explicitly when 'a' is complex: its value cannot be uniquely derived from "
                "a alone once relative phase is involved. Use create_state(name, a, b=...)." )
        if abs(a.real) > 1: 
            raise ValueError("|a| must be <= 1 to derive b = sqrt(1 - a**2).")
        b = complex(np.sqrt(1 - a.real ** 2))


    state = a*zero + b*one      # State built from values a, b


    # Checking Qubit normalize ----------
    if normalize:
        norm = np.linalg.norm(state)
        if norm == 0: raise ValueError("Cannot normalize a zero state vector (a = b = 0).")
        state = state / norm
    else:
        norm_sq = float(np.abs(a)**2 + np.abs(b)**2)
        if not np.isclose(norm_sq, 1.0, atol=1e-8):
            raise ValueError(
                "State is not normalized (|a|^2 + |b|^2 = {:.6f} != 1). "
                "Pass normalize=True (default) or supply normalized amplitudes.".format(norm_sq) )

    return Qubit(name, state)
