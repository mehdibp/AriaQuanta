from typing import Optional

from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.qml.encoding.base import amplitude_statevector


# -------------------------------------------------------------------------------------------
def amplitude_encoding(data, num_of_qubits: Optional[int] = None, normalize: bool = True) -> Circuit:
    """
    Amplitude encoding: encodes a classical vector x directly into the amplitudes of a
    quantum state, |psi> = sum_i x_i |i> / ||x||, packing 2**n classical numbers into n
    qubits -- exponentially more data per qubit than angle/basis encoding, at the cost of
    needing an (in general, deep) state-preparation circuit to physically realize |psi> on
    real hardware.

    AriaQuanta is a simulator, so this takes the standard simulator shortcut: the target
    amplitudes are written directly into the circuit's initial state rather than compiled
    into an explicit gate sequence. If/when AriaQuanta grows a state-preparation compiler
    (e.g. via a Mottonen-style routine), that would live alongside this function and produce
    an equivalent -- but hardware-realizable -- Circuit.

    :param data: Sequence of amplitudes (real or complex), length <= 2**num_of_qubits.
                 Padded with zeros if shorter.
    :param num_of_qubits: Number of qubits. Defaults to the minimum needed for len(data).
    :param normalize: If True, rescale data to unit norm. If False, data must already be
                       normalized (raises otherwise).
    :return: A new Circuit whose initial state is the (padded, normalized) data.
    """
    statevector, n = amplitude_statevector(data, num_of_qubits, normalize)
    qc = Circuit(n)
    qc.initial_state = statevector
    qc.statevector = statevector
    return qc
