from typing import Optional

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary import H, X, CNX
from AriaQuanta.qml._shared import validate_words


# -------------------------------------------------------------------------------------------
def qram_encoding(data, num_of_address_qubits: Optional[int] = None,
                  num_of_data_qubits: Optional[int] = None, superposition: bool = True) -> Circuit:
    """
    QRAM-style encoding (a quantum lookup table): stores a classical list of non-negative
    integers d_0, d_1, ... in a pair of registers -- an *address* register and a *data*
    register -- so that the circuit implements |i>|0> -> |i>|d_i> for every address i. With
    the address register in uniform superposition (superposition=True, the default) the
    result is the entangled state

        (1 / sqrt(2**n_a)) * sum_i |i>|d_i>

    which is the same structure as the NEQR image representation. Unlike amplitude_encoding
    (which writes amplitudes straight into the simulator's initial state), this is built
    only from real gates (X and multi-controlled-X), so it is hardware-realizable -- at the
    cost of O(2**n_a) multi-controlled gates, since each stored word needs its own address
    match. This is a QRAM *circuit* (a QROM-style lookup), not a physical memory
    architecture such as a bucket-brigade QRAM.

    Qubit layout (qubit 0 is the most significant bit, as everywhere in AriaQuanta):
        qubits 0 .. n_a-1          address register (address i in binary)
        qubits n_a .. n_a+n_d-1    data register    (word d_i in binary)

    :param data: Sequence of non-negative integers; data[i] is the word stored at address i.
                 Addresses beyond len(data) (up to 2**n_a - 1) store 0. Real-valued features
                 must be quantized to integers first.
    :param num_of_address_qubits: n_a. Defaults to ceil(log2(len(data))) (at least 1).
    :param num_of_data_qubits: n_d. Defaults to the bit length of max(data) (at least 1).
    :param superposition: If True, puts the address register in uniform superposition (H on
                          every address qubit) before loading. If False the address register
                          stays |0...0> and only data[0] is loaded -- useful when you want to
                          prepare your own address state before or after.
    :return: A new Circuit with n_a + n_d qubits (not yet run).
    """
    words = validate_words(data)

    min_address = max(1, int(np.ceil(np.log2(words.size))))
    n_a = num_of_address_qubits if num_of_address_qubits is not None else min_address
    if n_a < min_address:
        raise ValueError(
            "{} value(s) need at least {} address qubit(s); got num_of_address_qubits={}."
            .format(words.size, min_address, n_a)
        )

    min_data = max(1, int(words.max()).bit_length())
    n_d = num_of_data_qubits if num_of_data_qubits is not None else min_data
    if n_d < min_data:
        raise ValueError(
            "The largest value ({}) needs at least {} data qubit(s); got num_of_data_qubits={}."
            .format(int(words.max()), min_data, n_d)
        )

    qc = Circuit(n_a + n_d)
    address_qubits = list(range(n_a))

    if superposition:
        for q in address_qubits:
            qc | H(q)

    for address, word in enumerate(words.tolist()):
        if word == 0:
            continue                                    # |i>|0> is already |i>|d_i> -- nothing to do

        address_bits = format(address, '0{}b'.format(n_a))
        word_bits    = format(word,    '0{}b'.format(n_d))
        zero_address_qubits = [q for q, b in enumerate(address_bits) if b == '0']

        for q in zero_address_qubits:                   # turn "address == i" into "all address qubits are 1"
            qc | X(q)
        for j, b in enumerate(word_bits):
            if b == '1':
                qc | CNX(control_qubits=address_qubits, target_qubits=n_a + j)
        for q in zero_address_qubits:
            qc | X(q)

    return qc
