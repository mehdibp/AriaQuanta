from itertools import combinations
from typing import List, Sequence, Tuple, Union
 
# -------------------------------------------------------------------------------------------
# Shared "which qubits does this k-qubit block act on" logic -- used by both
# AriaQuanta.qml.feature_map (multi-qubit Pauli blocks) and AriaQuanta.qml.ansatz
# (two-qubit entanglers). Both boil down to the exact same question: given a connectivity
# pattern ('full'/'linear'/'circular'/explicit) and a block size, which qubit-index tuples
# does a layer act on? Kept here, one level above both sub-packages (like _pauli.py),
# instead of duplicated or cross-imported between them.
# -------------------------------------------------------------------------------------------
 
 
def resolve_qubit_subsets(block_size: int, num_of_qubits: int,
                           entanglement: Union[str, Sequence[Tuple[int, ...]]]) -> List[Tuple[int, ...]]:
    """
    All qubit subsets a block of size `block_size` (a Pauli term, a two-qubit entangler,
    ...) gets applied to.
 
    :param entanglement: 'full' (every combination of block_size qubits), 'linear' (only
                          consecutive windows, e.g. (0,1),(1,2),... for block_size=2),
                          'circular' (like 'linear', plus the wraparound window that closes
                          the ring, e.g. (0,1),(1,2),(2,0) for block_size=2, num_of_qubits=3),
                          or an explicit list of qubit-index tuples (used as-is, each of
                          length block_size). Ignored when block_size == 1 (applies to every
                          qubit regardless of `entanglement`).
    """
    if block_size == 1:
        return [(q,) for q in range(num_of_qubits)]
 
    if isinstance(entanglement, str):
        if entanglement == 'full':
            return list(combinations(range(num_of_qubits), block_size))
        elif entanglement == 'linear':
            return [tuple(range(i, i + block_size)) for i in range(num_of_qubits - block_size + 1)]
        elif entanglement == 'circular':
            if num_of_qubits < block_size:
                return []
            return [tuple((i + j) % num_of_qubits for j in range(block_size)) for i in range(num_of_qubits)]
        else:
            raise ValueError(
                "entanglement must be 'full', 'linear', 'circular', or an explicit list of qubit tuples, got {!r}."
                .format(entanglement)
            )
 
    subsets = [tuple(s) for s in entanglement]
    for s in subsets:
        if len(s) != block_size:
            raise ValueError(
                "Every explicit qubit subset must have length {} (the block size), got {}.".format(block_size, s)
            )
        if max(s) >= num_of_qubits:
            raise ValueError("Qubit subset {} references a qubit >= num_of_qubits ({}).".format(s, num_of_qubits))
    return subsets
