from typing import List, Optional


# -------------------------------------------------------------------------------------------
class Barrier:
    """
    Visual-only synchronization marker: drawn as a dashed separator line in
    CircuitVisualizer, but a complete no-op for simulation (apply/apply_density return the
    state unchanged) -- unlike every real gate or noise channel in AriaQuanta. Also forces
    later-added gates onto a new column, the way a barrier does in other libraries, since it
    still participates in Circuit._assign_columns' normal qubit-sharing scheduling.

    target_qubits=None (the default -- what the ready-made `barrier` below is) spans every
    qubit of whichever Circuit it ends up in. That's resolved by Circuit.add_gate, not here,
    since a bare Barrier() doesn't know a circuit's size yet and the same instance may be
    reused across several circuits (qc1 | barrier; qc2 | barrier).
    """
    name = 'Barrier'
    matrix = None

    def __init__(self, target_qubits: Optional[List[int]] = None) -> None:
        self._target_qubits = None if target_qubits is None else list(target_qubits)

    @property
    def qubits(self) -> List[int]:
        if self._target_qubits is None:
            raise ValueError(
                "This Barrier spans 'every qubit' and hasn't been added to a Circuit yet "
                "('qc | barrier' is what resolves it) -- its qubit span isn't defined outside one."
            )
        return self._target_qubits

    def _resolved_for(self, num_of_qubits: int) -> "Barrier":
        return self if self._target_qubits is not None else Barrier(target_qubits=list(range(num_of_qubits)))

    def __repr__(self) -> str:
        return "Barrier(qubits={})".format(self._target_qubits)

