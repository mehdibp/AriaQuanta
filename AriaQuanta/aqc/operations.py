from typing import Any, Callable, Dict, Optional, Tuple

from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase


# -------------------------------------------------------------------------------------------
class Operations:
    def __init__(self, name: str, conditions: Optional[Tuple[str, Any]], operation_gate: GateBase) -> None:
        self._validate_operation_gate(operation_gate)

        self.name = name
        self.conditions = conditions
        self.operation_gate = operation_gate
        self.qubits = operation_gate.qubits


    # ------------------------------------------------------------
    def _condition_met(self, clbit_values_dict: Optional[Dict[str, str]]) -> bool:
        # unconditional by default; subclasses override with their own logic
        return True

    # ------------------------------------------------------------
    def apply(self, num_of_qubits: int, multistate: np.ndarray, clbit_values_dict: Optional[Dict[str, str]]=None) -> np.ndarray:
        if not self._condition_met(clbit_values_dict):
            return multistate
        return self.operation_gate.apply(num_of_qubits, multistate)


    # ------------------------------------------------------------
    @staticmethod
    def _validate_operation_gate(operation_gate: Any) -> None:
        if not hasattr(operation_gate, 'qubits'):
            raise TypeError("operation_gate must expose a 'qubits' attribute (e.g. a Gate instance), got {}.".format(type(operation_gate).__name__))
        if not hasattr(operation_gate, 'apply'):
            raise TypeError("operation_gate must implement an apply(num_of_qubits, multistate) method, got {}.".format(type(operation_gate).__name__))

    @staticmethod
    def _validate_conditions(conditions: Any) -> Tuple[str, Any]:
        if not (isinstance(conditions, (tuple, list)) and len(conditions) == 2):
            raise ValueError("conditions must be a (clbit_name, expected_value) pair, got {}.".format(conditions))
        
        clbit_name, expected_value = conditions
        if not isinstance(clbit_name, str):
            raise ValueError("conditions[0] (the classical bit name) must be a string, got {}.".format(type(clbit_name).__name__))
        
        return (clbit_name, expected_value)


# -------------------------------------------------------------------------------------------
class If_cbit(Operations):
    def __init__(self, conditions: Tuple[str, Any], operation_gate: GateBase) -> None:
        conditions = self._validate_conditions(conditions)
        super().__init__(name='If_cbit', conditions=conditions, operation_gate=operation_gate)

    def _condition_met(self, clbit_values_dict: Optional[Dict[str, str]]) -> bool:
        clbit_name, expected_value = self.conditions
        if not clbit_values_dict or clbit_name not in clbit_values_dict:
            raise ValueError("If_cbit cannot evaluate its condition: classical bit '{}' has not been measured yet.".format(clbit_name))

        return clbit_values_dict[clbit_name] == str(expected_value)

# -------------------------------------------------------------------------------------------
class Else_cbit(Operations):
    def __init__(self, conditions: Tuple[str, Any], operation_gate: GateBase) -> None:
        conditions = self._validate_conditions(conditions)
        super().__init__(name='Else_cbit', conditions=conditions, operation_gate=operation_gate)

    def _condition_met(self, clbit_values_dict: Optional[Dict[str, str]]) -> bool:
        clbit_name, expected_value = self.conditions
        if not clbit_values_dict or clbit_name not in clbit_values_dict:
            raise ValueError("Else_cbit cannot evaluate its condition: classical bit '{}' has not been measured yet.".format(clbit_name))

        return clbit_values_dict[clbit_name] != str(expected_value)

# -------------------------------------------------------------------------------------------
class ClassicalControl(Operations):
    def __init__(self, condition: Callable[[Dict[str, str]], bool], operation_gate: GateBase) -> None:
        if not callable(condition):
            raise TypeError("condition must be a callable taking clbit_values_dict and returning bool, got {}.".format(type(condition).__name__))
        super().__init__(name='ClassicalControl', conditions=None, operation_gate=operation_gate)
        self.condition = condition

    def _condition_met(self, clbit_values_dict: Optional[Dict[str, str]]) -> bool:
        if clbit_values_dict is None:
            raise ValueError("ClassicalControl cannot evaluate its condition: no classical bit values are available yet.")

        return bool(self.condition(clbit_values_dict))

