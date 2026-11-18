import json
import os
from typing import Any, Dict, List, Optional

from AriaQuanta._utils import np
from AriaQuanta.aqc.ansatz import Ansatz


# -------------------------------------------------------------------------------------------
# Saving / restoring the trained parameters of an Ansatz.
#
# Only the trainable weights (ansatz.params_names / params_values) are stored -- not the
# circuit itself, not the input data (input_names), and not any callable (a feature_map is
# usually a lambda, which cannot be serialized). The circuit is always rebuilt by the user's
# own code; this file only puts the numbers back in. Parameters are matched BY NAME, not by
# position, so they stay meaningful even if the new ansatz lists them in a different order.
#
# The format is plain JSON (human-readable, no pickle -> loading a file can never execute
# code, and no dependency beyond the standard library). Python's json writes floats with the
# shortest repr that round-trips, so save -> load reproduces every value bit-for-bit.
# -------------------------------------------------------------------------------------------

_FORMAT  = 'AriaQuanta.params'
_VERSION = 1


# -------------------------------------------------------------------------------------------
def _structure(ansatz: Ansatz) -> List[List[Any]]:
    # A cheap fingerprint of the circuit layout: [gate name, qubits] for every real operation
    # (barriers excluded). Parameter names alone cannot tell e.g. a CX-entangled
    # HardwareEfficientAnsatz from a CZ-entangled one -- the names are identical -- so without this a
    # strict load would silently bind the numbers into a different circuit.
    return [[g.name, [int(q) for q in g.qubits]] for g in ansatz.operations]


# -------------------------------------------------------------------------------------------
def save_params(ansatz: Ansatz, path: str, extra: Optional[Dict[str, Any]]=None) -> None:
    """
    Writes the ansatz's current trainable parameters to a JSON file.

    :param ansatz: Ansatz whose params_names / params_values are saved.
    :param path: Destination file (str or pathlib.Path, e.g. 'model.json'). Missing parent folders are created,
                    an existing file is replaced.
    :param extra: Optional JSON-serializable metadata stored alongside the parameters
                    (VQC.save uses it for the training history).
    """
    if not isinstance(ansatz, Ansatz):
        raise TypeError("'ansatz' must be an Ansatz instance, got {}.".format(type(ansatz).__name__))

    values = np.asarray(ansatz.params_values, dtype=float).flatten()
    if not np.all(np.isfinite(values)):
        raise ValueError("The ansatz parameters contain NaN or Inf values (diverged training?); refusing to save them.")

    payload = {
        'format':        _FORMAT,
        'version':       _VERSION,
        'ansatz_class':  type(ansatz).__name__,
        'num_of_qubits': ansatz.num_of_qubits,
        'params_names':  list(ansatz.params_names),
        'params_values': values.tolist(),
        'structure':     _structure(ansatz),
        'extra':         extra if extra is not None else {},
    }
    text = json.dumps(payload, indent=2, allow_nan=False)   # serialize first: a bad 'extra' must not leave a half-written

    path = os.fspath(path)                                  # accept pathlib.Path too
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp_path, path)                              # atomic: an interrupted save never corrupts an older checkpoint


# -------------------------------------------------------------------------------------------
def read_params(path: str) -> Dict[str, Any]:
    """Reads and validates a file written by save_params(); returns its payload dict."""
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or payload.get('format') != _FORMAT:
        raise ValueError("'{}' is not an AriaQuanta parameter file.".format(path))
    if payload.get('version') != _VERSION:
        raise ValueError("Unsupported parameter-file version {!r} (this AriaQuanta reads version {}).".format(payload.get('version'), _VERSION))

    names, values = payload.get('params_names'), payload.get('params_values')
    if not (isinstance(names, list) and isinstance(values, list) and len(names) == len(values)):
        raise ValueError("'{}' is corrupted: 'params_names' and 'params_values' must be lists of the same length.".format(path))
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError("'{}' contains NaN or Inf parameter values.".format(path))

    payload.setdefault('extra', {})
    return payload


# -------------------------------------------------------------------------------------------
def load_params(ansatz: Ansatz, path: str, strict: bool = True) -> List[str]:
    """
    Loads saved parameters into `ansatz`, matching them by name.

    :param ansatz: Ansatz to bind the saved values to.
    :param path: File written by save_params().
    :param strict: True (default): the file must describe exactly this ansatz -- same qubit
                    count, same parameter names and same gate layout -- otherwise ValueError,
                    and nothing is changed. False: bind every name the two have in common and
                    leave the ansatz's other parameters as they are (e.g. warm-starting a
                    reps=3 HardwareEfficientAnsatz from a trained reps=1 one: its layer-0
                    names are a subset of the deeper ansatz's). Still raises if nothing matches.
    :return: Names of the parameters that were loaded.
    """
    if not isinstance(ansatz, Ansatz):
        raise TypeError("'ansatz' must be an Ansatz instance, got {}.".format(type(ansatz).__name__))

    payload = read_params(path)
    saved = dict(zip(payload['params_names'], payload['params_values']))

    if strict:
        if payload.get('num_of_qubits') != ansatz.num_of_qubits:
            raise ValueError("The file was saved from a {}-qubit ansatz, this one has {} qubits (use strict=False to "
                             "load the matching names only).".format(payload.get('num_of_qubits'), ansatz.num_of_qubits))
        if 'structure' in payload and payload['structure'] != _structure(ansatz):
            raise ValueError("The circuit layout (gates / qubits) differs from the one the parameters were saved from, "
                             "although the parameter names match (e.g. a different entangler or connectivity). "
                             "Use strict=False to bind the values anyway.")
        missing    = [n for n in ansatz.params_names if n not in saved]
        unexpected = [n for n in saved if n not in ansatz.params_names]
        if missing or unexpected:
            raise ValueError("Parameter names do not match. Missing from the file: {}. Not in this ansatz: {}. "
                             "(use strict=False to load the matching names only)".format(missing, unexpected))

    matched = {n: saved[n] for n in ansatz.params_names if n in saved}
    if not matched:
        raise ValueError("None of the saved parameter names {} exist in this ansatz ({})."
                         .format(list(saved), ansatz.params_names))

    ansatz.bind_parameters(matched)
    return list(matched)


