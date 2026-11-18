import json

import numpy as np
import pytest

from AriaQuanta.qml.ansatz import HardwareEfficientAnsatz
from AriaQuanta.qml.models.persistence import save_params, load_params, read_params


def trained(n=3, reps=2, **kwargs):
    ansatz = HardwareEfficientAnsatz(n, rotation='RY', reps=reps, **kwargs)
    ansatz.set_params_values(np.random.default_rng(0).uniform(-np.pi, np.pi, len(ansatz.params_names)))
    return ansatz


def test_roundtrip_is_bit_exact(tmp_path):
    source = trained()
    path = tmp_path / 'model.json'
    save_params(source, path)
    target = HardwareEfficientAnsatz(3, rotation='RY', reps=2)
    loaded = load_params(target, path)
    assert loaded == source.params_names
    np.testing.assert_array_equal(target.params_values, source.params_values)

def test_save_creates_missing_folders_and_replaces_existing_files(tmp_path):
    path = tmp_path / 'a' / 'b' / 'model.json'
    save_params(trained(), path)
    save_params(trained(reps=1), path)
    assert len(read_params(path)['params_names']) == len(trained(reps=1).params_names)     # the second save replaced the first

def test_extra_metadata_is_stored(tmp_path):
    path = tmp_path / 'model.json'
    save_params(trained(), path, extra={'history': {'loss': [1.0, 0.5]}})
    assert read_params(path)['extra'] == {'history': {'loss': [1.0, 0.5]}}

def test_strict_load_rejects_different_qubit_count_and_layout(tmp_path):
    path = tmp_path / 'model.json'
    save_params(trained(), path)
    with pytest.raises(ValueError):
        load_params(HardwareEfficientAnsatz(4, rotation='RY', reps=2), path)                 # qubit count
    with pytest.raises(ValueError):
        load_params(HardwareEfficientAnsatz(3, rotation='RY', reps=2, entangler='CZ'), path) # same names, different circuit
    with pytest.raises(ValueError):
        load_params(HardwareEfficientAnsatz(3, rotation='RY', reps=3), path)                 # different names

def test_non_strict_load_warm_starts_a_deeper_ansatz(tmp_path):
    path = tmp_path / 'model.json'
    shallow = trained(reps=1)
    save_params(shallow, path)
    deep = HardwareEfficientAnsatz(3, rotation='RY', reps=2)
    loaded = load_params(deep, path, strict=False)
    assert loaded == shallow.params_names
    k = len(shallow.params_names)
    np.testing.assert_array_equal(deep.params_values[:k], shallow.params_values)
    np.testing.assert_array_equal(deep.params_values[k:], 0.0)

def test_non_strict_load_still_fails_when_nothing_matches(tmp_path):
    path = tmp_path / 'model.json'
    save_params(trained(), path)
    other = HardwareEfficientAnsatz(3, rotation='RX', reps=2)
    with pytest.raises(ValueError):
        load_params(other, path, strict=False)

def test_refuses_to_save_non_finite_parameters(tmp_path):
    ansatz = trained()
    ansatz.set_params_values(np.full(len(ansatz.params_names), np.nan))
    with pytest.raises(ValueError):
        save_params(ansatz, tmp_path / 'model.json')
    assert not (tmp_path / 'model.json').exists()

def test_read_params_rejects_foreign_or_corrupted_files(tmp_path):
    foreign = tmp_path / 'foreign.json'
    foreign.write_text(json.dumps({'hello': 'world'}))
    with pytest.raises(ValueError):
        read_params(foreign)

    path = tmp_path / 'model.json'
    save_params(trained(), path)
    payload = json.loads(path.read_text())
    payload['params_values'] = payload['params_values'][:-1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_params(path)

    payload['version'] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        read_params(path)

def test_non_ansatz_is_rejected(tmp_path):
    with pytest.raises(TypeError):
        save_params('nope', tmp_path / 'model.json')


# VQC.save / VQC.load (these live in the project's own vqc.py) ------------------------------------------------------------
from AriaQuanta.qml.models.vqc import VQC

@pytest.mark.skipif(not hasattr(VQC, 'save'), reason='this vqc.py has no save/load')
def test_vqc_save_and_load_restore_parameters_and_history(tmp_path, ry_cx_ansatz, ry_feature_map, three_observables, exact_measurements):
    from AriaQuanta.qml.models.vqc import SoftmaxOutputMap
    from AriaQuanta.qml.training.loss import CategoricalCrossEntropyLoss
    from AriaQuanta.qml.training.optimizer import Adam
    make = lambda: VQC(ry_feature_map(3), ry_cx_ansatz(3, 2), three_observables, CategoricalCrossEntropyLoss(), Adam(),
                       output_map=SoftmaxOutputMap(2.0), random_state=0, num_of_iter_measure=10)
    X = np.random.default_rng(3).uniform(0, np.pi, (6, 3))
    y = np.argmax(X, axis=1)

    model = make()
    model.fit(X, y, epochs=2, batch_size=3, random_state=0)
    path = tmp_path / 'vqc.json'
    model.save(path)

    fresh = make()
    fresh.ansatz.set_params_values(np.zeros(len(fresh.ansatz.params_names)))
    fresh.load(path)
    np.testing.assert_array_equal(fresh.ansatz.params_values, model.ansatz.params_values)
    assert fresh.history['loss'] == model.history['loss']
    np.testing.assert_allclose(fresh.predict(X), model.predict(X))

