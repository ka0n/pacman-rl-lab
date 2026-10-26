# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for study protocol persistence and checkpoint provenance."""
import pytest


def test_unchanged_saved_protocol_allows_retry_before_any_training(tmp_path, monkeypatch):
    from pacman_lab import cnn_motion_study as study

    class BeforeTraining(Exception):
        pass

    monkeypatch.setattr(study, 'prepare_study', lambda *a, **kw:
                        ([('cnn_motion', study.SEEDS[0], tmp_path / 'run', True)], []))

    def stop_before_training(**kwargs):
        raise BeforeTraining

    monkeypatch.setattr(study, 'run_batched', stop_before_training)
    argv = ['run', '--root', str(tmp_path / 'study'), '--baseline', str(tmp_path / 'baseline')]
    with pytest.raises(BeforeTraining):
        study.main(argv)
    # The exact same invocation should pass the protocol check a second time.
    with pytest.raises(BeforeTraining):
        study.main(argv)
    import json
    protocol = tmp_path / 'study' / 'study_protocol.json'
    changed = json.loads(protocol.read_text())
    changed['params']['learning_rate'] *= 2
    protocol.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='protocol changed'):
        study.main(argv)


@pytest.mark.parametrize('corruption', ['seed', 'fine_tune', 'architecture'])
def test_completion_checkpoint_must_belong_to_same_scratch_run(tmp_path, corruption):
    torch = pytest.importorskip('torch')
    from pacman_lab.cnn_motion_study import HYPERPARAMS, SEEDS, STEPS, verify_checkpoint
    config = dict(HYPERPARAMS, architecture='cnn_single', seed=SEEDS[0], engine='rust')
    # Metadata fixtures are deliberately not neural checkpoints or experiment data.
    best = dict(config=config, architecture='cnn_single', fine_tune=None,
                input_shape=(4, 11, 13), step=25_008)
    last = dict(best, step=STEPS)
    if corruption == 'seed':
        last['config'] = dict(config, seed=SEEDS[1])
    elif corruption == 'fine_tune':
        last['fine_tune'] = {'checkpoint': 'unrelated-source'}
    else:
        last['architecture'] = 'cnn_motion'
    torch.save(best, tmp_path / 'best.pt')
    torch.save(last, tmp_path / 'last.pt')
    with pytest.raises(ValueError):
        verify_checkpoint(tmp_path / 'best.pt', architecture='cnn_single',
                          seed=SEEDS[0], engine='rust', complete_dir=tmp_path)


@pytest.mark.parametrize('field,value', [('step', 0), ('step', 1_000_001),
                                         ('config_architecture', 'cnn_motion')])
def test_selected_checkpoint_has_valid_step_and_consistent_architecture(tmp_path, field, value):
    torch = pytest.importorskip('torch')
    from pacman_lab.cnn_motion_study import HYPERPARAMS, SEEDS, verify_checkpoint
    config = dict(HYPERPARAMS, architecture='cnn_single', seed=SEEDS[0], engine='rust')
    payload = dict(config=config, architecture='cnn_single', fine_tune=None,
                   input_shape=(4, 11, 13), step=25_008)
    if field == 'step':
        payload['step'] = value
    else:
        payload['config']['architecture'] = value
    torch.save(payload, tmp_path / 'best.pt')
    with pytest.raises(ValueError):
        verify_checkpoint(tmp_path / 'best.pt', architecture='cnn_single',
                          seed=SEEDS[0], engine='rust')
