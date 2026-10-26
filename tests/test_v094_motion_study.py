# SPDX-License-Identifier: GPL-3.0-or-later
"""Matched from-scratch initialization and strict study provenance."""
from pathlib import Path
import pytest


def test_matched_scratch_functional_parity_and_reject_invalid_configs(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from pacman_lab.cnn_models import make_model
    from pacman_lab.cnn_motion import transfer_cnn_single_to_motion
    from pacman_lab.cnn_batch import run_batched
    from pacman_lab.marathon_train import checkpoint_read

    torch.manual_seed(123)
    control = make_model('cnn_single', (4,7,9)).eval()
    torch.manual_seed(9999)
    motion = make_model('cnn_motion', (5,7,9)).eval()
    transfer_cnn_single_to_motion(control.state_dict(), motion)
    with torch.no_grad():
        x = torch.randn(4,4,7,9)
        prev = torch.randn(4,1,7,9)
        assert torch.allclose(control(x), motion(torch.cat([x,prev],dim=1)), atol=1e-5)

    kwargs = dict(steps=48, seed=123, worlds=2, device_name='cpu',
                  eval_every=24, save_every=24, eval_episodes=3,
                  replay_capacity=48, batch_size=4, warmup=8, train_every=8,
                  burnin=1, unroll=2, maze='small', engine='python',
                  death_penalty=-20.0)
    directory = tmp_path / 'motion'
    run_batched(architecture='cnn_motion',output=directory,
                matched_scratch_init=True,**kwargs)
    checkpoint = checkpoint_read(directory/'last.pt',torch.device('cpu'))
    assert checkpoint['fine_tune'] is None
    assert checkpoint['config']['matched_scratch_init'] is True
    assert checkpoint['input_shape'] == (5,7,9)
    with pytest.raises(ValueError,match='only valid'):
        run_batched(architecture='cnn_single',output=tmp_path/'invalid',
                    matched_scratch_init=True,**kwargs)


def test_study_verifies_baseline_and_does_not_overwrite(tmp_path):
    torch = pytest.importorskip('torch')
    from pacman_lab.cnn_motion_study import (HYPERPARAMS, prepare_study,
                                              verify_checkpoint, SEEDS)
    from pacman_lab.marathon_train import atomic_save_torch
    base = tmp_path/'old_run'
    base.mkdir()
    config = dict(HYPERPARAMS, architecture='cnn_single', seed=SEEDS[0],
                  engine='rust')
    payload = dict(config=config,architecture='cnn_single',fine_tune=None,
                   input_shape=(4,11,13),step=975008)
    atomic_save_torch(payload,base/'best.pt')
    atomic_save_torch(dict(payload,step=1_000_000),base/'last.pt')
    verify_checkpoint(base/'best.pt',architecture='cnn_single',seed=SEEDS[0],
                      engine='rust',complete_dir=base)
    tasks, checkpoints = prepare_study(tmp_path/'study',base/'best.pt','rust',verbose=False)
    assert len(tasks) == 5
    assert len(checkpoints) == 1
    assert tasks[0][0] == 'cnn_motion' and tasks[0][3] is True
    with pytest.raises(ValueError, match='not a from-scratch'):
        atomic_save_torch(dict(payload, fine_tune={'checkpoint':'bad'}),base/'best.pt')
        verify_checkpoint(base/'best.pt',architecture='cnn_single',seed=SEEDS[0],engine='rust')
    atomic_save_torch(payload,base/'best.pt')
    partial = tmp_path/'study'/'cnn_motion_seed930042'
    partial.mkdir(parents=True)
    (partial/'metrics.csv').write_text('incomplete',encoding='utf-8')
    with pytest.raises(FileExistsError,match='Partial run'):
        prepare_study(tmp_path/'study',base/'best.pt','rust',verbose=False)
