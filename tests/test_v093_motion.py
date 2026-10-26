# SPDX-License-Identifier: GPL-3.0-or-later
"""Ghost-motion model transfer, deterministic eval and retrospective risk windows."""
from __future__ import annotations

import hashlib
import numpy as np
import pytest

from pacman_lab.cnn_motion import GhostMotionHistory, observer_for, current_frame
from pacman_lab.cnn_inspect import _time_window_risks, RISK_KEYS


def obs(*positions):
    a = np.zeros((4, 7, 9), dtype=np.uint8)
    a[0, 0] = 1
    a[2, 1, 1] = 1
    for r, c in positions:
        a[3, r, c] = 1
    return a


def test_motion_history_reset_and_step_hide_no_future_information():
    history = GhostMotionHistory()
    with pytest.raises(RuntimeError, match='reset'):
        history.step(obs((3, 3)))
    first = obs((3, 3), (4, 5))
    state = history.reset(first)
    assert state.shape == (5, 7, 9)
    assert np.array_equal(state[:4], first)
    assert np.array_equal(state[4], first[3])
    next_frame = obs((3, 4), (4, 5))
    state2 = history.step(next_frame)
    assert np.array_equal(state2[:4], next_frame)
    assert np.array_equal(state2[4], first[3])
    assert np.array_equal(current_frame(state2, 'cnn_motion'), next_frame)
    # No ghost from the previous episode survives reset.
    new = history.reset(obs((1, 7)))
    assert np.array_equal(new[3], new[4])
    assert int(new[4].sum()) == 1
    assert observer_for('cnn_single', 1).reset(first).shape == (4, 7, 9)
    assert observer_for('cnn_history4', 4).reset(first).shape == (16, 7, 9)
    with pytest.raises(ValueError, match='requires'):
        history.reset(np.zeros((5,7,9), dtype=np.uint8))


def test_risk_windows_are_20_to_11_10_to_6_and_last_5():
    trail = []
    for i in range(20):
        trail.append({key: (i == n) for n, key in zip((2, 13, 17, 19), RISK_KEYS)})
    analysis = _time_window_risks(trail)
    assert all(x['eligible'] for x in analysis.values())
    assert analysis['early_11_20'][RISK_KEYS[0]]
    assert analysis['middle_6_10'][RISK_KEYS[1]]
    assert analysis['terminal_1_5'][RISK_KEYS[2]]
    assert analysis['terminal_1_5'][RISK_KEYS[3]]
    assert not analysis['early_11_20'][RISK_KEYS[2]]
    short = _time_window_risks(trail[:7])
    assert not short['early_11_20']['eligible']
    assert not short['middle_6_10']['eligible']
    assert short['terminal_1_5']['eligible']


def test_transplanted_model_matches_old_greedy_policy_and_safe_training(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from pacman_lab.cnn_models import make_model
    from pacman_lab.cnn_motion import transfer_cnn_single_to_motion
    from pacman_lab.cnn_batch import eval_batched, run_batched
    from pacman_lab.cnn_inspect import audit
    from pacman_lab.marathon_train import checkpoint_read
    from pacman_lab.cnn_v09 import _config_kwargs
    torch.manual_seed(31)
    old = make_model('cnn_single', (4, 7, 9)).eval()
    new = make_model('cnn_motion', (5, 7, 9)).eval()
    transfer_cnn_single_to_motion(old.state_dict(), new)
    with torch.no_grad():
        x = torch.rand(3,4,7,9)
        augmented = torch.cat((x, torch.rand(3,1,7,9)), 1)
        assert torch.allclose(old(x), new(augmented), rtol=1e-5, atol=1e-5)
        augmented[:,4] = torch.rand_like(augmented[:,4])
        assert torch.allclose(old(x), new(augmented), rtol=1e-5, atol=1e-5)
    assert not torch.any(new.state_dict()['0.conv.0.weight'][:,4])
    old_score = eval_batched(old, 'cnn_single', torch.device('cpu'),
                             maze='small',episodes=7,seed=81233,engine='python',worlds=3)
    new_score = eval_batched(new, 'cnn_motion', torch.device('cpu'),
                             maze='small',episodes=7,seed=81233,engine='python',worlds=3)
    assert old_score == new_score
    old_dir = tmp_path/'old'
    opts = dict(architecture='cnn_single',steps=64,seed=62,output=old_dir,worlds=3,
                device_name='cpu',eval_every=32,save_every=32,eval_episodes=3,
                replay_capacity=64,batch_size=4,warmup=8,train_every=8,
                burnin=1,unroll=2,maze='small',engine='python',death_penalty=-20.)
    run_batched(**opts)
    src = old_dir/'best.pt'
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    new_dir=tmp_path/'motion'
    run_batched(**dict(opts,architecture='cnn_motion',output=new_dir,seed=12042,
                       init_checkpoint=str(src),risk_shaping_weight=0.5))
    ck=checkpoint_read(new_dir/'last.pt',torch.device('cpu'))
    assert ck['architecture'] == 'cnn_motion'
    assert ck['input_shape'] == (5,7,9)
    assert ck['fine_tune']['motion_transfer'] is True
    assert ck['fine_tune']['optimizer_restored'] is False
    assert ck['config']['death_penalty'] == -20.
    assert ck['config']['risk_shaping_weight'] == 0.5
    assert hashlib.sha256(src.read_bytes()).hexdigest() == digest
    # Fair ablation: same source + fresh Adam states for single-CNN control.
    control=tmp_path/'control'
    run_batched(**dict(opts,output=control,seed=12042,
                       init_checkpoint=str(src),fresh_optimizer=True,risk_shaping_weight=0.5))
    cp_control=checkpoint_read(control/'last.pt',torch.device('cpu'))
    assert cp_control['config']['fresh_optimizer']
    assert cp_control['fine_tune']['optimizer_restored'] is False
    # The old source model and new architecture must be loadable by the audit.
    report = audit(new_dir/'best.pt',maze='small',seed=222222,episodes=6,
                   worlds=2,engine='python',device_name='cpu',
                   output=tmp_path/'risk.json',risk_analysis=True)
    assert report['summary']['risk_time_windows']['early_11_20']['loss']['episodes'] >= 0
    assert 'time_windows' in report['episodes_detail'][0]['risk']
    restored=make_model('cnn_motion',ck['input_shape'])
    restored.load_state_dict(checkpoint_read(new_dir/'best.pt',torch.device('cpu'))['model'])
    check=eval_batched(restored,'cnn_motion',torch.device('cpu'),
                       maze='small',episodes=6,seed=222222,engine='python',worlds=2)
    assert report['summary']['wins'] == check['wins']
    run_batched(output=new_dir,device_name='cpu',resume=str(new_dir/'last.pt'),
                **_config_kwargs(ck))
