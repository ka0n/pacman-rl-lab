# SPDX-License-Identifier: GPL-3.0-or-later
"""Safety training's reward override, topology and safe checkpoint provenance."""
import hashlib

import numpy as np
import pytest

from pacman_lab.cnn_safety import (adjusted_reward, maze_risk, shaped_reward)
from pacman_lab.core import PacmanGame


def test_death_penalty_preserves_non_deaths_and_environment_rewards():
    assert adjusted_reward(-5.01, terminated=True, won=False, death_penalty=-20) == pytest.approx(-20.01)
    assert adjusted_reward(-0.01, terminated=False, won=False, death_penalty=-20) == -0.01
    assert adjusted_reward(10.99, terminated=True, won=True, death_penalty=-20) == 10.99
    # Timeout ends the episode but is NOT a terminal death.
    assert adjusted_reward(-0.01, terminated=False, won=False, death_penalty=-20) == -0.01
    assert adjusted_reward(-5.01, terminated=True, won=False) == -5.01
    with pytest.raises(ValueError, match='negative'):
        adjusted_reward(-5., terminated=True, won=False, death_penalty=10.)


def test_topology_detects_opposing_ghosts_and_branches():
    topo = maze_risk('large')
    assert (5,1) in topo.blind_branch
    assert (5,11) in topo.blind_branch
    assert (1,1) not in topo.blind_branch
    base = PacmanGame('large', ghosts=2).observe()
    base[2] = 0
    base[3] = 0
    base[2,1,6] = 1
    base[3,1,4] = 1
    base[3,1,8] = 1
    f = topo.features(base)
    assert f['two_ghosts_within_5']
    assert f['opposite_approach']
    assert f['ghost_distances'] == [2,2]
    # Danger inside a genuine leaf branch, not merely any 2-wide corridor.
    base[2] = 0
    base[3] = 0
    base[2,5,1] = 1
    base[3,5,2] = 1
    leaf = topo.features(base)
    assert leaf['blind_branch_threat']
    assert leaf['blind_branch']


def test_risk_shaping_is_potential_difference_and_zero_by_default():
    topo = maze_risk('large')
    game = PacmanGame('large',ghosts=2,seed=10)
    before = game.observe()
    after = game.step(3).observation
    baseline = shaped_reward(1.,before,after,topology=None,risk_weight=0.,
                             terminated=False,truncated=False,won=False)
    assert baseline == 1.
    observed = shaped_reward(1., before, after, topology=topo,risk_weight=0.5,
                             terminated=False,truncated=False,won=False)
    assert observed == pytest.approx(1. + 0.99*topo.potential(after,0.5) - topo.potential(before,0.5))
    terminal = shaped_reward(-5.01,before,after,topology=topo,risk_weight=0.5,
                             terminated=True,truncated=False,won=False,death_penalty=-20)
    assert terminal == pytest.approx(-20.01 - topo.potential(before,0.5))
    with pytest.raises(ValueError,match='topology'):
        shaped_reward(0.,before,after,topology=None,risk_weight=1.,
                      terminated=False,truncated=False,won=False)


def test_reward_override_does_not_change_episode_dynamics():
    """Reward postprocessing cannot mutate the world or its transitions."""
    game = PacmanGame('small', ghosts=2, seed=17)
    other = PacmanGame('small', ghosts=2, seed=17)
    for action in [3,3,1,1,2,2,1,3]:
        x,y = game.step(action), other.step(action)
        assert x.reward == y.reward
        assert shaped_reward(x.reward,game.observe(),x.observation,topology=None,
                             risk_weight=0,terminated=x.terminated,truncated=x.truncated,
                             won=x.info['won'],death_penalty=-20) == pytest.approx(
               shaped_reward(y.reward,other.observe(),y.observation,topology=None,
                             risk_weight=0,terminated=y.terminated,truncated=y.truncated,
                             won=y.info['won'],death_penalty=-20))
        if x.terminated or x.truncated:
            break


def test_safety_finetune_roundtrip_and_audit(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    from pacman_lab.cnn_batch import run_batched, eval_batched
    from pacman_lab.cnn_inspect import audit
    from pacman_lab.cnn_v09 import _config_kwargs
    from pacman_lab.marathon_train import checkpoint_read
    base = tmp_path/'base'
    opts = dict(architecture='cnn_single',steps=64,seed=42,output=base,worlds=4,
                device_name='cpu',eval_every=32,save_every=32,eval_episodes=3,
                replay_capacity=64,batch_size=4,warmup=8,train_every=8,
                burnin=1,unroll=2,maze='small',engine='python')
    run_batched(**opts)
    src = base/'best.pt'
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path/'safety_ft'
    ft = dict(opts,output=dest,seed=5000042,init_checkpoint=str(src),
              death_penalty=-20.,risk_shaping_weight=0.5)
    run_batched(**ft)
    cp = checkpoint_read(dest/'last.pt',torch.device('cpu'))
    assert cp['config']['death_penalty']==-20.
    assert cp['config']['risk_shaping_weight']==0.5
    assert cp['fine_tune']['optimizer_restored']
    assert hashlib.sha256(src.read_bytes()).hexdigest() == digest
    # Resume with exact safety objective, no accidental reversion to old -5.
    kwargs = _config_kwargs(cp)
    run_batched(output=dest,device_name='cpu',resume=str(dest/'last.pt'),**kwargs)
    with pytest.raises(ValueError,match='same v0.8 batch settings'):
        run_batched(**dict(ft,init_checkpoint=None,resume=str(dest/'last.pt'),
                           death_penalty=-5.,risk_shaping_weight=0.5))
    result = audit(dest/'best.pt',maze='small',seed=90000,episodes=7,
                   worlds=3,engine='python',device_name='cpu',
                   output=tmp_path/'diag.json',risk_analysis=True)
    assert set(result['summary']['loss_risk_last_5']) == {
        'two_ghosts_near_last_5','opposite_approach_last_5',
        'blind_branch_threat_last_5','constrained_exits_last_5'}
    model = None
    from pacman_lab.cnn_models import make_model
    model = make_model('cnn_single',cp['input_shape'])
    model.load_state_dict(checkpoint_read(dest/'best.pt',torch.device('cpu'))['model'])
    baseline = eval_batched(model,'cnn_single',torch.device('cpu'),maze='small',
                            episodes=7,seed=90000,engine='python',worlds=3)
    assert result['summary']['wins'] == baseline['wins']
    assert all('risk' in entry for entry in result['episodes_detail'])
