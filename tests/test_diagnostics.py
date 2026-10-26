# SPDX-License-Identifier: GPL-3.0-or-later
import random

import numpy as np
import pytest

from pacman_lab.core import PacmanGame
from pacman_lab.probe import moved_ghost_observation
from pacman_lab.train import stage


def test_wall_bumps_and_stalls_reset():
    game = PacmanGame(ghosts=0, seed=3)
    for _ in range(3):
        game.step(0)
    info = game.info()
    assert info['wall_bumps'] == 3
    assert info['stationary_steps'] == 3
    assert info['stall_streak'] == 3
    assert info['max_stall_streak'] == 3
    game.step(3)
    assert game.stall_streak == 0
    assert game.wall_bumps == 3
    game.reset(seed=1)
    assert game.wall_bumps == game.stationary_steps == game.reversals == 0


def test_reversal():
    game = PacmanGame(ghosts=0)
    game.step(3)  # right
    game.step(2)  # left, returns
    assert game.reversals == 1


def test_ghost_probe_preserves_other_channels():
    game = PacmanGame(ghosts=2)
    before = game.observe()
    altered, old, new = moved_ghost_observation(game, random.Random(2))
    assert old != new
    assert np.array_equal(before[:3], altered[:3])
    assert before[3].sum() == altered[3].sum() == 2
    assert not np.array_equal(before[3], altered[3])
    assert np.array_equal(before, game.observe()), 'probe must not mutate live game'


def test_new_stages():
    assert [stage('curriculum', x) for x in [0, .099, .10, .249, .25, .99]] == [
        (0,1),(0,1),(1,2),(1,2),(2,1),(2,1)
    ]
    with pytest.raises(ValueError):
        stage('unknown', .5)


def test_v01_checkpoint_warm_start(tmp_path):
    torch = pytest.importorskip('torch')
    from pacman_lab.dqn import make_qnet
    from pacman_lab.train import run_training

    torch.set_num_threads(1)
    env = PacmanGame(ghosts=2)
    path = tmp_path / 'v01.pt'
    torch.save({'model': make_qnet(env.observe().shape).state_dict(),
                'maze': 'small', 'input_shape': env.observe().shape,
                'steps': 50000, 'seed': 42, 'mode': 'direct', 'metrics': {}}, path)
    target = tmp_path / 'continued'
    run_training(steps=8, eval_every=4, output=str(target), device_name='cpu',
                 batch_size=4, replay_capacity=10, resume=str(path), eval_episodes=2)
    p = torch.load(target / 'last.pt', weights_only=True)
    assert p['steps'] == 50008
    assert p['version'] == '0.3'
    assert 'optimizer' in p and 'target' in p
    assert (target / 'metrics.csv').exists()
    with pytest.raises(FileExistsError):
        run_training(steps=1, eval_every=1, output=str(target), device_name='cpu',
                     eval_episodes=1)
