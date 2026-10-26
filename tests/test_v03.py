# SPDX-License-Identifier: GPL-3.0-or-later
import json

import pytest

from pacman_lab.core import PacmanGame
from pacman_lab.train import exploration_epsilon


def test_epsilon_schedule_and_invalid_inputs():
    assert exploration_epsilon(1, 100, 1.0, 0.1, .8) == 1.0
    assert exploration_epsilon(41, 100, 1.0, 0.1, .8) == pytest.approx(.55)
    assert exploration_epsilon(81, 100, 1.0, 0.1, .8) == pytest.approx(.10)
    assert exploration_epsilon(100, 100, 1.0, 0.1, .8) == pytest.approx(.10)
    assert exploration_epsilon(3, 5, .4, .1, .8) == pytest.approx(.25)
    with pytest.raises(ValueError):
        exploration_epsilon(1, 100, .1, .4, .8)
    with pytest.raises(ValueError):
        exploration_epsilon(1, 100, 1, .1, 0)


def test_synthetic_ghost_placement_is_adjacent_and_legal():
    import random
    import numpy as np
    from pacman_lab.danger import _threatened_state
    game = PacmanGame(ghosts=0)
    original = game.observe()
    state, danger_action, legal = _threatened_state(game, random.Random(1))
    assert danger_action in legal
    assert len(legal) >= 2
    assert np.array_equal(original[:3], state[:3])
    assert state[3].sum() == 2
    assert original[3].sum() == 0
    assert np.array_equal(original, game.observe())


def test_training_config_and_benchmark(tmp_path):
    torch = pytest.importorskip('torch')
    from pacman_lab.train import run_training
    from pacman_lab.benchmark import benchmark
    from pacman_lab.danger import run_danger_probe

    torch.set_num_threads(1)
    folder = tmp_path / 'v03'
    run_training(steps=16, eval_every=8, output=str(folder), device_name='cpu',
                 batch_size=4, replay_capacity=10, eval_episodes=2)
    config = json.loads((folder / 'config.json').read_text())
    assert config['epsilon_start'] == 1
    assert config['epsilon_end'] == .1
    assert config['mode'] == 'direct'
    assert (folder / 'best.pt').exists()
    assert (folder / 'last.pt').exists()
    data = torch.load(folder / 'last.pt', weights_only=True)
    assert data['version'] == '0.3'
    assert data['steps'] == 16
    assert data['training_config']['eval_seed'] == 10000
    result = benchmark([str(folder / 'best.pt'), str(folder / 'last.pt')],
                       episodes=3, seed=20000, device_name='cpu')
    assert len(result['agents']) == 2
    assert result['random']['episodes'] == 3
    assert all(0 <= x['metrics']['pellet_percent'] <= 100 for x in result['agents'])
    danger = run_danger_probe(str(folder / 'last.pt'), samples=12,
                              device_name='cpu')
    assert (danger['ghost_step_percent'] + danger['wall_step_percent']
            + danger['safe_legal_step_percent']) == pytest.approx(100)
    assert danger['random_expected_ghost_step_percent'] == 25


def test_warm_start_exploration_metadata(tmp_path):
    torch = pytest.importorskip('torch')
    from pacman_lab.train import run_training
    torch.set_num_threads(1)
    a = tmp_path / 'fresh'
    b = tmp_path / 'warm'
    run_training(steps=4, eval_every=4, eval_episodes=1, device_name='cpu',
                 output=str(a), batch_size=4, replay_capacity=10)
    run_training(steps=4, eval_every=4, eval_episodes=1, device_name='cpu',
                 output=str(b), batch_size=4, replay_capacity=10,
                 resume=str(a / 'last.pt'))
    config = json.loads((b / 'config.json').read_text())
    assert config['epsilon_start'] == .4
    assert config['previous_steps'] == 4
    payload = torch.load(b / 'last.pt', weights_only=True)
    assert payload['steps'] == 8
