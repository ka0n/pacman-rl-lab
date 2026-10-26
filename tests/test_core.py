# SPDX-License-Identifier: GPL-3.0-or-later
from collections import deque

import numpy as np
import pytest

from pacman_lab.core import MAPS, PacmanGame
from pacman_lab.baseline import evaluate_random
from pacman_lab.train import stage


@pytest.mark.parametrize("maze", ["small", "large"])
def test_all_pellets_reachable(maze):
    game = PacmanGame(maze=maze, ghosts=0)
    visited = {game.player}
    queue = deque([game.player])
    while queue:
        for _, dest in game.legal_destinations(queue.popleft()):
            if dest not in visited:
                visited.add(dest)
                queue.append(dest)
    assert game.pellets <= visited


def test_observation_shape_and_channels():
    game = PacmanGame(ghosts=1)
    obs = game.observe()
    assert obs.shape == (4, 7, 9)
    assert obs.dtype == np.uint8
    assert obs[0].sum() == sum(row.count("#") for row in MAPS["small"])
    assert obs[1].sum() == len(game.pellets)
    assert obs[2].sum() == 1
    assert obs[3].sum() == 1


def test_ghostless_pellet_reward_and_wall():
    game = PacmanGame(ghosts=0)
    initial_pellets = len(game.pellets)
    result = game.step(3)  # player at row 1 col 1 -> right to pellet
    assert result.reward > 0
    assert game.collected == 1
    assert len(game.pellets) == initial_pellets - 1
    game.reset(seed=42)
    result = game.step(0)  # up into wall
    assert result.reward < 0
    assert game.player == game.initial_player


def test_seeded_ghost_runs_reproduce():
    left = PacmanGame(maze="large", ghosts=2, seed=77)
    right = PacmanGame(maze="large", ghosts=2, seed=77)
    for i in range(50):
        if left.done or right.done:
            break
        r1, r2 = left.step(i % 4), right.step(i % 4)
        assert np.array_equal(r1.observation, r2.observation)
        assert r1.reward == r2.reward
        assert r1.terminated == r2.terminated
    assert left.done == right.done


def test_ghostless_win():
    game = PacmanGame(ghosts=0)
    # Play legal shortest paths to each next pellet, without requiring the agent to do so.
    while game.pellets:
        queue = deque([(game.player, [])])
        seen = {game.player}
        route = None
        while queue and route is None:
            pos, actions = queue.popleft()
            if pos in game.pellets:
                route = actions
                break
            for action, nxt in game.legal_destinations(pos):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, actions + [action]))
        assert route is not None
        for action in route:
            game.step(action)
    assert game.won
    assert game.done
    assert game.info()["collected_fraction"] == 1


def test_max_steps_truncates():
    game = PacmanGame(ghosts=0, max_steps=1)
    result = game.step(0)
    assert result.truncated is True
    assert result.terminated is False


def test_no_step_after_terminal():
    game = PacmanGame(ghosts=0, max_steps=1)
    game.step(0)
    with pytest.raises(RuntimeError):
        game.step(0)


def test_baseline_and_stages():
    a = evaluate_random(episodes=3, seed=11)
    b = evaluate_random(episodes=3, seed=11)
    assert a == b
    assert stage("direct", 0.0) == (2, 1)
    assert stage("curriculum", 0.05) == (0, 1)
    assert stage("curriculum", 0.2) == (1, 2)
    assert stage("curriculum", 0.6) == (2, 1)
    assert stage("curriculum", 0.9) == (2, 1)
