# SPDX-License-Identifier: GPL-3.0-or-later
from collections import deque

import numpy as np
import pytest

from pacman_lab.core import MAPS, PacmanGame
from pacman_lab.behavior import move_nearest_ghost_far


@pytest.mark.parametrize("maze", ["unseen_a", "unseen_b"])
def test_unseen_mazes_same_dimensions_all_pellets_reachable(maze):
    original = PacmanGame(maze="small", ghosts=2)
    game = PacmanGame(maze=maze, ghosts=2)
    assert game.observe().shape == original.observe().shape == (4, 7, 9)
    assert game.walls != original.walls
    assert len(game.ghosts) == 2
    visited = {game.player}
    queue = deque([game.player])
    while queue:
        for _, dest in game.legal_destinations(queue.popleft()):
            if dest not in visited:
                visited.add(dest)
                queue.append(dest)
    assert game.pellets <= visited
    assert MAPS[maze] != MAPS["small"]


def test_counterfactual_moves_nearest_ghost_and_preserves_world():
    game = PacmanGame(maze="small", ghosts=2, seed=1)
    original = game.observe()
    intervention = move_nearest_ghost_far(game, original)
    assert intervention is not None
    altered, old_pos, new_pos = intervention
    assert old_pos in game.ghosts
    assert new_pos not in game.walls
    assert new_pos not in game.ghosts
    assert int(altered[3].sum()) == int(original[3].sum()) == 2
    assert np.array_equal(original[:3], altered[:3])
    assert not np.array_equal(original[3], altered[3])
    assert np.array_equal(game.observe(), original)


def test_transfer_and_behavior_on_trusted_legacy_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from pacman_lab.train import run_training
    from pacman_lab.transfer import run_transfer
    from pacman_lab.behavior import run_behavior
    torch.set_num_threads(1)
    folder = tmp_path / "train"
    run_training(steps=6, eval_every=6, output=str(folder), device_name="cpu",
                 eval_episodes=2, batch_size=4, replay_capacity=10)
    checkpoint = folder / "best.pt"
    result = run_transfer(str(checkpoint), episodes=3, seed=40000, device_name="cpu")
    assert len(result["maps"]) == 3
    assert {entry["maze"] for entry in result["maps"]} == {"small", "unseen_a", "unseen_b"}
    assert sum(entry["trained_map"] for entry in result["maps"]) == 1
    for entry in result["maps"]:
        assert 0 <= entry["agent"]["pellet_percent"] <= 100
        assert 0 <= entry["random"]["avg_pellet_percent"] <= 100
    # Repeatability (same checkpoint, seeds, no new training).
    duplicate = run_transfer(str(checkpoint), episodes=3, seed=40000, device_name="cpu")
    assert result == duplicate
    behavior = run_behavior(str(checkpoint), episodes=2, seed=40000,
                            examples=3, device_name="cpu")
    assert behavior["near_steps"] + behavior["far_steps"] > 0
    assert 0 <= behavior["mean_pellets_percent"] <= 100
    assert len(behavior["examples"]) <= 3
    behavior_unseen = run_behavior(str(checkpoint), episodes=1, seed=40000,
                                   maze="unseen_a", examples=0, device_name="cpu")
    assert behavior_unseen["maze"] == "unseen_a"


def test_transfer_rejects_incompatible_large_layout(tmp_path):
    torch = pytest.importorskip("torch")
    from pacman_lab.train import run_training
    from pacman_lab.transfer import run_transfer
    torch.set_num_threads(1)
    folder = tmp_path / "train"
    run_training(steps=2, eval_every=2, output=str(folder), device_name="cpu",
                 eval_episodes=1, batch_size=4, replay_capacity=10)
    with pytest.raises(ValueError, match="shape"):
        run_transfer(str(folder / "best.pt"), episodes=1, seed=40000,
                     device_name="cpu", maps=("large",))
