# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env
from pacman_lab.gym_env import PacmanEnv


def test_environment_api():
    env = PacmanEnv(ghosts=2)
    assert env.action_space.n == 4
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=1, options={"ghosts": 0})
    assert env.observation_space.contains(obs)
    assert info["ghosts"] == 0
    obs, reward, terminated, truncated, info = env.step(3)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    env.close()
