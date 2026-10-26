# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic-seed grid game. Only Python stdlib + numpy required."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

MAPS = {
    "small": (
        "#########",
        "#P......#",
        "#.##.##.#",
        "#.......#",
        "#.#.#.#.#",
        "#...G...#",
        "#########",
    ),
    # Same 7x9 shape as small, but unseen connectivity and ghost placements.
    # Keep these out of training: use only for zero-shot post-training tests.
    "unseen_a": (
        "#########",
        "#P...#..#",
        "#.#...#.#",
        "#.#.#...#",
        "#...#.#.#",
        "#G......#",
        "#########",
    ),
    "unseen_b": (
        "#########",
        "#P......#",
        "###.#.#.#",
        "#...#...#",
        "#.#...#.#",
        "#..G....#",
        "#########",
    ),
    "large_unseen_a": (
        "#############",
        "#P..........#",
        "#.##.###.##.#",
        "#...........#",
        "##.##.#.##.##",
        "#...........#",
        "##.##.#.##.##",
        "#...........#",
        "#.##.###.##.#",
        "#G.........G#",
        "#############",
    ),
    "large_unseen_b": (
        "#############",
        "#P..........#",
        "#..###.###..#",
        "#...........#",
        "###.#.#.#.###",
        "#...........#",
        "#.###...###.#",
        "#...........#",
        "###.#.#.#.###",
        "#G.........G#",
        "#############",
    ),
    "xlarge": (
        "###################",
        "#P................#",
        "#.###.###.###.###.#",
        "#.................#",
        "###.#.###.#.###.###",
        "#.................#",
        "#.###.#.###.#.###.#",
        "#........G........#",
        "#.###.###.###.###.#",
        "#.................#",
        "###.#.###.#.###.###",
        "#.................#",
        "#.###.#.###.#.###.#",
        "#G................#",
        "###################",
    ),
    "large": (
        "#############",
        "#P..........#",
        "#.###.#.###.#",
        "#.....#.....#",
        "###.#.#.#.###",
        "#...#...#...#",
        "###.#.#.#.###",
        "#.....#.....#",
        "#.###.#.###.#",
        "#G.........G#",
        "#############",
    ),
}
# Actions: up, down, left, right. Coordinates: row, column.
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def validate_batch_actions(actions, count: int) -> np.ndarray:
    """Validate before narrowing to the native action type or stepping any lane."""
    arr = np.asarray(actions)
    if arr.shape != (count,):
        raise ValueError(f"Expected {count} actions")
    if arr.dtype.kind not in "iu" or np.any(arr < 0) or np.any(arr > 3):
        raise ValueError("Actions must be integers between 0 and 3")
    return np.ascontiguousarray(arr, dtype=np.uint8)


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class PacmanGame:
    def __init__(
        self,
        maze: str = "small",
        ghosts: int = 1,
        ghost_every: int = 1,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        if maze not in MAPS:
            raise ValueError(f"maze must be one of {tuple(MAPS)}")
        if ghosts not in (0, 1, 2):
            raise ValueError("ghosts must be 0, 1, or 2")
        if ghost_every < 1:
            raise ValueError("ghost_every must be at least 1")
        self.maze_name = maze
        self.layout = MAPS[maze]
        self.height, self.width = len(self.layout), len(self.layout[0])
        if any(len(row) != self.width for row in self.layout):
            raise ValueError("rows in maze must all be the same width")
        self.walls = {
            (r, c) for r, row in enumerate(self.layout)
            for c, tile in enumerate(row) if tile == "#"
        }
        # Compute the immutable maze geometry once. The original implementation
        # rebuilt adjacency lists and every wall channel for every simulation step.
        self._neighbors = {
            (r,c): tuple((i,(r+dr,c+dc)) for i,(dr,dc) in enumerate(DIRECTIONS)
                         if (r+dr,c+dc) not in self.walls and
                         0 <= r+dr < self.height and 0 <= c+dc < self.width)
            for r,row in enumerate(self.layout) for c,tile in enumerate(row) if tile != '#'
        }
        self._base_frame = np.zeros((4,self.height,self.width),dtype=np.uint8)
        for r,c in self.walls:
            self._base_frame[0,r,c] = 1
        self._initial_pellets = tuple((r,c) for r,row in enumerate(self.layout)
                                      for c,tile in enumerate(row) if tile == '.')
        for r,c in self._initial_pellets:
            self._base_frame[1,r,c] = 1
        player_positions = [(r, c) for r, row in enumerate(self.layout)
                            for c, tile in enumerate(row) if tile == "P"]
        self.initial_player = player_positions[0]
        self.initial_ghosts = [(r, c) for r, row in enumerate(self.layout)
                               for c, tile in enumerate(row) if tile == "G"]
        if len(self.initial_ghosts) < 2:
            # Use an additional distant cell on tiny maps.
            self.initial_ghosts += [self._furthest_floor(self.initial_ghosts)]
        self.ghost_count = ghosts
        self.ghost_every = ghost_every
        self.max_steps = max_steps or (self.width * self.height * 8)
        self.rng = random.Random()
        self.reset(seed=seed)

    def _furthest_floor(self, from_positions: list[tuple[int, int]]) -> tuple[int, int]:
        floors = [(r, c) for r in range(self.height) for c in range(self.width)
                  if (r, c) not in self.walls and (r, c) != self.initial_player]
        return max(floors, key=lambda p: min(abs(p[0] - q[0]) + abs(p[1] - q[1])
                                              for q in from_positions))

    def reset(self, seed: int | None = None, ghosts: int | None = None,
              ghost_every: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng.seed(seed)
        if ghosts is not None:
            if ghosts not in (0, 1, 2):
                raise ValueError("ghosts must be 0, 1, 2")
            self.ghost_count = ghosts
        if ghost_every is not None:
            if ghost_every < 1:
                raise ValueError("ghost_every must be >= 1")
            self.ghost_every = ghost_every
        self.player = self.initial_player
        self.ghosts = list(self.initial_ghosts[:self.ghost_count])
        self.ghost_directions: list[int | None] = [None] * len(self.ghosts)
        self.pellets = set(self._initial_pellets)
        self._live_base = self._base_frame.copy()
        self.initial_pellet_count = len(self.pellets)
        self.collected = 0
        self.steps = 0
        # Diagnostics only: never alter the agent's reward or observation.
        self.wall_bumps = 0
        self.stationary_steps = 0
        self.stall_streak = 0
        self.max_stall_streak = 0
        self.reversals = 0
        self.previous_player = None
        self.done = False
        self.won = False
        return self.observe(), self.info()

    def info(self) -> dict:
        return {
            "collected": self.collected,
            "pellets_left": len(self.pellets),
            "collected_fraction": self.collected / max(1, self.initial_pellet_count),
            "steps": self.steps,
            "wall_bumps": self.wall_bumps,
            "stationary_steps": self.stationary_steps,
            "stall_streak": self.stall_streak,
            "max_stall_streak": self.max_stall_streak,
            "reversals": self.reversals,
            "won": self.won,
            "ghosts": len(self.ghosts),
        }

    def observe(self) -> np.ndarray:
        # 4 binary planes: walls, pellets, player, ghosts.
        obs = self._live_base.copy()
        obs[2, self.player[0], self.player[1]] = 1
        for r, c in self.ghosts:
            obs[3, r, c] = 1
        return obs

    def legal_destinations(self, position: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
        # Returning a list maintains the public API without rechecking walls.
        return list(self._neighbors.get(position, ()))

    def _move_ghosts(self) -> None:
        updated = []
        for pos, last_action in zip(self.ghosts, self.ghost_directions):
            moves = self._neighbors[pos]
            if not moves:
                updated.append((pos, last_action))
                continue
            same_direction = [move for move in moves if move[0] == last_action]
            if same_direction and self.rng.random() < 0.7:
                action, destination = same_direction[0]
            else:
                action, destination = self.rng.choice(moves)
            updated.append((destination, action))
        self.ghosts = [pos for pos, _ in updated]
        self.ghost_directions = [action for _, action in updated]

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("episode finished; call reset() before another step")
        if int(action) not in (0, 1, 2, 3):
            raise ValueError("action must be 0:up, 1:down, 2:left or 3:right")
        self.steps += 1
        reward = -0.01
        old_player = self.player
        dr, dc = DIRECTIONS[int(action)]
        destination = (old_player[0] + dr, old_player[1] + dc)
        if destination in self.walls or not (0 <= destination[0] < self.height and 0 <= destination[1] < self.width):
            reward -= 0.1
            self.wall_bumps += 1
        else:
            self.player = destination

        if self.player == old_player:
            self.stationary_steps += 1
            self.stall_streak += 1
            self.max_stall_streak = max(self.max_stall_streak, self.stall_streak)
        else:
            self.stall_streak = 0
            if self.previous_player is not None and self.player == self.previous_player:
                self.reversals += 1
        self.previous_player = old_player

        if self.player in self.ghosts:
            self.done = True
            reward -= 5.0
        if not self.done and self.player in self.pellets:
            self.pellets.remove(self.player)
            self._live_base[1, self.player[0], self.player[1]] = 0
            self.collected += 1
            reward += 1.0
        if not self.done and not self.pellets:
            self.done = self.won = True
            reward += 10.0
        if not self.done and self.ghosts and self.steps % self.ghost_every == 0:
            old_ghosts = list(self.ghosts)
            self._move_ghosts()
            # Detect occupying same tile or swapping tiles in one step.
            if self.player in self.ghosts or any(
                before == self.player and after == old_player
                for before, after in zip(old_ghosts, self.ghosts)
            ):
                self.done = True
                reward -= 5.0
        truncated = self.steps >= self.max_steps and not self.done
        if truncated:
            self.done = True
        return StepResult(self.observe(), reward, self.done and not truncated, truncated, self.info())
