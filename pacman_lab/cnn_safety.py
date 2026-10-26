# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure maze-topology risk features for postmortem analysis and reward shaping.

This module does not choose or override actions. Its features come exclusively
from observable player/ghost positions and known walls. No ghost RNG or hidden
movement direction is exposed to the policy.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache

from .core import MAPS, DIRECTIONS


def adjusted_reward(original: float, *, terminated: bool, won: bool,
                    death_penalty: float | None = None) -> float:
    """Change only a terminal *death* penalty from -5 to a requested value.

    A timeout is not a death; use terminated, not terminated OR truncated.
    Rewards emitted by the simulator are not mutated, so Rust/Python stay in
    parity and evaluations keep the original game mechanics.
    """
    if death_penalty is None:
        return float(original)
    if death_penalty >= 0:
        raise ValueError('death_penalty must be a negative number, e.g. -20')
    if terminated and not won:
        return float(original) + (death_penalty - (-5.0))
    return float(original)


class MazeRisk:
    def __init__(self, maze: str):
        rows = MAPS[maze]
        self.maze = maze
        floors = {(r, c) for r, row in enumerate(rows) for c, tile in enumerate(row)
                  if tile != '#'}
        self.neighbors = {
            p: tuple((p[0] + dr, p[1] + dc)
                     for dr, dc in DIRECTIONS if (p[0] + dr, p[1] + dc) in floors)
            for p in floors
        }
        self.distance = {}
        for start in floors:
            dist = {start: 0}
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                for nxt in self.neighbors[cur]:
                    if nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        queue.append(nxt)
            self.distance[start] = dist
        # Remove degree-1 leaves iteratively until reaching a cycle or junction.
        # Tiles peeled this way belong to a true blind branch, not just a corridor.
        degree = {p: len(nbs) for p, nbs in self.neighbors.items()}
        queue = deque(p for p, n in degree.items() if n <= 1)
        branch = set()
        while queue:
            p = queue.popleft()
            if p in branch:
                continue
            branch.add(p)
            for nb in self.neighbors[p]:
                if nb not in branch:
                    degree[nb] -= 1
                    if degree[nb] == 1:
                        queue.append(nb)
        self.blind_branch = frozenset(branch)

    def features(self, frame):
        """Input frame: four planes [walls,pellets,player,ghosts]."""
        import numpy as np
        player_at = np.argwhere(frame[2] != 0)
        if len(player_at) != 1:
            raise ValueError('Expected exactly one Pac-Man position')
        p = tuple(map(int, player_at[0]))
        ghosts = [tuple(map(int, g)) for g in np.argwhere(frame[3] != 0)]
        distances = sorted(self.distance[p].get(g, 10**6) for g in ghosts)
        near = sum(d <= 4 for d in distances)
        exit_neighbors = self.neighbors[p]
        safe_exits = sum(all(self.distance[nb].get(g, 10**6) > 1 for g in ghosts)
                         for nb in exit_neighbors)
        both_close = len(distances) >= 2 and distances[1] <= 5
        opposite_approach = False
        if both_close and len(ghosts) == 2:
            first_steps = []
            for ghost in ghosts:
                d = self.distance[p].get(ghost, 10**6)
                first_steps.append({nb for nb in exit_neighbors
                                    if self.distance[nb].get(ghost, 10**6) == d - 1})
            opposite_approach = bool(first_steps[0] and first_steps[1] and
                                     first_steps[0].isdisjoint(first_steps[1]))
        in_branch = p in self.blind_branch
        branch_threat = in_branch and bool(distances) and distances[0] <= 5
        constrained = (len(exit_neighbors) >= 2 and safe_exits <= 1 and
                       bool(distances) and distances[0] <= 3)
        # Bounded heuristic risk score for optional potential-based shaping.
        # This score is not the ground-truth probability of death.
        double_risk = max(0., (6 - distances[1]) / 6) if both_close else 0.
        branch_risk = max(0., (6 - distances[0]) / 6) if branch_threat else 0.
        risk = (2. * double_risk + 1.3 * double_risk * bool(opposite_approach)
                + branch_risk + 0.5 * bool(constrained))
        return {
            'player': list(p), 'ghosts': [list(g) for g in ghosts],
            'ghost_distances': [int(d) for d in distances],
            'ghosts_within_4': int(near),
            'two_ghosts_within_5': bool(both_close),
            'opposite_approach': bool(opposite_approach),
            'blind_branch': bool(in_branch),
            'blind_branch_threat': bool(branch_threat),
            'legal_exits': len(exit_neighbors), 'safe_exits': int(safe_exits),
            'constrained_exits': bool(constrained),
            'risk_score': round(float(risk), 6),
        }

    def potential(self, frame, weight: float) -> float:
        return -weight * self.features(frame)['risk_score']


def shaped_reward(base_reward: float, previous_frame, next_frame, *,
                  topology: MazeRisk | None, risk_weight: float,
                  terminated: bool, truncated: bool, won: bool,
                  death_penalty: float | None = None, gamma: float = 0.99) -> float:
    reward = adjusted_reward(base_reward, terminated=terminated, won=won,
                             death_penalty=death_penalty)
    if risk_weight:
        if topology is None:
            raise ValueError('topology required for risk shaping')
        # Potential-based shaping. The absorbing terminal potential is zero.
        before = topology.potential(previous_frame, risk_weight)
        after = 0. if terminated or truncated else topology.potential(next_frame, risk_weight)
        reward += gamma * after - before
    return reward


@lru_cache(maxsize=None)
def maze_risk(maze: str) -> MazeRisk:
    return MazeRisk(maze)
