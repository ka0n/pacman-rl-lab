# SPDX-License-Identifier: GPL-3.0-or-later
"""Mandatory deterministic parity test and simple simulator benchmark.

Use `python -m pacman_lab.native_bench verify` before trusting Rust runs.
"""
from __future__ import annotations
import argparse
import random
import time
import numpy as np
from .core import PacmanGame, MAPS
from .native import NativePacmanGame, NativeBatch


def verify(episodes=100, seed=415):
    steps = 0
    for maze in MAPS:
        for ghosts in (0, 1, 2):
            for ghost_every in (1, 2):
                # Ghost timing and Python Random RNG must agree as well as rewards.
                python = PacmanGame(maze=maze, ghosts=ghosts, ghost_every=ghost_every)
                native = NativePacmanGame(maze=maze, ghosts=ghosts, ghost_every=ghost_every)
                try:
                    for ep in range(episodes):
                        ep_seed = seed + ep * 1009
                        a, ai = python.reset(seed=ep_seed)
                        b, bi = native.reset(seed=ep_seed)
                        np.testing.assert_array_equal(a, b, err_msg=f"reset {maze}/{ghosts}/{ep}")
                        assert ai == bi, (maze,ep, ai,bi)
                        actions = random.Random(ep_seed + 500000)
                        while not python.done:
                            action = actions.randrange(4)
                            py = python.step(action)
                            rs = native.step(action)
                            steps += 1
                            np.testing.assert_array_equal(py.observation, rs.observation,
                                err_msg=f"step {maze}/{ghosts}/{ep}/{python.steps}")
                            assert abs(py.reward-rs.reward)<1e-9, (maze,ep,python.steps,py.reward,rs.reward)
                            assert (py.terminated,py.truncated)==(rs.terminated,rs.truncated)
                            assert py.info == rs.info, (maze,ep,python.steps,py.info,rs.info)
                        assert native.done == python.done
                finally:
                    native.close()
    print(f"PARITY PASS: {len(MAPS)} mazes, 3 ghost counts, 2 ghost speeds, "
          f"{episodes} seeded episodes per setting, {steps:,} identical steps")


def bench(steps=100000, worlds=16, maze="large", seed=42):
    if worlds < 1 or steps < worlds:
        raise ValueError("Require steps >= worlds >= 1")
    actions = np.random.default_rng(seed).integers(0,4, size=(steps//worlds,worlds), dtype=np.uint8)
    # Fair Python comparison: each produces 4-plane observation and diagnostics.
    games = [PacmanGame(maze=maze, ghosts=2, seed=seed+i) for i in range(worlds)]
    counter = [0] * worlds
    t0=time.perf_counter()
    checksum_py = 0
    for row in actions:
        for i,action in enumerate(row):
            if games[i].done:
                counter[i]+=1
                games[i].reset(seed=seed+i+counter[i]*1000003)
            r=games[i].step(int(action))
            checksum_py += int(r.observation[1].sum())
    python_time = time.perf_counter()-t0
    with NativeBatch(maze=maze,count=worlds,ghosts=2,seed=seed) as native:
        counter = [0] * worlds
        t0=time.perf_counter()
        checksum_rs = 0
        done = np.zeros(worlds, dtype=bool)
        for row in actions:
            for i in np.flatnonzero(done):
                counter[i]+=1
                native.reset_one(int(i),seed=seed+int(i)+counter[i]*1000003)
            result=native.step(row)
            checksum_rs += int(result.observations[:,1].sum())
            done=result.terminated.astype(bool)|result.truncated.astype(bool)
        rust_time = time.perf_counter()-t0
    n=actions.shape[0]*worlds
    print(f"maze={maze} worlds={worlds} steps={n:,}")
    print(f"Python: {n/python_time:,.0f} steps/s | {python_time:.2f}s | checksum {checksum_py}")
    print(f"Rust:   {n/rust_time:,.0f} steps/s | {rust_time:.2f}s | checksum {checksum_rs}")
    print(f"Speedup: {python_time/rust_time:.2f}x (simulation+Python binding; NOT CNN training)")
    if checksum_py != checksum_rs:
        raise AssertionError("Simulation parity failed: pellet checksum differs")


def main():
    p = argparse.ArgumentParser(description="Verify and benchmark optional Rust simulator")
    p.add_argument("mode",choices=("verify","bench"))
    p.add_argument("--episodes",type=int,default=30)
    p.add_argument("--steps",type=int,default=100000)
    p.add_argument("--worlds",type=int,default=16)
    p.add_argument("--maze",choices=tuple(MAPS),default="large")
    a=p.parse_args()
    if a.mode=="verify":verify(episodes=a.episodes)
    else:
        verify(episodes=3)
        bench(steps=a.steps,worlds=a.worlds,maze=a.maze)

if __name__ == "__main__":main()
