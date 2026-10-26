# SPDX-License-Identifier: GPL-3.0-or-later
"""Checkpoint-compatible CNN viewer and loss-cause audit for Pac-Man Lab v0.9.

This is a read-only add-on. Same deterministic greedy policy as cnn_v09.evaluate,
2 ghosts by default, no exploratory actions. Audit uses the batch Rust/Python
engine. Viewer uses the parity-tested Python game so the map can be drawn.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .cnn_models import SPECS, make_model
from .core import MAPS, PacmanGame
from .dqn import select_device
from .marathon_train import checkpoint_read
from .cnn_motion import observer_for, current_frame
from .cnn_safety import maze_risk


def _load(path: Path, device_name: str):
    device = select_device(device_name)
    payload = checkpoint_read(path, device)
    arch = payload.get('architecture')
    if arch not in SPECS:
        raise ValueError(f'Unknown architecture in checkpoint: {arch!r}')
    shape = tuple(payload['input_shape'])
    model = make_model(arch, shape).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, SPECS[arch], shape, device, payload


def _assert_compatible(shape, maze):
    if maze not in MAPS:
        raise ValueError(f'Unknown maze {maze!r}')
    h, w = len(MAPS[maze]), len(MAPS[maze][0])
    if shape[1:] != (h, w):
        raise ValueError(f'Checkpoint grid {shape[1:]} != maze {maze} {(h, w)}')


def _positions(frame, plane):
    return [tuple(int(z) for z in pair) for pair in np.argwhere(frame[plane] != 0)]


def _death_type(before, after, truncated):
    if truncated:
        return 'timeout'
    # The game first checks collisions when Pac-Man moves; only afterwards
    # do ghosts move. A move into an occupied ghost cell is unambiguous.
    old_ghosts = set(_positions(before, 3))
    new_player = _positions(after, 2)[0]
    if new_player in old_ghosts:
        return 'player_entered_ghost'
    # New ghost on player or ghost/player swapped positions in one step.
    return 'ghost_moved_or_swapped'


def _infer_actions(model, spec, device, states, hidden):
    import torch
    with torch.inference_mode():
        obs = torch.as_tensor(np.stack(states), device=device, dtype=torch.float32)
        if spec.recurrent:
            q, hidden = model(obs[:, None], hidden)
            q = q[:, -1]
        else:
            q = model(obs)
        actions = q.argmax(dim=-1).cpu().numpy().astype(np.uint8)
    return actions, hidden


RISK_KEYS = ('two_ghosts_within_5', 'opposite_approach',
             'blind_branch_threat', 'constrained_exits')


def _time_window_risks(trail):
    """Retrospective WINDOWS, NOT causal predictions or counterfactual rollouts.

    Window 11-20 is only populated for episodes with at least 20 decisions.
    """
    intervals = {'early_11_20': (20, 10), 'middle_6_10': (10, 5),
                 'terminal_1_5': (5, 0)}
    result = {}
    for name, (outer, inner) in intervals.items():
        available = len(trail) >= outer
        frames = (trail[-outer:-inner] if inner else trail[-outer:]) if available else []
        result[name] = {
            'eligible': bool(available),
            'num_states': len(frames),
            **{key: bool(any(f[key] for f in frames)) for key in RISK_KEYS},
        }
    return result


def audit(checkpoint: Path, *, maze='large', seed=80000, episodes=1000,
          worlds=16, engine='rust', device_name='auto', output: Path,
          risk_analysis: bool = False):
    """Exact greedy eval seeds like v0.9, now keeping per-episode end diagnostics."""
    import torch
    if episodes < 1 or worlds < 1 or seed < 0:
        raise ValueError('episodes/worlds must be >= 1 and seed >= 0')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    checkpoint = Path(checkpoint).resolve(strict=True)
    model, spec, shape, device, payload = _load(checkpoint, device_name)
    _assert_compatible(shape, maze)
    if engine == 'rust':
        from .native import NativeBatch as GameBatch
    elif engine == 'python':
        from .cnn_batch import PythonBatch as GameBatch
    else:
        raise ValueError(f'Unknown engine {engine}')
    results = []
    topology = maze_risk(maze) if risk_analysis else None
    with torch.inference_mode():
        for base in range(0, episodes, worlds):
            count = min(worlds, episodes-base)
            env = GameBatch(maze=maze, count=count, ghosts=2)
            try:
                hist = [observer_for(spec.architecture, spec.frames) for _ in range(count)]
                states = []
                for i in range(count):
                    obs, _ = env.reset_one(i, seed=seed+base+i)
                    states.append(hist[i].reset(obs))
                done = np.zeros(count, dtype=bool)
                last_pellet_steps = [0] * count
                prior_collected = [0] * count
                risk_traces = [deque(maxlen=20) for _ in range(count)] if risk_analysis else None
                risk_exposures = ([Counter() for _ in range(count)] if risk_analysis else None)
                hidden = None
                filler_seed = seed+episodes+base*1_000_000
                while not bool(done.all()):
                    previous = [current_frame(state, spec.architecture).copy() for state in states]
                    if risk_analysis:
                        for i in range(count):
                            if not done[i]:
                                snap = topology.features(previous[i])
                                risk_traces[i].append(snap)
                                for key in ('two_ghosts_within_5', 'opposite_approach',
                                            'blind_branch_threat', 'constrained_exits'):
                                    if snap[key]:
                                        risk_exposures[i][key] += 1
                    actions, hidden = _infer_actions(model, spec, device, states, hidden)
                    outcome = env.step(actions)
                    for i in range(count):
                        if not done[i]:
                            info = outcome.stats[i]
                            if info['collected'] > prior_collected[i]:
                                last_pellet_steps[i] = info['steps']
                                prior_collected[i] = info['collected']
                            ended = bool(outcome.terminated[i] or outcome.truncated[i])
                            if ended:
                                done[i] = True
                                won = bool(info['won'])
                                reason = ('win' if won else _death_type(
                                    previous[i], outcome.observations[i],
                                    bool(outcome.truncated[i])))
                                record = {
                                    'seed': int(seed+base+i), 'won': won,
                                    'reason': reason,
                                    'steps': int(info['steps']),
                                    'pellets_left': int(info['pellets_left']),
                                    'pellet_percent': round(100*float(info['collected_fraction']), 3),
                                    'steps_since_last_pellet': int(info['steps']-last_pellet_steps[i]),
                                    'wall_bumps': int(info['wall_bumps']),
                                    'stationary_steps': int(info['stationary_steps']),
                                    'max_stall_streak': int(info['max_stall_streak']),
                                    'reversals': int(info['reversals']),
                                    'final_player': _positions(outcome.observations[i], 2)[0],
                                    'final_ghosts': _positions(outcome.observations[i], 3),
                                }
                                if risk_analysis:
                                    trail = list(risk_traces[i])
                                    last_five = trail[-5:]
                                    record['risk'] = {
                                        'exposure_steps': dict(risk_exposures[i]),
                                        'last_step': trail[-1] if trail else None,
                                        'last_5': last_five,
                                        'max_risk_last_12': max((e['risk_score'] for e in trail[-12:]), default=0),
                                        'max_risk_last_20': max((e['risk_score'] for e in trail), default=0),
                                        'time_windows': _time_window_risks(trail),
                                        'two_ghosts_near_last_5': any(
                                            e['two_ghosts_within_5'] for e in last_five),
                                        'opposite_approach_last_5': any(
                                            e['opposite_approach'] for e in last_five),
                                        'blind_branch_threat_last_5': any(
                                            e['blind_branch_threat'] for e in last_five),
                                        'constrained_exits_last_5': any(
                                            e['constrained_exits'] for e in last_five),
                                    }
                                results.append(record)
                        if done[i]:
                            # Same filler procedure as cnn_batch.eval_batched;
                            # it cannot affect another independent lane.
                            obs, _ = env.reset_one(i, seed=filler_seed)
                            filler_seed += 1
                            states[i] = hist[i].reset(obs)
                            if hidden is not None:
                                hidden[:, i, :] = 0
                        else:
                            states[i] = hist[i].step(outcome.observations[i])
            finally:
                env.close()
    results.sort(key=lambda record: record['seed'])
    losses = [r for r in results if not r['won']]
    reasons = Counter(r['reason'] for r in losses)
    wins = len(results)-len(losses)
    summary = {
        'episodes': len(results), 'wins': wins, 'losses': len(losses),
        'win_percent': round(100*wins/len(results), 2),
        'pellet_percent': round(float(np.mean([r['pellet_percent'] for r in results])), 2),
        'loss_reasons': dict(sorted(reasons.items())),
        'loss_pellet_percent_mean': round(float(np.mean([r['pellet_percent'] for r in losses])), 2) if losses else None,
        'loss_pellet_percent_median': round(float(np.median([r['pellet_percent'] for r in losses])), 2) if losses else None,
        'losses_with_95plus_pellets': sum(r['pellet_percent'] >= 95 for r in losses),
        'timeouts_without_pellet_for_100plus_steps': sum(
            r['reason'] == 'timeout' and r['steps_since_last_pellet'] >= 100
            for r in losses),
        'first_loss_seeds': [r['seed'] for r in losses[:12]],
        'near_win_loss_seeds': [r['seed'] for r in sorted(losses,
                               key=lambda r: -r['pellet_percent'])[:12]],
    }
    if risk_analysis:
        features = ('two_ghosts_near_last_5', 'opposite_approach_last_5',
                    'blind_branch_threat_last_5', 'constrained_exits_last_5')
        for label, subset in (('loss', losses), ('win', [r for r in results if r['won']])):
            summary[f'{label}_risk_last_5'] = {
                name: sum(bool(r['risk'][name]) for r in subset) for name in features
            }
        summary['risk_time_windows'] = {}
        for window in ('early_11_20', 'middle_6_10', 'terminal_1_5'):
            summary['risk_time_windows'][window] = {}
            for label, subset in (('loss', losses), ('win', [r for r in results if r['won']])):
                eligible = [r['risk']['time_windows'][window] for r in subset
                            if r['risk']['time_windows'][window]['eligible']]
                summary['risk_time_windows'][window][label] = {
                    'episodes': len(eligible),
                    **{key: sum(bool(row[key]) for row in eligible) for key in RISK_KEYS},
                }
        summary['risk_notes'] = (
            'Risk flags are descriptive and overlap. Their presence before death '
            'does not prove the situation caused or guaranteed the collision. '
            'Risk is measured in shortest-path grid steps, not Manhattan distance.'
        )
    report = {
        'version': '0.9.2-safety-inspection' if risk_analysis else '0.9.1-inspection',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(checkpoint),
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'architecture': spec.architecture,
        'maze': maze, 'seed_start': seed, 'episodes': episodes,
        'engine': engine, 'device': str(device), 'worlds': worlds,
        'policy': 'greedy (argmax Q, no exploratory actions)',
        'notes': [
            'Timeout is max_steps; ghost collisions are terminal.',
            'player_entered_ghost: player stepped onto a ghost present before the step.',
            'ghost_moved_or_swapped: contact caused by a ghost movement, including swaps.',
            'Heuristics (e.g. 100 steps without a pellet) indicate symptoms, not root causes.',
            'Runs are on the same known maze; no validation on unseen maps is implied.',
            'With risk analysis: the previous 20 pre-action states are tracked. time_windows separates 11-20, 6-10 and 1-5 actions before episode end.',
        ],
        'summary': summary, 'episodes_detail': results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"INSPECT {spec.architecture}: {wins}/{episodes} wins ({summary['win_percent']:.2f}%), "
          f"pellets {summary['pellet_percent']:.2f}%")
    print(f"Losses: {len(losses)} = " + ', '.join(f'{k}: {v}' for k, v in sorted(reasons.items())))
    print(f"Near-win loss seeds: {summary['near_win_loss_seeds']}")
    print(f"First loss seeds: {summary['first_loss_seeds']}")
    if risk_analysis:
        print('Risks during last five actions of losing games:', summary['loss_risk_last_5'])
        print('Same risk flags during last five actions of winning games:', summary['win_risk_last_5'])
        for window, groups in summary['risk_time_windows'].items():
            print('Risk',window, groups)
    print(f'Report: {output}')
    return report


def watch(checkpoint: Path, *, maze='large', seed=80000, fps=10,
          device_name='auto', pause_on_end=False, report: Path | None = None,
          max_frames: int | None = None):
    import pygame
    import torch
    if fps < 1 or seed < 0:
        raise ValueError('fps must be >= 1 and seed >= 0')
    if max_frames is not None and max_frames < 1:
        raise ValueError('max_frames must be >= 1')
    model, spec, shape, device, payload = _load(Path(checkpoint).resolve(strict=True), device_name)
    _assert_compatible(shape, maze)
    seeds = None
    if report is not None:
        data = json.loads(Path(report).read_text(encoding='utf-8'))
        if Path(data['checkpoint']).resolve() != Path(checkpoint).resolve():
            raise ValueError('Report was created with a different checkpoint')
        if data['maze'] != maze:
            raise ValueError('Report maze differs from selected maze')
        seeds = [int(x['seed']) for x in data['episodes_detail'] if not x['won']]
        if not seeds:
            print('No losses in report to replay!')
            return
        seed = seeds[0]
        print(f'Watching {len(seeds)} losing seeds from report; N to show next loss')

    game = PacmanGame(maze=maze, ghosts=2, seed=seed)
    hist = observer_for(spec.architecture, spec.frames)
    state = hist.reset(game.observe())
    hidden = None
    chosen = None
    qvalues = [0.0] * 4
    last_reason = ''
    episode_index = 0
    done_at_ms = None
    paused = False
    diagnostics = True
    names = ('UP', 'DOWN', 'LEFT', 'RIGHT')
    pygame.init()
    tile = 37 if game.width <= 13 else 30
    board_w = game.width * tile
    board_h = game.height * tile
    sidebar = 310
    hud = 110
    screen = pygame.display.set_mode((board_w+sidebar, board_h+hud))
    pygame.display.set_caption(f'Pac-Man CNN viewer | {spec.architecture} | {maze}')
    font = pygame.font.SysFont('consolas', 18)
    small = pygame.font.SysFont('consolas', 15)
    bold = pygame.font.SysFont('consolas', 21, bold=True)
    clock = pygame.time.Clock()

    def restart(next_seed):
        nonlocal game, hist, state, hidden, chosen, last_reason, done_at_ms, seed
        seed = next_seed
        game.reset(seed=seed)
        state = hist.reset(game.observe())
        hidden = None
        chosen = None
        last_reason = ''
        done_at_ms = None
        print(f'WATCH seed={seed} maze={maze}')

    def next_episode():
        nonlocal episode_index
        if seeds is not None:
            episode_index = (episode_index+1) % len(seeds)
            restart(seeds[episode_index])
        else:
            restart(seed+1)

    def put(t, x, y, color=(230, 238, 246), mini=False, headline=False):
        source = bold if headline else small if mini else font
        screen.blit(source.render(str(t), True, color), (x, y))

    try:
        running = True
        displayed_frames = 0
        while running:
            advance_one = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        restart(seed)
                    elif event.key == pygame.K_n:
                        next_episode()
                    elif event.key == pygame.K_s:
                        advance_one = True
                        paused = True
                    elif event.key == pygame.K_d:
                        diagnostics = not diagnostics
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        fps = min(60, fps+2)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        fps = max(1, fps-2)
            if not running:
                break
            if game.done and not pause_on_end and done_at_ms is not None and pygame.time.get_ticks()-done_at_ms >= 1500:
                next_episode()
            if not game.done and (not paused or advance_one):
                with torch.inference_mode():
                    obs = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
                    if spec.recurrent:
                        q, hidden = model(obs[:,None], hidden)
                        q = q[0,-1]
                    else:
                        q = model(obs)[0]
                    qvalues = q.detach().cpu().tolist()
                    chosen = int(q.argmax().item())
                before = game.observe()
                result = game.step(chosen)
                state = hist.step(result.observation)
                if game.done:
                    last_reason = ('win' if game.won else
                                   _death_type(before, result.observation, result.truncated))
                    print(f'END seed={seed} reason={last_reason} ' +
                          f'pellets={100*game.collected/max(1,game.initial_pellet_count):.1f}% '
                          f'steps={game.steps}')
                    done_at_ms = pygame.time.get_ticks()
            screen.fill((9, 15, 29))
            for r in range(game.height):
                for c in range(game.width):
                    x, y = c*tile, r*tile
                    if (r,c) in game.walls:
                        pygame.draw.rect(screen, (34, 79, 174), (x+1,y+1,tile-2,tile-2), border_radius=5)
                    elif (r,c) in game.pellets:
                        pygame.draw.circle(screen, (250,213,137), (x+tile//2,y+tile//2),3)
            pr,pc = game.player
            pygame.draw.circle(screen,(255,212,48),(pc*tile+tile//2,pr*tile+tile//2),tile//2-4)
            for i,(gr,gc) in enumerate(game.ghosts):
                col = ((255,102,113),(105,222,246))[i%2]
                rect = pygame.Rect(gc*tile+5,gr*tile+5,tile-10,tile-10)
                pygame.draw.rect(screen,col,rect,border_top_left_radius=10,border_top_right_radius=10)
                for dx in (-5,5):
                    pygame.draw.circle(screen,(255,255,255),(gc*tile+tile//2+dx,gr*tile+tile//2-2),3)
            sx=board_w+14
            pygame.draw.rect(screen,(24,34,58),(board_w,0,sidebar,board_h+hud))
            put('CNN checkpoint replay',sx,16,headline=True)
            put(spec.architecture,sx,48)
            put(f'Seed {seed} | {fps} FPS',sx,80)
            put(f'Pellets {game.collected}/{game.initial_pellet_count}',sx,110)
            put(f'Steps {game.steps}/{game.max_steps}',sx,138)
            put('Q-values (last decision):',sx,187,mini=True)
            if diagnostics and chosen is not None:
                for a,n in enumerate(names):
                    color=(85,233,173) if a==chosen else (227,236,245)
                    put(f'{">" if a==chosen else " "} {n:>5}  {qvalues[a]:+8.2f}',sx,211+30*a,color)
            if game.done:
                final='WIN' if game.won else f'LOSS: {last_reason}'
                put(final,sx,347,color=(85,233,173) if game.won else (251,127,127))
                if pause_on_end:
                    put('N = next episode',sx,376,mini=True)
            elif paused:
                put('PAUSED',sx,348)
            put(f'Space pause | S step | N next',12,board_h+12,mini=True)
            put('R replay | +/- speed | D Q | ESC exit',12,board_h+36,mini=True)
            put(f'Walls {game.wall_bumps} | Reversals {game.reversals}',12,board_h+60,mini=True)
            pygame.display.flip()
            displayed_frames += 1
            if max_frames is not None and displayed_frames >= max_frames:
                break
            clock.tick(fps)
    finally:
        pygame.quit()


def main():
    p = argparse.ArgumentParser(description='View CNN/GRU playing or diagnose why episodes fail')
    commands = p.add_subparsers(dest='command', required=True)
    w = commands.add_parser('watch', help='Open a Pygame window; N next, Space pause, R replay')
    w.add_argument('--checkpoint', type=Path, required=True)
    w.add_argument('--maze', choices=tuple(MAPS), default='large')
    w.add_argument('--seed', type=int, default=80000)
    w.add_argument('--fps', type=int, default=10)
    w.add_argument('--device', choices=('auto','cpu','cuda','mps'), default='auto')
    w.add_argument('--max-frames', type=int, help='Exit after N display frames (GUI smoke test)')
    w.add_argument('--report', type=Path, help='Show only losing seeds from an audit report')
    w.add_argument('--pause-on-end', action='store_true')
    a = commands.add_parser('audit', help='Count collision vs timeout, output seeds to replay')
    a.add_argument('--checkpoint', type=Path, required=True)
    a.add_argument('--maze', choices=tuple(MAPS), default='large')
    a.add_argument('--seed', type=int, default=80000, help='Episode seed base (v0.9 --test-seed)')
    a.add_argument('--episodes', type=int, default=1000)
    a.add_argument('--worlds', type=int, default=16)
    a.add_argument('--engine', choices=('rust','python'), default='rust')
    a.add_argument('--device', choices=('auto','cpu','cuda','mps'), default='auto')
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--risk-analysis', action='store_true', help='Record ghost topology and escape-route risks before every terminal event')
    opts = p.parse_args()
    if opts.command=='watch':
        watch(opts.checkpoint,maze=opts.maze,seed=opts.seed,fps=opts.fps,
              device_name=opts.device,pause_on_end=opts.pause_on_end,
              report=opts.report,max_frames=opts.max_frames)
    else:
        audit(opts.checkpoint,maze=opts.maze,seed=opts.seed,
              episodes=opts.episodes,worlds=opts.worlds,engine=opts.engine,
              device_name=opts.device,output=opts.output,
              risk_analysis=opts.risk_analysis)


if __name__ == '__main__':
    main()
