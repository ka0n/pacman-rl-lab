# SPDX-License-Identifier: GPL-3.0-or-later
"""Pygame viewer with optional DQN Q-value diagnostics (press D)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .core import MAPS, PacmanGame
from .memory import FrameHistory, infer_frames


def main():
    parser = argparse.ArgumentParser(description="Watch or play Pac-Man RL Lab")
    parser.add_argument("--mode", choices=("manual", "random", "dqn"), default="manual")
    parser.add_argument("--maze", choices=tuple(MAPS), default=None,
                        help="Defaults to checkpoint training maze for DQN, small otherwise. "
                             "Use unseen_a or unseen_b for zero-shot play.")
    parser.add_argument("--ghosts", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--ghost-every", type=int, default=1)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-frames", type=int, help="Exit after N display frames (GUI smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be >= 1")
    if args.mode == "dqn" and not args.checkpoint:
        parser.error("--mode dqn requires --checkpoint path")

    import pygame
    if args.mode == "dqn":
        import torch

    rng = random.Random(args.seed)
    network = None
    device = None
    history = FrameHistory(1)
    if args.mode == "dqn":
        from .dqn import make_qnet, select_device
        device = select_device(args.device)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if args.maze is None:
            args.maze = payload["maze"]
        frames = infer_frames(tuple(payload["input_shape"]))
        history = FrameHistory(frames)
        expected = (4 * frames, len(MAPS[args.maze]), len(MAPS[args.maze][0]))
        if tuple(payload["input_shape"]) != expected:
            parser.error(f"Checkpoint expects {tuple(payload['input_shape'])}, "
                         f"but maze {args.maze!r} has observation shape {expected}")
        network = make_qnet(tuple(payload["input_shape"])).to(device)
        network.load_state_dict(payload["model"])
        network.eval()

    if args.maze is None:
        args.maze = "small"
    game = PacmanGame(maze=args.maze, ghosts=args.ghosts,
                      ghost_every=args.ghost_every, seed=args.seed)
    history.reset(game.observe())
    tile = 38 if game.width <= 9 else 32
    hud = 96
    board_width = game.width * tile
    sidebar_width = 276 if network is not None else 0
    pygame.init()
    screen = pygame.display.set_mode((board_width + sidebar_width, game.height * tile + hud))
    pygame.display.set_caption(f"Pac-Man RL Lab v0.5 [{args.maze}] | R restart, SPACE pause, D diagnostics, ESC quit")
    font = pygame.font.SysFont("consolas", 17)
    small_font = pygame.font.SysFont("consolas", 14)
    headline_font = pygame.font.SysFont("consolas", 19, bold=True)
    clock = pygame.time.Clock()
    wall_color = (32, 76, 170)
    empty_color = (8, 13, 31)
    text_color = (238, 236, 226)
    muted_color = (180, 195, 214)
    green_color = (63, 220, 170)
    running = True
    displayed_frames = 0
    paused = False
    show_diagnostics = True
    episode = 0
    last_action = None
    manual_actions = {
        pygame.K_UP: 0, pygame.K_w: 0,
        pygame.K_DOWN: 1, pygame.K_s: 1,
        pygame.K_LEFT: 2, pygame.K_a: 2,
        pygame.K_RIGHT: 3, pygame.K_d: 3,
    }
    names = ("UP", "DOWN", "LEFT", "RIGHT")
    while running:
        choice = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_d:
                    show_diagnostics = not show_diagnostics
                elif event.key == pygame.K_r:
                    episode += 1
                    game.reset(seed=args.seed + episode)
                    history.reset(game.observe())
                    paused = False
                    last_action = None
                elif args.mode == "manual" and event.key in manual_actions:
                    choice = manual_actions[event.key]
        if not running:
            break
        if not paused and not game.done:
            if args.mode == "random":
                choice = rng.randrange(4)
            elif network is not None:
                with torch.inference_mode():
                    obs = torch.as_tensor(history.observe(), dtype=torch.float32,
                                          device=device).unsqueeze(0)
                    choice = int(network(obs).argmax(dim=1).item())
            if choice is not None:
                last_action = choice
                outcome = game.step(choice)
                history.step(outcome.observation)

        screen.fill(empty_color)
        for r in range(game.height):
            for c in range(game.width):
                x, y = c * tile, r * tile
                if (r, c) in game.walls:
                    pygame.draw.rect(screen, wall_color,
                                     (x + 1, y + 1, tile - 2, tile - 2), border_radius=4)
                elif (r, c) in game.pellets:
                    pygame.draw.circle(screen, (255, 226, 170),
                                       (x + tile // 2, y + tile // 2), max(2, tile // 12))
        pr, pc = game.player
        pygame.draw.circle(screen, (255, 216, 44),
                           (pc * tile + tile // 2, pr * tile + tile // 2), tile // 2 - 4)
        for i, (gr, gc) in enumerate(game.ghosts):
            color = ((252, 99, 109), (112, 215, 240))[i % 2]
            rect = pygame.Rect(gc * tile + 5, gr * tile + 5, tile - 10, tile - 10)
            pygame.draw.rect(screen, color, rect, border_top_left_radius=12,
                             border_top_right_radius=12)
            pygame.draw.circle(screen, (255, 255, 255),
                               (gc * tile + tile // 2 - 5, gr * tile + tile // 2 - 1), 3)
            pygame.draw.circle(screen, (255, 255, 255),
                               (gc * tile + tile // 2 + 5, gr * tile + tile // 2 - 1), 3)
        start_y = game.height * tile + 8
        line = (f"{args.mode.upper()} | Pellets {game.collected}/{game.initial_pellet_count} "
                f"| Steps {game.steps}")
        screen.blit(font.render(line, True, text_color), (9, start_y))
        line2 = f"Wall {game.wall_bumps} | Rev {game.reversals} | Stall {game.stall_streak}"
        screen.blit(small_font.render(line2, True, muted_color), (9, start_y + 25))
        if game.done:
            status = "WIN! R restart" if game.won else "GAME OVER / TIMEOUT - R restart"
        elif paused:
            status = "PAUSED - SPACE resume"
        else:
            status = "SPACE pause | R reset | D Q | ESC quit"
        screen.blit(small_font.render(status, True, muted_color), (9, start_y + 47))

        if network is not None:
            sx = board_width + 14
            pygame.draw.rect(screen, (21, 30, 53), (board_width, 0, sidebar_width, game.height * tile + hud))
            screen.blit(headline_font.render(f"DQN diagnostics | {history.frames} frame(s)", True, text_color), (sx, 13))
            if show_diagnostics:
                with torch.inference_mode():
                    obs = torch.as_tensor(history.observe(), dtype=torch.float32,
                                          device=device).unsqueeze(0)
                    qvals = network(obs)[0].detach().cpu().tolist()
                chosen = max(range(4), key=lambda i: qvals[i])
                if last_action is not None:
                    screen.blit(small_font.render(f"Last move: {names[last_action]}", True, muted_color), (sx, 45))
                screen.blit(small_font.render("NEXT move estimates:", True, muted_color), (sx, 76))
                low, high = min(qvals), max(qvals)
                legal = {a for a, _ in game.legal_destinations(game.player)}
                for i, label in enumerate(names):
                    y = 104 + 49 * i
                    marker = ">" if i == chosen else " "
                    blocked = "WALL" if i not in legal else ""
                    color = green_color if i == chosen else text_color
                    screen.blit(font.render(f"{marker} {label:<5} {qvals[i]:+7.3f}", True, color), (sx, y))
                    fraction = (qvals[i] - low) / max(1e-6, high - low)
                    pygame.draw.rect(screen, (49, 64, 88), (sx + 12, y + 24, 178, 8), border_radius=3)
                    pygame.draw.rect(screen, color, (sx + 12, y + 24, int(fraction * 178), 8), border_radius=3)
                    if blocked:
                        screen.blit(small_font.render(blocked, True, (253, 135, 135)), (sx + 195, y + 19))
                line_y = 310
                for info_line in (
                    "Highest Q wins (walls allowed)",
                    "Q is not a probability. D: hide",
                ):
                    screen.blit(small_font.render(info_line, True, muted_color), (sx, line_y))
                    line_y += 21
            else:
                screen.blit(font.render("Press D to show Q values", True, muted_color), (sx, 54))
        pygame.display.flip()
        displayed_frames += 1
        if args.max_frames is not None and displayed_frames >= args.max_frames:
            break
        clock.tick(args.fps if args.mode != "manual" else 30)
    pygame.quit()


if __name__ == "__main__":
    main()
