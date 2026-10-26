# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion audit: observation/replay invariants and analytical learning checks.

No experiment or optimizer updates: the trainer integration test uses scripted
worlds, disables learning, and intercepts checkpoint writes. Backward-only tests
inspect gradients on random weights, not the unavailable Windows checkpoints.
"""
from collections import deque
import random

import numpy as np
import pytest

from pacman_lab.cnn_motion import GhostMotionHistory, current_frame, observer_for
from pacman_lab.core import PacmanGame


def frame(t, world=0):
    obs = np.zeros((4, 7, 9), dtype=np.uint8)
    obs[0, 0, :] = 1
    obs[1, 2, t % 7 + 1] = 1
    obs[2, 1, t % 7 + 1] = 1
    obs[3, world + 3, t % 7 + 1] = 1
    return obs


def test_motion_snapshots_are_owned_and_all_current_planes_stay_aligned():
    history = GhostMotionHistory()
    raw = frame(0)
    s0 = history.reset(raw)
    raw[:] = 0  # Simulator storage may be reused after handing off a frame.
    np.testing.assert_array_equal(history.observe()[:4], frame(0))
    saved = s0.copy()
    for t in range(1, 10):
        state = history.step(frame(t))
        assert state.dtype == np.uint8
        assert set(np.unique(state)) <= {0, 1}
        np.testing.assert_array_equal(state[:4], frame(t))
        np.testing.assert_array_equal(state[4], frame(t - 1)[3])
        np.testing.assert_array_equal(current_frame(state, 'cnn_motion'), frame(t))
        state[:] = 0  # A consumer cannot corrupt history through its result.
        np.testing.assert_array_equal(history.observe()[:4], frame(t))
    np.testing.assert_array_equal(s0, saved)
    fresh = history.reset(frame(4, world=2))
    np.testing.assert_array_equal(fresh[4], fresh[3])
    assert not np.shares_memory(fresh, s0)


@pytest.mark.parametrize('ghost_every', [1, 2])
@pytest.mark.parametrize('engine', ['python', 'rust'])
def test_real_game_motion_alignment_through_asynchronous_resets(engine, ghost_every):
    from pacman_lab.cnn_batch import PythonBatch
    from pacman_lab.native import NativeBatch, _load_library
    if engine == 'rust':
        try:
            _load_library()
        except FileNotFoundError:
            pytest.skip('Rust library not built; Python coverage runs independently')
        env = NativeBatch('large', count=3, ghosts=2, ghost_every=ghost_every, max_steps=9)
    else:
        env = PythonBatch('large', 3, ghosts=2)
        for game in env.games:
            game.ghost_every = ghost_every
            game.max_steps = 9
    histories = [GhostMotionHistory() for _ in range(3)]
    controls = [PacmanGame('large', ghosts=2, ghost_every=ghost_every, max_steps=9)
                for _ in range(3)]
    previous = []
    reset_counts = [0, 0, 0]
    actions_rng = random.Random(8711)
    try:
        for i in range(3):
            raw, _ = env.reset_one(i, seed=8700 + i)
            expected, _ = controls[i].reset(seed=8700 + i)
            np.testing.assert_array_equal(raw, expected)
            previous.append(raw.copy())
            histories[i].reset(raw)
        for step in range(40):
            actions = [actions_rng.randrange(4) for _ in range(3)]
            outcome = env.step(actions)
            for i in range(3):
                expected = controls[i].step(actions[i])
                np.testing.assert_array_equal(outcome.observations[i], expected.observation)
                assert outcome.rewards[i] == pytest.approx(expected.reward)
                assert bool(outcome.terminated[i]) == expected.terminated
                assert bool(outcome.truncated[i]) == expected.truncated
                state = histories[i].step(outcome.observations[i])
                np.testing.assert_array_equal(state[:4], expected.observation)
                np.testing.assert_array_equal(state[4], previous[i][3])
                # Force staggered resets too, including before episode completion.
                if expected.terminated or expected.truncated or (step + i) % 7 == 0:
                    seed = 8800 + step * 3 + i
                    raw, _ = env.reset_one(i, seed=seed)
                    controls[i].reset(seed=seed)
                    reset = histories[i].reset(raw)
                    np.testing.assert_array_equal(reset[3], reset[4])
                    previous[i] = raw.copy()
                    reset_counts[i] += 1
                else:
                    previous[i] = expected.observation.copy()
        assert min(reset_counts) > 2
    finally:
        env.close()


@pytest.mark.parametrize('architecture', ['cnn_single', 'cnn_motion'])
def test_batched_replay_keeps_terminal_frames_and_world_history(tmp_path, monkeypatch, architecture):
    torch = pytest.importorskip('torch')
    from pacman_lab import cnn_batch as trainer
    from pacman_lab.native import BatchOutcome

    expected = []
    appended = []
    buffers = []
    reset_seeds = []

    class CaptureDeque(deque):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.maxlen == 6:
                buffers.append(self)

        def append(self, item):
            if self.maxlen == 6:
                appended.append(item)
            super().append(item)

    class ScriptedBatch:
        height, width = 7, 9

        def __init__(self, **kwargs):
            self.ticks = [0, 0, 0]
            self.episodes = [-1, -1, -1]
            self.states = [None] * 3
            self.past = [None] * 3
            self.calls = 0
            self.closed = False

        def reset_one(self, i, seed):
            reset_seeds.append(seed)
            self.ticks[i] = 0
            self.episodes[i] += 1
            raw = frame(4 * self.episodes[i], i)
            self.states[i] = raw.copy()
            self.past[i] = raw[3].copy()
            return raw, {}

        def step(self, actions):
            self.calls += 1
            raws, rewards, terms, truncs, stats = [], [], [], [], []
            for i in range(3):
                self.ticks[i] += 1
                t = self.ticks[i]
                raw = frame(4 * self.episodes[i] + t, i)
                # World 0: death; world 1: timeout; world 2: win.
                term = (i == 0 and t == 2) or i == 2
                trunc = i == 1 and t == 2
                won = i == 2
                base = -5.01 if i == 0 and term else (10.99 if won else -0.01)
                reward = base - 15 if term and not won else base
                if self.calls <= 2 or i == 0:  # Budget = 3 + 3 + 1 transitions.
                    state = self.states[i].copy()
                    ns = raw.copy()
                    if architecture == 'cnn_motion':
                        state = np.concatenate([state, self.past[i][None]])
                        ns = np.concatenate([ns, self.states[i][3:4]])
                    expected.append((state, int(actions[i]), reward, ns, term or trunc))
                self.past[i] = self.states[i][3].copy()
                self.states[i] = raw.copy()
                raws.append(raw)
                rewards.append(base)
                terms.append(term)
                truncs.append(trunc)
                stats.append({'won': won})
            return BatchOutcome(np.stack(raws), np.array(rewards), np.array(terms),
                                np.array(truncs), stats)

        def close(self):
            self.closed = True

    env = ScriptedBatch()
    monkeypatch.setattr(trainer, 'PythonBatch', lambda **kwargs: env)
    monkeypatch.setattr(trainer, 'deque', CaptureDeque)
    # Avoid constructing/learning an actual CNN in this control-flow test.
    monkeypatch.setattr(trainer, 'make_model', lambda *args: torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear((5 if architecture == 'cnn_motion' else 4) * 7 * 9, 4)))
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda *args, **kw: pytest.fail('No training authorized'))
    monkeypatch.setattr(trainer, 'eval_batched', lambda *args, **kwargs:
                        {'win_percent': 0., 'pellet_percent': 0.})
    monkeypatch.setattr(trainer, 'atomic_save_torch', lambda *args: None)
    trainer.run_batched(architecture=architecture, steps=7, seed=441, worlds=3,
                        output=tmp_path, device_name='cpu', engine='python',
                        replay_capacity=6, batch_size=1, warmup=100,
                        eval_every=7, save_every=7, eval_episodes=1, death_penalty=-20.)
    assert env.closed
    assert len(appended) == len(expected) == 7
    assert len(buffers[0]) == 6  # Capacity counts transitions, not raw channels.
    assert reset_seeds == list(range(1_000_441, 1_000_441 + len(reset_seeds)))
    for actual, reference in zip(appended, expected):
        np.testing.assert_array_equal(actual[0], reference[0])
        assert actual[1:3] == reference[1:3]
        np.testing.assert_array_equal(actual[3], reference[3])
        assert actual[4] == reference[4]
        assert not np.shares_memory(actual[0], actual[3])
    assert [x[4] for x in appended] == [False, False, True, True, True, True, False]
    # Random replay order cannot change history already stored within each item.
    for item in random.Random(1).sample(buffers[0], 6):
        if architecture == 'cnn_motion':
            np.testing.assert_array_equal(item[3][4], item[0][3])


@pytest.mark.parametrize('architecture', ['cnn_single', 'cnn_motion'])
@pytest.mark.parametrize('done', [False, True])
def test_double_dqn_uses_online_argmax_target_value_and_masks_done(architecture, done):
    torch = pytest.importorskip('torch')
    from pacman_lab.cnn_train import minibatch_loss

    class TableQ(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, observations):
            # Player column 2 identifies s_next; no dependence on previous ghosts.
            return self.values[observations[:, 2, 1, 2].long()]

    history = observer_for(architecture, 1)
    state = history.reset(frame(0))
    ns = history.step(frame(1))
    online = TableQ([[1, 2, 3, 4], [9, 8, 2, 1]])
    target = TableQ([[0, 0, 0, 0], [10, 50, 0, 0]])
    loss = minibatch_loss(online, target, [(state, 2, 1., ns, done)],
                          architecture, torch.device('cpu'))
    # Online chooses a=0; target Q=10 (target's own max=50 must not be used).
    y = 1. if done else 1. + .99 * 10.
    error = abs(3. - y)
    expected = .5 * error**2 if error < 1 else error - .5
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert target.values.grad is None
    assert torch.count_nonzero(online.values.grad).item() == 1
    assert online.values.grad[0, 2] != 0


@pytest.mark.parametrize('at_reset', [True, False])
def test_matched_models_have_equal_losses_shared_gradients_and_live_extra_channel(at_reset):
    torch = pytest.importorskip('torch')
    from pacman_lab.cnn_models import make_model, parameters_count
    from pacman_lab.cnn_motion import transfer_cnn_single_to_motion
    from pacman_lab.cnn_train import minibatch_loss
    torch.set_num_threads(1)
    torch.manual_seed(491)
    control = make_model('cnn_single', (4, 11, 13))
    motion = make_model('cnn_motion', (5, 11, 13))
    transfer_cnn_single_to_motion(control.state_dict(), motion)
    target_control = make_model('cnn_single', (4, 11, 13))
    target_control.load_state_dict(control.state_dict())
    target_motion = make_model('cnn_motion', (5, 11, 13))
    target_motion.load_state_dict(motion.state_dict())
    game = PacmanGame('large', ghosts=2, seed=491)
    raw = game.observe()
    history = GhostMotionHistory()
    # Identical planes at reset isolate correlated-input gradients; real moving
    # frames also check nonterminal bootstrapping through the transferred target.
    augmented = history.reset(raw)
    ns_raw, ns_augmented = raw, augmented
    if not at_reset:
        raw = game.step(3).observation
        augmented = history.step(raw)
        ns_raw = game.step(3).observation
        ns_augmented = history.step(ns_raw)
    control_samples = [(raw, 0, 5., ns_raw, at_reset)]
    motion_samples = [(augmented, 0, 5., ns_augmented, at_reset)]
    lc = minibatch_loss(control, target_control, control_samples, 'cnn_single', torch.device('cpu'))
    lm = minibatch_loss(motion, target_motion, motion_samples, 'cnn_motion', torch.device('cpu'))
    torch.testing.assert_close(lc, lm)
    lc.backward()
    lm.backward()
    for (name, pc), (other_name, pm) in zip(control.named_parameters(), motion.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(pc.grad, pm.grad[:, :4] if name == '0.conv.0.weight' else pm.grad)
    grad = motion[0].conv[0].weight.grad
    assert torch.count_nonzero(grad[:, 4]).item() > 0
    if at_reset:
        torch.testing.assert_close(grad[:, 3], grad[:, 4])
    else:
        assert not torch.equal(grad[:, 3], grad[:, 4])
    assert parameters_count(control) == 4_976_708
    assert parameters_count(motion) - parameters_count(control) == 576
    assert (all(p.grad is None for p in target_control.parameters()) and
            all(p.grad is None for p in target_motion.parameters()))


@pytest.mark.parametrize('seed', [930042, 930043, 930044])
def test_actual_trainer_initializes_matched_online_target_and_fresh_adam(tmp_path, monkeypatch, seed):
    torch = pytest.importorskip('torch')
    from copy import deepcopy
    from pacman_lab import cnn_batch as trainer
    from pacman_lab.cnn_models import make_model
    checkpoints = {}

    def capture(payload, path):
        checkpoints[payload['architecture']] = deepcopy(payload)

    monkeypatch.setattr(trainer, 'atomic_save_torch', capture)
    monkeypatch.setattr(trainer, 'eval_batched', lambda *args, **kwargs:
                        {'win_percent': 0., 'pellet_percent': 0.})
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda *args, **kw: pytest.fail('No training authorized'))
    for architecture in ('cnn_single', 'cnn_motion'):
        trainer.run_batched(architecture=architecture, steps=1, seed=seed,
                            output=tmp_path / architecture, worlds=1, device_name='cpu',
                            engine='python', maze='large', replay_capacity=1,
                            batch_size=1, warmup=100, eval_episodes=1,
                            matched_scratch_init=architecture == 'cnn_motion')
    control, motion = (checkpoints[a] for a in ('cnn_single', 'cnn_motion'))
    for checkpoint in (control, motion):
        assert checkpoint['optimizer']['state'] == {}
        assert checkpoint['fine_tune'] is None
        for name, value in checkpoint['model'].items():
            torch.testing.assert_close(value, checkpoint['target'][name], rtol=0, atol=0)
    for name, value in control['model'].items():
        counterpart = motion['model'][name]
        torch.testing.assert_close(value, counterpart[:, :4] if name == '0.conv.0.weight' else counterpart,
                                   rtol=0, atol=0)
    assert torch.count_nonzero(motion['model']['0.conv.0.weight'][:, 4]).item() == 0
    baseline_model = make_model('cnn_single', (4, 11, 13))
    motion_model = make_model('cnn_motion', (5, 11, 13))
    baseline_model.load_state_dict(control['model'])
    motion_model.load_state_dict(motion['model'])
    raw = PacmanGame('large', ghosts=2, seed=seed).observe()
    augmented = GhostMotionHistory().reset(raw)
    # Deliberately vary history: all initial Q-values must still be identical.
    augmented[4] = 1 - augmented[4]
    with torch.no_grad():
        torch.testing.assert_close(baseline_model(torch.tensor(raw[None], dtype=torch.float32)),
                                   motion_model(torch.tensor(augmented[None], dtype=torch.float32)))
