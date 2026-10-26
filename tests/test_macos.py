# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU tests run everywhere; MPS tests run when the process has Metal access."""
import json
import pytest


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('architecture', ['mlp_wide', 'cnn_single', 'cnn_motion', 'cnn_history4', 'cnn_gru'])
def test_all_architectures_have_finite_gradients_and_an_adam_update(device, architecture):
    torch = pytest.importorskip('torch')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS requires an Apple GPU and Metal access')
    from pacman_lab.macos import gradient_probe
    result = gradient_probe(architecture, device)
    assert result['cpu_q_parity'] and result['finite_gradients']
    assert result['optimizer_updates'] == 1


def test_auto_selects_mps_and_explicit_request_fails_clearly_without_metal(monkeypatch):
    torch = pytest.importorskip('torch')
    from pacman_lab.dqn import select_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert select_device('auto').type == 'mps'
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert select_device('auto').type == 'cpu'
    with pytest.raises(RuntimeError, match='Metal GPU access'):
        select_device('mps')


@pytest.mark.parametrize('body_fails', [False, True])
def test_caffeinate_is_released_after_suite_and_on_error(monkeypatch, body_fails):
    from pacman_lab import marathon
    events = []

    class Process:
        def terminate(self):
            events.append('terminate')

        def wait(self, timeout):
            events.append('wait')

    def launch(args):
        assert args[:3] == ['/usr/bin/caffeinate', '-i', '-w']
        assert int(args[3]) > 0
        events.append('start')
        return Process()

    monkeypatch.setattr(marathon.sys, 'platform', 'darwin')
    monkeypatch.setattr(marathon.subprocess, 'Popen', launch)
    try:
        with marathon.prevent_sleep():
            events.append('body')
            if body_fails:
                raise ValueError('body')
    except ValueError:
        assert body_fails
    assert events == ['start', 'body', 'terminate', 'wait']


def test_profile_report_never_compares_rust_mps_with_python_cpu(tmp_path):
    from pacman_lab.profile_report import create_report
    for engine, device, speed in [('python', 'cpu', 100), ('rust', 'mps', 200)]:
        profile = {'benchmark': 'pacman_rl_lab_v0.7_pipeline_profile_v1',
                   'settings': {'architecture': 'cnn_motion', 'engine': engine, 'device': device},
                   'steps_per_second': speed, 'counts': {'updates': 1},
                   'timings_percent': {'game_step_and_reset': 100},
                   'gpu_peak_allocated_mb': None}
        (tmp_path / f'profile_{engine}.json').write_text(json.dumps(profile))
    create_report(tmp_path)
    report = (tmp_path / 'PROFILE_REPORT.md').read_text()
    assert 'Rust/Python =' not in report
    assert '**cnn_motion, cpu:** both Python and Rust' in report
    assert '**cnn_motion, mps:** both Python and Rust' in report
    assert '| cnn_motion | mps | rust |' in report
    assert '| n/a |' in report


def test_smoke_is_bounded_before_creating_results(tmp_path):
    from pacman_lab.macos import run_smoke
    for steps in (0, 257, 1_000_000):
        with pytest.raises(ValueError, match='16..256'):
            run_smoke(output=tmp_path / 'not-created', steps=steps)
    assert not (tmp_path / 'not-created').exists()


def test_smoke_roundtrip_reports_real_updates_and_never_overwrites(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from pacman_lab import macos
    monkeypatch.setattr(macos, 'gradient_probe', lambda architecture, device:
                        {'architecture': architecture, 'device': device})
    report = macos.run_smoke(output=tmp_path / 'new-smoke', device_name='cpu',
                             engine='python', steps=16)
    assert len(report['runs']) == 2
    assert all(run['optimizer_updates'] > 0 and run['checkpoint_roundtrip'] for run in report['runs'])
    assert all(run['transitions'] == 16 for run in report['runs'])
    with pytest.raises(FileExistsError):
        macos.run_smoke(output=tmp_path / 'new-smoke', device_name='cpu', engine='python', steps=16)


def test_pygame_random_viewer_exits_after_requested_display_frames(monkeypatch):
    pytest.importorskip('pygame')
    from pacman_lab import play
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    monkeypatch.setattr('sys.argv', ['play', '--mode', 'random', '--max-frames', '2'])
    play.main()


@pytest.mark.parametrize('architecture', ['cnn_motion', 'cnn_gru'])
def test_cnn_viewer_can_render_motion_and_recurrent_models(tmp_path, monkeypatch, architecture):
    torch = pytest.importorskip('torch')
    pytest.importorskip('pygame')
    from pacman_lab.cnn_inspect import watch
    from pacman_lab.cnn_models import make_model
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    shape = (5 if architecture == 'cnn_motion' else 4, 7, 9)
    checkpoint = tmp_path / 'random_fixture.pt'
    torch.save({'architecture': architecture, 'input_shape': shape,
                'model': make_model(architecture, shape).state_dict()}, checkpoint)
    watch(checkpoint, maze='small', device_name='cpu', max_frames=2)
