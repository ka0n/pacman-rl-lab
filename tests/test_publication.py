# SPDX-License-Identifier: GPL-3.0-or-later
import importlib.util
from pathlib import Path


def checker():
    path = Path(__file__).parents[1] / 'scripts' / 'check_publication.py'
    spec = importlib.util.spec_from_file_location('publication', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.problems


def test_publication_allows_source_and_recorded_experiment_results():
    check = checker()
    assert check('pacman_lab/core.py', b'print("gridworld")') == []
    assert check('docs/summary_620000.json', b'{"win_percent": 89.53}') == []


def test_publication_blocks_weights_generated_logs_and_binary_files():
    check = checker()
    for path in ['runs/private.pt', 'docs/validation/run.json', '.env.local',
                 'native/pacman_native/target/lib.dylib', 'output.log']:
        assert check(path, b'local data')
    assert check('image.dat', b'\0binary')


def test_publication_blocks_secrets_without_printing_them():
    check = checker()
    value = ('ghp_' + 'x' * 40).encode()
    reasons = check('config.txt', value)
    assert reasons == ['credential-like content']
    assert not any(value.decode() in reason for reason in reasons)


def test_publication_blocks_personal_machine_paths():
    check = checker()
    path = ('/Users/' + 'example-person' + '/private/config').encode()
    assert check('README.md', path) == ['personal machine path']
