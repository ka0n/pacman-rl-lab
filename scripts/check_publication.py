# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit the exact Git index before publishing; never print matched secrets."""
from __future__ import annotations

from pathlib import PurePosixPath
import re
import subprocess


FORBIDDEN_DIRS = {'.venv', '__pycache__', '.pytest_cache', 'target', 'runs',
                  'replay', 'checkpoints', 'node_modules', '.idea', '.vscode'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.pyc', '.dylib', '.so', '.dll',
                      '.whl', '.zip', '.log', '.pem', '.key', '.p12', '.pfx',
                      '.safetensors', '.onnx', '.npy', '.npz', '.sqlite', '.db'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|'
                    rb'AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
PERSONAL_PATH = re.compile(rb'(?:/' + rb'Users/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\)')


def problems(name: str, data: bytes):
    path = PurePosixPath(name)
    reasons = []
    if (set(path.parts) & FORBIDDEN_DIRS or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or name.startswith('docs/validation/') or path.name.startswith('.env')):
        reasons.append('private/generated artifact path')
    if len(data) > 500_000 or b'\0' in data:
        reasons.append('large or binary content requires explicit review')
    if SECRET.search(data):
        reasons.append('credential-like content')
    if PERSONAL_PATH.search(data):
        reasons.append('personal machine path')
    return reasons


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    failed = []
    for name in filter(None, names):
        data = subprocess.check_output(['git', 'show', ':' + name])
        reasons = problems(name, data)
        if reasons:
            failed.append((name, reasons))
    for name, reasons in failed:
        print(f'{name}: {", ".join(reasons)}')
    if failed:
        raise SystemExit(1)
    print(f'Publication index audit passed: {len(list(filter(None, names)))} text files; '
          'no blocked artifacts, credential patterns or personal paths.')


if __name__ == '__main__':
    main()
