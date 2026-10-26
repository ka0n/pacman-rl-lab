# SPDX-License-Identifier: GPL-3.0-or-later
"""Record installed distribution license evidence without machine paths or secrets."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform


def inventory():
    packages = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name = dist.metadata['Name']
        if name.lower() in ('pip', 'pacman-rl-lab'):
            continue
        licenses = []
        native = []
        for entry in dist.files or []:
            path = str(entry)
            if Path(path).name.lower().startswith(('license', 'copying', 'notice')):
                file = dist.locate_file(entry)
                if file.is_file():
                    licenses.append({'file': path, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
            if path.endswith(('.dylib', '.so')):
                native.append(path)
        if name.lower() == 'pygame':
            path = 'pygame/docs/generated/LGPL.txt'
            licenses.append({'file': path, 'sha256': hashlib.sha256(dist.locate_file(path).read_bytes()).hexdigest()})
        packages.append({'name': name, 'version': dist.version,
                         'declared_license': dist.metadata.get('License-Expression') or dist.metadata.get('License') or None,
                         'license_files': licenses, 'bundled_native_files': native})
    return {'scope': 'Installed macOS arm64 Python 3.13 runtime, test and build distributions; no optional extras beyond dev/ml. Pip and OS/toolchain are discussed separately in THIRD_PARTY_LICENSES.md.',
            'python': platform.python_version(), 'distributions': packages,
            'rust': {'external_crates': [], 'binding': 'ctypes C ABI; no PyO3'}}


if __name__ == '__main__':
    print(json.dumps(inventory(), indent=2))
