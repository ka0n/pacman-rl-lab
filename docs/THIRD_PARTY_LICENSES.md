# Third-party license review

## Scope and release decision

Reviewed for the initial **source-only** public release under GPL-3.0-or-later.
Original project code is covered by that declaration; dependencies and upstream
portions keep their own terms. No third-party wheels, shared libraries, model
weights, ROMs, fonts, images or sounds are redistributed in this repository.

The review used the actual macOS arm64 Python 3.13 installation, its installed
license files and metadata, upstream Pygame 2.6.1 notices, Cargo.toml/Cargo.lock,
and the first-public-commit candidate tree. A reproducible inventory records
versions, package-relative license paths, SHA-256 evidence hashes and native-file
names in [dependency_inventory.json](dependency_inventory.json). Recreate it with
`.venv/bin/python scripts/license_inventory.py`; the pinned installation is
[requirements/macos-py313.txt](../requirements/macos-py313.txt).

**No identified license conflict blocks this source-only release.** This is not
an assertion that every future binary build or optional backend is covered:
redistributors of wheels, app bundles or linked executables must review their
exact artifacts and satisfy the obligations below. In particular, a Python
package's headline license is not a complete inventory of its native wheel.

## Resolved Python packages

These are the installed runtime, transitive, test and build distributions, not a
list of every optional extra advertised by their upstream projects. Optional
Atari/ROM, Box2D, MuJoCo, JAX, cloud-filesystem, CUDA/Linux and other extras were
not enabled. License evidence comes from the exact distribution files referenced
and hashed in the inventory, rather than inferred solely from package names.

| Distribution | Version | Verified license / evidence |
|---|---|---|
| torch | 2.14.1 | Main LICENSE: BSD-3-Clause; wheel declares Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT; individual bundled notices retained upstream |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; distribution also carries conditional BLAS/GCC-runtime notices |
| gymnasium | 1.4.0 | MIT |
| pygame | 2.6.1 | LGPL-2.1, verified in packaged `pygame/docs/generated/LGPL.txt` and the tagged upstream README |
| pytest | 9.1.1 | MIT |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| Farama-Notifications | 0.0.6 | MIT, verified in LICENSE although metadata does not declare it |
| filelock | 4.2.4 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| iniconfig | 2.3.1 | MIT |
| Jinja2 | 3.1.6 | BSD-3-Clause, verified in LICENSE.txt despite missing simple metadata field |
| MarkupSafe | 3.0.4 | BSD-3-Clause |
| mpmath | 1.3.0 | BSD-3-Clause text in distribution LICENSE |
| networkx | 3.7 | BSD-3-Clause |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause; both option texts packaged |
| pluggy | 1.7.0 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause; upstream AUTHORS retained in its wheel |
| setuptools | 84.2.1 | MIT; vendored subpackages have their own notices (see below) |
| sympy | 1.14.0 | BSD-3-Clause; additional parser notice packaged |
| typing_extensions | 4.16.0 | PSF-2.0 |
| wheel | 0.46.3 | MIT |

The permissive MIT/BSD/0BSD/Zlib/BSL/PSF/CC0 terms and Apache-2.0 combination do
not impose a conflicting restriction on this GPLv3 source release. Copyright,
license and disclaimer retention still matters. Apache NOTICE and patent terms
must be preserved where applicable; LLVM's exception is an additional permission,
not a requirement to remove notices. The FSF documents Apache-2.0 compatibility
with GPLv3, not GPLv2-only: [license list](https://www.gnu.org/licenses/license-list.html),
[ASF explanation](https://www.apache.org/licenses/GPL-compatibility.html).

## Native and vendored components

### PyTorch

The arm64 wheel carries native `torch/lib` libraries, including LLVM OpenMP,
and over a hundred license-evidence files covering bundled third-party source.
Examples include FP16, FXdiv, NNPACK, XNNPACK, cpuinfo, pthreadpool, Gloo, pybind11,
SLEEF, fmt, protobuf, ONNX and LLVM OpenMP. Its metadata's combined SPDX expression
is reproduced above; the main BSD text alone would be an incomplete account.
Notices for source/backend components do not establish that every such backend
is linked into the macOS wheel. The installed wheel's file inventory is recorded.
CUDA/NVIDIA Linux packages were not resolved on this platform and are not part of
this review. No PyTorch binary or vendored PyTorch source is copied into this repo.

### NumPy

The inspected `_multiarray_umath` native extension links to macOS **Accelerate**,
libc++ and libSystem. No `numpy/.dylibs` directory was present in this installed
wheel. Its packaged LICENSE nevertheless includes template notices for OpenBLAS,
LAPACK and GCC runtime libraries; the text explicitly warns that listing a
license does not prove it applies to every binary. Do not misreport OpenBLAS or
libgfortran as locally bundled based on that template alone.

For wheels that do bundle them: OpenBLAS/LAPACK use permissive BSD terms and GCC
runtimes use GPL-3.0-or-later WITH GCC-exception-3.1. Their notices and the runtime
exception must travel with redistributed binaries. NumPy's source-component
licenses include pocketfft, libdivide, Highway, random-number implementations,
pythoncapi compatibility and other embedded notices recorded in the inventory.
Apple system frameworks are not redistributed by this project.

### Pygame and its macOS wheel

Pygame's top-level LGPL declaration does not cover all its native dependencies.
The inspected wheel contains 28 `.dylib` files, including SDL2, SDL2_image,
SDL2_mixer, SDL2_ttf, FreeType, libjpeg, libpng, libtiff, WebP/sharpyuv, zlib-ng,
Brotli, Ogg/Vorbis, Opus/opusfile, FLAC, mpg123, libsndfile, libmodplug,
FluidSynth, GLib/GThread, libintl and PortMidi. Names are recorded in the inventory;
filename versions are not treated as a complete upstream version identification.

Relevant license families: SDL2 components use zlib terms; FreeType offers its
FTL/GPL options; common image/audio codecs use BSD/MIT/zlib-style notices;
mpg123, FluidSynth, libsndfile, GLib/GThread and libintl involve LGPL terms.
The tagged upstream [dependency-license directory](https://github.com/pygame/pygame/tree/2.6.1/docs/licenses)
contains the component notices; the review specifically checked FreeType,
FluidSynth, mpg123 and SDL2 files, as well as the packaged LGPL text. FreeType's
original attribution must be retained when its applicable option requires it.

Pygame is compatible with use from this GPLv3 project; it remains independently
LGPL-licensed. If redistributing its libraries or a frozen app: include the
applicable LGPL texts/notices, provide the corresponding library source or a
compliant source offer as the exact license requires, identify modifications,
and preserve users' ability to replace/relink LGPL components and debug such
modifications. Do not impose contrary reverse-engineering restrictions.
Dynamic loading alone is not a blanket exemption from these obligations.
See [LGPL 2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html) and the
[GNU compatibility table](https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility).

Pygame's default font and system-font fallback are runtime resources supplied by
Pygame/the OS. They are not copied into the repository. A future bundled app must
also examine the actual font licenses; do not assume the project's GPL declaration
licenses operating-system fonts or all Pygame resources.

### Build and installation tools

Setuptools vendors autocommand, backports.tarfile, importlib_metadata, jaraco
packages, more-itertools, packaging, platformdirs, tomli, wheel and zipp. Their
MIT/BSD/Apache notices are present in its installed distribution and recorded in
the inventory. Python's PSF and incorporated-software notices remain upstream.
Pip 26.2 is an installer, not an application dependency: it includes separately
licensed vendored software, notably certifi (MPL-2.0), alongside MIT/BSD/Apache
components. Pip itself, the Python interpreter, Rust/LLVM, Homebrew and Apple SDKs
are not redistributed. Publishing this project's source does not relicense those
tools. Shipping a prebuilt interpreter/toolchain or application would require a
separate artifact-specific notice/source review.

## Rust and incorporated algorithm notices

`Cargo.lock` and `cargo metadata --locked --no-deps` contain only this project's
`pacman_native` crate: **zero external crates, no PyO3 or Rust binding package**.
The bridge uses Python's `ctypes` and the Rust standard library. Rust-generated
binary/runtime redistribution needs its own corresponding-source and toolchain
notice assessment; no dylib is part of this publication.

The in-tree `PythonMt` implements standard MT19937 seeding/twisting/tempering and
CPython-compatible integer seed and float behavior. Its structure follows the
published algorithm; retain the upstream copyright/conditions conservatively
rather than claim exclusive project authorship. The original Matsumoto/Nishimura
BSD notice is preserved in [MT19937-LICENSE.txt](../third_party/MT19937-LICENSE.txt),
with the [Python license](../third_party/Python-LICENSE.txt) for CPython compatibility
portions. Reference inspected: [CPython 3.13 random implementation](https://github.com/python/cpython/blob/v3.13.0/Modules/_randommodule.c).
These permissive portions can be included in the GPLv3 project while keeping their
notices; project licensing does not erase their rights or invent new holders.

## Assets, provenance and trademark scope

The source tree contains programmatically drawn primitive shapes and ASCII
research-grid layouts, not commercial sprites, audio, ROMs or a commercial game
logo. There are no vendored asset packs or external neural checkpoints. The
starting snapshot had no prior Git history to audit; original project-code
licensing is applied at the project owner's explicit direction. Historical
experiment summaries remain unchanged, and raw local reports are excluded.

The name describes an independent Pac-Man-inspired research project. NOTICE and
the README disclaim affiliation/endorsement by Bandai Namco and do not grant
trademark rights. No concrete unlicensed commercial asset was identified in the
publication tree. This factual source review does not claim trademark clearance
for commercial branding, merchandise or future logos.

## Distribution obligations and limits

For this release, recipients get the project's full source and build/test scripts,
GPLv3 LICENSE, SPDX declarations and the incorporated-algorithm notices. Dependency
installation downloads upstream packages; none are repackaged here. Do not remove
their notices or state that all dependencies are GPLv3.

Before any future binary/app release, inventory the exact linked files and their
versions, preserve all notices/attributions, supply corresponding source and
installation/relinking information where required, and review fonts/codecs/SDK
terms. The present review does **not** approve arbitrary optional extras, other
platform wheels, an app-store package or static linking. No unresolved license
conflict was found in the source-only publication scope.
