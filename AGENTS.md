# Pac-Man RL Lab development guidance

Target macOS Apple Silicon, PyTorch MPS and the native Rust simulator. Inspect the
existing code and run `.venv/bin/python -m pytest -q` before changes and again after
changes. Add focused regression tests for behavioral changes. Build/test Rust and
check Python/Rust parity when simulator, observation or binding code changes.

Preserve reproducibility: do not silently change game rules, rewards, observations,
episode endings, maze layouts, training algorithms or evaluation seeds. Do not
start long training runs without an explicit request. Small bounded smoke tests
and benchmarks are appropriate. Distinguish CPU/Python, CPU/Rust, MPS/Python and
MPS/Rust performance; short profiling runs do not establish learning quality.

Current research question: the controlled three-seed v0.9.4 study found 89.53%
wins for the single-frame CNN and 76.23% for Motion-CNN. Alignment/reset/replay/
Double-DQN bugs have not been demonstrated; the cause remains open. See
`docs/MACOS_MOTION_AUDIT.md` and `docs/EXPERIMENT_RESULTS.md`. Propose controlled
experiments before adding complexity. Quality claims require multiple training
seeds and at least 1,000 common held-out episode seeds per model, reporting both
win rate and pellet percentage. Do not repeatedly tune on a final test set.

Original model checkpoints and run directories are not included. Never fabricate
results or assume access to weights. Do not overwrite or delete existing models,
training data, experiment results or reports. Use new output directories. Never
commit weights, replay, logs, private configuration, credentials or machine paths.

For future changes, prefer feature branches and pull requests against `main`.
After the requested change passes relevant checks, push the branch and open a PR.
Do not merge without explicit authorization. Keep chat updates concise: report
commit/PR, CI results and real blockers; do not paste full diffs or large changesets.

Original code is GPL-3.0-or-later. Preserve third-party notices and do not claim
rights to imported assets or dependencies. No original commercial game assets or
logos may be added. See NOTICE and docs/THIRD_PARTY_LICENSES.md.
