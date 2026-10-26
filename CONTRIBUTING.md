# Contributing

Use a short-lived feature branch and open a pull request against `main`.
Keep changes focused and explain the problem, resulting behavior, and tests.
Contributions to original project code are under GPL-3.0-or-later; copyright
stays with the contributor. Do not submit code or assets you cannot license.

Run tests before editing and after the change:

```sh
./scripts/lab.sh test
cargo test --release --locked --manifest-path native/pacman_native/Cargo.toml
./scripts/lab.sh verify
```

Add focused regression tests for behavioral fixes. Preserve Python/Rust parity.
Do not change rules, observations, rewards, maps, terminal handling, or evaluation
seeds as an incidental refactor. Document deliberate experiment changes.

Do not start large training runs unless explicitly requested. Do not commit
models, replay, private configuration, local logs, generated reports, or secrets.
Do not overwrite existing experiment output. Use new run directories.

Checkpoint-selection validation is not a final test: model-quality claims need
multiple training seeds and at least 1,000 common held-out episodes per model,
reporting both win rate and pellet percentage. Do not tune on the historical
620000–620999 final test window. Unknown-maze generalization remains unresolved.

CI runs native Apple Silicon macOS tests, builds Rust, checks engine parity,
and runs bounded CPU smoke tests. MPS tests run when the host exposes Metal;
hosted VM limitations must not be misreported as MPS coverage.
