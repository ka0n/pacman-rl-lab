# Issue #8 audit fixes

Scope: the six independent Review-MCP findings against
`143e4701dd585a5d3e27daeab2ca377e84d31374`. All six were reproduced before
production changes. The original baseline had 99 passing Python tests and five
MPS skips inside the restricted process. The initial focused reproductions had
35 failures and 10 passes, including control cases.

## Confirmed defects and regressions

| Finding | Reproduction and correction |
| --- | --- |
| Ghost-count reset increase | A small maze constructed with zero or one ghost could not reset to two consistently. Both engines now prepare the same two-spawn pool independently of the active count. Tests cover 0→2, 1→2 and 2→0, single/multiple-G maps, Gymnasium options and subsequent movement. |
| Invalid batch actions | Narrowing 256 or -256 to uint8 produced a legal action; Python could advance earlier lanes before rejecting a later one. Both batch APIs validate the original shape, integer dtype and 0..3 range before any mutation. Float, NaN, infinity, bool, string, signed and unsigned overflow, malformed shape and finished-lane atomicity are covered. |
| Reset metadata | Native public attributes stayed stale after reset, and batch attributes could change before a failing reset. Metadata now commits only after success. Tests cover validation failures, an injected native error and a later reset without overrides. |
| GRU replay starvation | A six-transition episode in a capacity-five replay was immediately evicted; both trainers saved an empty optimizer state. Replay now keeps bounded contiguous tails and makes full live tails sampleable. Tests check memory limits, episode/world isolation, contiguous windows and actual GRU weight updates. |
| Interrupted best save | Crashing after the first final-evaluation save left best.pt absent or stale in each of the three trainers. Best is now saved before last. Resume reconciles an already committed best score, and the CNN suite recovers a lone best checkpoint. Fault injection covers first and improved final evaluations, warm restart with a lower subsequent score and completed-run restart. |
| Legacy timeout target | At max_steps=1 the v0.3 trainer added gamma times a nonzero next-Q value. Replay now masks terminated or truncated. A deterministic wall-timeout regression observes the actual Bellman target: -0.11 rather than -0.11 + 0.99 × 8. |

## Replay policy and reproducibility

Completed episodes retain the existing episode-uniform, then window-uniform
sampling policy. Each live episode retains at most its transition budget; old
prefixes are evicted without joining episodes. A live tail becomes sampleable
when its budget is full. Every sampled window still contains the entire
`burnin + unroll`; loss masking and recurrent forward/backward semantics are
unchanged. Vector replay divides capacity across worlds, distributing remainder
slots in world order, and shares the remaining storage with completed episodes.
Capacity must fit at least one full sequence per world; recurrent warmup must
fit within capacity. These previously infeasible configurations now fail early.

The checkpoint dictionary format is unchanged. Warm restart still resets replay,
RNG and episode state. Saving best before last protects against process failure
between the two atomic file replacements; it is not a two-file transaction.
Both suites reconcile best checkpoints before marking runs complete, including
old last-before-best interruptions. Recovery from last requires its saved
evaluation to match the recorded best score; otherwise unavailable best weights
produce an explicit recovery error. An external warm restart copies the best
model from the source checkpoint's directory into the new output directory.

Sourcery's follow-up edge cases also have focused regressions: effective world
count is clamped before recurrent capacity validation; profiling uses the same
live-tail readiness as training; and Python batch cases run without a native
build. Cadence overrides, like ghost-count overrides, require a single-game
native batch so settings cannot leak between lanes.

No Motion-CNN architecture, game rules, rewards, observations, evaluation seeds
or historical experiment artifacts were changed. The legacy episodic timeout
mask and defective replay/reset behavior above are the explicit corrections.
No learning-quality conclusion follows from these fixes or smoke tests.

## Local validation

- Full Python suite with Metal access: **180 passed, no skips**.
- Rust debug and release suites: **3 passed each**; release library rebuilt.
- Python/Rust parity: all 7 maps × 3 ghost counts × 2 cadences × 30 episodes,
  **501,488 identical steps**, including observations, rewards, endings and stats.
- Apple Silicon MPS and CPU: all five architectures passed initial-Q parity,
  finite-gradient checks and one real Adam update each.
- CPU/Python, CPU/Rust, MPS/Python and MPS/Rust: CNN and Motion-CNN each passed
  32-transition smoke runs with three optimizer updates and checkpoint roundtrip.
- All generated smoke checkpoints and reports use new temporary directories.
  No existing training data or checkpoints were used or modified.
