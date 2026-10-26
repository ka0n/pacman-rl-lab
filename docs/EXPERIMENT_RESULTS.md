# Recorded experiments through v0.9.4

These are numbers supplied from the user's own Windows CUDA/Rust runs; no neural checkpoints are bundled.

## Matched v0.9.4 scratch comparison

All runs trained from random initialization with death penalty -20, with matching seeds 930042, 930043, and 930044. Final held-out evaluation: 1,000 episodes per run, test-seed 620000.

| Train seed | CNN win % | Motion CNN win % | Difference (motion - CNN) pp |
|---|---:|---:|---:|
| 930042 | 89.3 | 75.4 | -13.9 |
| 930043 | 90.1 | 76.2 | -13.9 |
| 930044 | 89.2 | 77.1 | -12.1 |

Means: CNN 89.53% wins, Motion CNN 76.23%; difference -13.30 percentage points. Mean pellets: CNN 97.1969%, Motion 93.5177%.

## Prior fine-tuning / safety observations (single training run per configuration)

On 1,000 identical held-out episodes (test-seed 150000): baseline fine-tune CNN 93.3% wins; safety fine-tune death penalty -20 93.5%; safety fine-tune plus risk-shaping 93.0%; scratch CNN death penalty -20 91.9%. Different testing sets gave slightly different percentages. Distinguish these from v0.9.4.

Risks before failure: limited escape paths and two nearby ghosts were common in the final five actions of losing episodes; correlation alone does not identify the earlier causal decision.

## Limitations

No original model weights, local `runs/` directory, or compiled Rust binaries here. To reproduce inference exactly, export selected `best.pt` files from the Windows PC later. Avoid using held-out seed 620000 as a hyperparameter tuning set.
