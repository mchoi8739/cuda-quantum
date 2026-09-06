# VQE validation records

These checks were executed by the assistant from the copied example in WSL2.
They establish a working baseline; the learner checklist remains open.

## Environment

- OS: Ubuntu 24.04 in WSL2.
- Python: 3.12.3.
- Installed packages: cudaq 0.15.1, numpy 2.5.2, scipy 1.18.1, matplotlib
  3.11.1.
- First recorded run (UTC): 2026-09-06T00:59:05.395044+00:00.

The runs use the installed CUDA-Q wheel, not a build of this source checkout.

## Executed commands

With the existing Python environment activated, from this example directory:

    python vqe.py
    python vqe.py --target qpp-cpu
    python vqe.py --max-evals 3

| Run | Evaluations | Final energy (MeV) | Validation passed | Exit status |
| --- | ---: | ---: | --- | --- |
| [GPU](recorded_runs/gpu/result.json) | 22 | -1.7488648556 | true | 0 |
| [CPU](recorded_runs/cpu/result.json) | 23 | -1.7488649117 | true | 0 |
| [GPU, budget 3](recorded_runs/budget_limit/result.json) | 3 | -1.7148662201 | false | 2 (expected) |

Each run starts at 0 radians, with energy -0.4362900000 MeV. The GPU and CPU
runs use a budget of 100 evaluations. The deliberately limited run reports
budget exhaustion rather than claiming convergence.

The GPU run's optimized angle is 0.59420355 radians.
The independent model ground energy is -1.7488649142 MeV.
The GPU energy error is 5.856e-08 MeV, and its ground-state
fidelity is 0.9999999986.

## Evidence

All three records include the full optimizer trace, the separate 65-angle
sweep, and a result.json file. The GPU record also includes the PNG and SVG
plots. See [recorded_runs](recorded_runs/).

The calling process checked the expected exit status for each command.
CSV row counts match the evaluation counts, and the recorded SHA256 hashes
match energy.py and vqe.py as published. Optimizer success, energy agreement,
state fidelity, and the independent sweep checks are recorded in each JSON.

New runs save timestamped directories under results. Those directories are
ignored by Git; recorded_runs contains selected copies for review.
Counts and final digits may vary with simulator precision and dependencies.

Saved CSV line endings were normalized to LF; all values were preserved.
