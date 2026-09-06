# Bell-state validation records

These are assistant-executed checks of the copied exercise in Ubuntu 24.04
under WSL2. Learner exercises and the optional decoder remain pending.

## Environment

- Python: 3.12.3.
- CUDA-Q: 0.15.1.
- NumPy: 2.5.2.
- Targets: nvidia GPU simulator and qpp-cpu simulator.

The installed CUDA-Q wheel was used; this repository was not built.

## Commands and results

Run from this example directory with the Python environment activated:

    python bell_phase.py
    python bell_phase.py --phase-flip
    python bell_phase.py --target qpp-cpu
    python bell_phase.py --target qpp-cpu --phase-flip

All four commands exited with status 0 and used 1,000 sampling shots.

| Run | 00 counts | 11 counts | X0 X1 | Z0 Z1 |
| --- | ---: | ---: | ---: | ---: |
| [gpu_phi_plus](recorded_runs/gpu_phi_plus.txt) | 492 | 508 | 0.99999994 | 0.99999994 |
| [gpu_phi_minus](recorded_runs/gpu_phi_minus.txt) | 482 | 518 | -0.99999994 | 0.99999994 |
| [cpu_phi_plus](recorded_runs/cpu_phi_plus.txt) | 495 | 505 | 1.00000000 | 1.00000000 |
| [cpu_phi_minus](recorded_runs/cpu_phi_minus.txt) | 491 | 509 | -1.00000000 | 1.00000000 |

Both phase settings produced only 00 and 11 samples. Counts fluctuate
between runs; they need not be exactly equal. The X correlation changed
sign with the phase, while the Z correlation remained approximately +1.

## Checks performed

- The printed phase flag matches the command.
- The printed statevector matches the expected Phi+ or Phi- amplitudes.
- Counts sum to 1,000 and contain no other bitstrings.
- Both expectation values agree with theory within 2e-6.
- The copied script has the same Python syntax tree as the original
  playground exercise; packaging changed its header and formatting.

The statevector check uses the amplitudes printed to six decimal places.
The expectation values retain their printed precision.

The [machine-readable record](recorded_runs/results.json) stores the actual
counts, expectations, versions, timestamps, and the published script's SHA256.
Linked transcripts contain the exact commands and captured output, with
trailing whitespace removed.

The separate superdense-coding decoder in [APPLICATION.md](APPLICATION.md)
is proposed practice code and was not included in these runs.
