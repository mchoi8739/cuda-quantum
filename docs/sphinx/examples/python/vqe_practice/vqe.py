# This source code and the accompanying materials are made available under
# the Apache License 2.0. See LICENSE at the repository root.
"""Run an iterative CUDA-Q/SciPy VQE and independently check its result."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import cudaq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from energy import ansatz, energy

ENERGY_TOLERANCE = 2e-5  # MeV; allows the default NVIDIA simulator's rounding.


def reference_matrix():
    """Independent matrix, ordered |00>, |10>, |01>, |11> for |q0 q1>."""
    identity = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    return (5.907 * np.eye(4) - 2.1433 * np.kron(x, x) -
            2.1433 * np.kron(y, y) + 0.21829 * np.kron(identity, z) -
            6.125 * np.kron(z, identity))


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-angle",
                        type=float,
                        default=0.0,
                        help="Starting angle in radians (default: 0.0).")
    parser.add_argument("--target",
                        choices=["nvidia", "qpp-cpu"],
                        default="nvidia")
    parser.add_argument(
        "--max-evals",
        type=int,
        default=100,
        help="Maximum optimizer energy evaluations (default: 100).")
    args = parser.parse_args()
    if not np.isfinite(args.initial_angle) or args.max_evals < 3:
        parser.error("Use a finite initial angle and at least 3 evaluations.")

    cudaq.set_target(args.target)
    started = datetime.now(timezone.utc)
    folder = Path(__file__).resolve().parent
    output = folder / "results" / started.strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    trace = []
    print(f"CUDA-Q {version('cudaq')} | target={cudaq.get_target().name}",
          flush=True)
    print(f"COBYLA | initial angle={args.initial_angle:.6f} radians",
          flush=True)
    print(
        "Each row is an optimizer energy evaluation, not necessarily an accepted step.",
        flush=True)

    # This is the VQE feedback loop: SciPy proposes parameters, then CUDA-Q
    # prepares that state and returns its Hamiltonian expectation value.
    def objective(parameters):
        theta = float(parameters[0])
        value = float(energy(theta))
        if not np.isfinite(value):
            raise RuntimeError(
                "The energy evaluation returned a non-finite value.")
        best = min(value, trace[-1][3]) if trace else value
        trace.append((len(trace) + 1, theta, value, best))
        print(
            f"Eval {len(trace):02d} | theta={theta: .8f} | "
            f"E={value: .10f} MeV | best={best: .10f}",
            flush=True)
        return value

    result = minimize(
        objective,
        x0=[args.initial_angle],
        method="COBYLA",
        options={
            "rhobeg": 0.5,
            "tol": 1e-4,
            "maxiter": args.max_evals
        },
    )
    theta = float(result.x[0])
    final_energy = float(result.fun)
    write_csv(output / "optimization_trace.csv",
              ["evaluation", "theta_radians", "energy_mev", "best_so_far_mev"],
              trace)

    # Independent validation happens AFTER optimization. These values are never
    # supplied to the optimizer or used to choose its initial angle.
    matrix = reference_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    ground_energy = float(eigenvalues[0])
    ground_state = eigenvectors[:, 0]
    state = np.asarray(cudaq.get_state(ansatz, theta), dtype=complex)
    fidelity = float(
        abs(np.vdot(ground_state, state))**2 / np.vdot(state, state).real)
    error = abs(final_energy - ground_energy)

    # Separate sweep for the energy-landscape plot; it is not the optimizer.
    print(
        "Optimization finished. Evaluating a separate 65-angle landscape for the plot.",
        flush=True)
    angles = np.linspace(-np.pi, np.pi, 65)
    landscape = []
    for angle in angles:
        value = float(energy(float(angle)))
        trial_state = np.array([0, np.cos(angle / 2), np.sin(angle / 2), 0])
        reference = float(np.vdot(trial_state, matrix @ trial_state).real)
        landscape.append((float(angle), value, reference))
    write_csv(output / "energy_sweep.csv",
              ["theta_radians", "cudaq_energy_mev", "matrix_energy_mev"],
              landscape)
    sweep_error = max(abs(row[1] - row[2]) for row in landscape)
    landscape_varies = bool(np.ptp([row[1] for row in landscape]) > 1.0)
    matches_reference = bool(error < ENERGY_TOLERANCE and fidelity > 1 - 1e-5)
    validation_passed = bool(result.success and matches_reference and
                             sweep_error < ENERGY_TOLERANCE and
                             landscape_varies)

    summary = {
        "started_utc":
            started.isoformat(),
        "target":
            cudaq.get_target().name,
        "versions": {
            name: version(name)
            for name in ["cudaq", "numpy", "scipy", "matplotlib"]
        },
        "optimizer":
            "scipy.optimize.minimize / COBYLA",
        "options": {
            "rhobeg": 0.5,
            "tol": 1e-4,
            "maxiter": args.max_evals
        },
        "initial_angle_radians":
            args.initial_angle,
        "initial_energy_mev":
            trace[0][2],
        "optimized_angle_radians":
            theta,
        "optimized_energy_mev":
            final_energy,
        "optimizer_success":
            bool(result.success),
        "optimizer_message":
            str(result.message),
        "optimizer_evaluations":
            int(result.nfev),
        "exact_ground_energy_mev":
            ground_energy,
        "absolute_energy_error_mev":
            error,
        "ground_state_fidelity":
            fidelity,
        "reference_ground_state_weight_in_ansatz_sector":
            float(np.sum(abs(ground_state[1:3])**2)),
        "landscape_evaluations":
            len(landscape),
        "max_landscape_matrix_error_mev":
            sweep_error,
        "landscape_varies":
            landscape_varies,
        "energy_tolerance_mev":
            ENERGY_TOLERANCE,
        "validation_passed":
            validation_passed,
        "source_sha256": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ["energy.py", "vqe.py"]
        },
    }
    (output / "result.json"
    ).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    evaluations = [row[0] for row in trace]
    axes[0].plot(evaluations, [row[2] for row in trace],
                 ".-",
                 color="#4b77be",
                 label="Evaluated energy",
                 linewidth=1)
    axes[0].plot(evaluations, [row[3] for row in trace],
                 color="#19826d",
                 label="Best so far",
                 linewidth=2)
    axes[0].axhline(ground_energy,
                    color="#333333",
                    linestyle="--",
                    label="Exact ground energy")
    axes[0].set(xlabel="Optimizer energy evaluation",
                ylabel="Energy (MeV)",
                title="VQE search")
    axes[1].plot(angles, [row[1] for row in landscape],
                 color="#4b77be",
                 label="CUDA-Q sweep")
    axes[1].axhline(ground_energy,
                    color="#333333",
                    linestyle="--",
                    label="Exact ground energy")
    wrapped_theta = (theta + np.pi) % (2 * np.pi) - np.pi
    axes[1].scatter([wrapped_theta], [final_energy],
                    color="#d55e00",
                    s=55,
                    zorder=5,
                    label="Optimizer result")
    axes[1].set(xlabel="Angle (radians)",
                ylabel="Energy (MeV)",
                title="Separate energy landscape")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"Two-qubit deuteron model | {args.target} | start = {args.initial_angle:g} rad"
    )
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=170)
    fig.savefig(output / "convergence.svg")
    plt.close(fig)

    print(f"\nOptimizer success: {bool(result.success)}", flush=True)
    print(f"Stop reason: {result.message}", flush=True)
    print(f"Optimizer evaluations: {result.nfev}", flush=True)
    print(f"Optimized angle: {theta:.8f} radians", flush=True)
    print(f"Optimized energy: {final_energy:.10f} MeV", flush=True)
    print(f"Exact ground energy: {ground_energy:.10f} MeV", flush=True)
    print(f"Absolute energy error: {error:.3e} MeV", flush=True)
    print(f"Ground-state fidelity: {fidelity:.10f}", flush=True)
    print(f"Validation: {'PASS' if validation_passed else 'NOT PASSED'}",
          flush=True)
    print(f"Results: {output}", flush=True)
    return 0 if validation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
