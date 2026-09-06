# This source code and the accompanying materials are made available under
# the Apache License 2.0. See LICENSE at the repository root.
"""Exercise 1: compare Bell states with and without a relative phase flip."""
import argparse
from importlib.metadata import version

import cudaq
import numpy as np
from cudaq import spin


@cudaq.kernel
def bell(phase_flip: bool):
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])
    if phase_flip:
        z(q[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-flip", action="store_true")
    parser.add_argument("--target",
                        choices=["nvidia", "qpp-cpu"],
                        default="nvidia")
    parser.add_argument("--shots", type=int, default=1000)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")

    cudaq.set_target(args.target)
    print(f"CUDA-Q {version('cudaq')} | target={cudaq.get_target().name}")
    print(f"Phase flip: {args.phase_flip}")
    print(cudaq.draw(bell, args.phase_flip))

    # Each API executes the preparation kernel for its own result.
    state = np.asarray(cudaq.get_state(bell, args.phase_flip))
    print("Statevector:", np.round(state, 6))

    # With no mz in this preparation kernel, sample measures in Z by default.
    counts = cudaq.sample(bell, args.phase_flip, shots_count=args.shots)
    print(f"Z-basis counts ({args.shots} shots):", counts)

    xx = spin.x(0) * spin.x(1)
    zz = spin.z(0) * spin.z(1)
    print("<X0 X1>:", cudaq.observe(bell, xx, args.phase_flip).expectation())
    print("<Z0 Z1>:", cudaq.observe(bell, zz, args.phase_flip).expectation())
