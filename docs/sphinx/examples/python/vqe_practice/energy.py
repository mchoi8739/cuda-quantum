# This source code and the accompanying materials are made available under
# the Apache License 2.0. See LICENSE at the repository root.
"""Exercise 2, step 1: evaluate the energy of a two-qubit trial state."""
from importlib.metadata import version

import cudaq
from cudaq import spin

# Change this value from 0.0 to 0.59 for the first comparison (radians).
ANGLE = 0.0
TARGET = "nvidia"


@cudaq.kernel
def ansatz(theta: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(theta, q[1])
    x.ctrl(q[1], q[0])


# Fixed two-qubit deuteron model from the NVIDIA example; coefficients in MeV.
HAMILTONIAN = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
               2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
               6.125 * spin.z(1))


def energy(theta: float) -> float:
    """Return <psi(theta)|H|psi(theta)> for the currently selected target."""
    return cudaq.observe(ansatz, HAMILTONIAN, theta).expectation()


if __name__ == "__main__":
    cudaq.set_target(TARGET)
    print(f"CUDA-Q {version('cudaq')} | target={cudaq.get_target().name}")
    print(f"Angle: {ANGLE:.6f} radians")
    print(cudaq.draw(ansatz, ANGLE))
    print(f"Energy: {energy(ANGLE):.8f} MeV")
