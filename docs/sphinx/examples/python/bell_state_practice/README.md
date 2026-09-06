# Bell-state practice: see a relative phase change

Compare two Bell states using CUDA-Q statevectors, measurement counts, and
expectation values. The central observation is that a relative phase change
can be invisible in the Z-basis histogram but visible in another observable.

This is personal learning material in a CUDA-Q fork. The validation records
come from assistant-executed checks; the learner exercises remain open.

## Setup and run

Use Python 3.12 in Linux or WSL2. From the repository root:

    cd examples/python/bell_state_practice
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python bell_phase.py
    python bell_phase.py --phase-flip

You can also activate an existing environment with these dependencies.
The default target is the NVIDIA GPU simulator and requires a compatible
NVIDIA GPU and driver. To use the CPU simulator:

    python bell_phase.py --target qpp-cpu
    python bell_phase.py --target qpp-cpu --phase-flip

These commands use an installed CUDA-Q release. They do not build this
repository or validate its unbuilt source revision.

## Where the phase setting is declared

In [bell_phase.py](bell_phase.py), the argument is defined as:

    parser.add_argument("--phase-flip", action="store_true")

Without the flag, args.phase_flip is False. Adding --phase-flip makes it True.
The selected value is printed before the circuit and passed into each
CUDA-Q call. Inside the kernel, True adds Z on q0 after creating the Bell pair.

## Start with the circuit

Kets below are written as |q0 q1>.

$$
|00\rangle
\xrightarrow{H_0}
\frac{|00\rangle+|10\rangle}{\sqrt{2}}
\xrightarrow{CX_{0\to1}}
\frac{|00\rangle+|11\rangle}{\sqrt{2}}
=|\Phi^+\rangle.
$$

The optional Z gate obeys:

$$
Z=\begin{bmatrix}1&0\\0&-1\end{bmatrix},
\qquad Z|0\rangle=|0\rangle,
\qquad Z|1\rangle=-|1\rangle.
$$

Applied to q0, it leaves |00> unchanged and negates |11>:

$$
|\Phi^+\rangle
\xrightarrow{Z_0}
\frac{|00\rangle-|11\rangle}{\sqrt{2}}
=|\Phi^-\rangle.
$$

This is a relative phase change. Multiplying the entire state by -1 would
instead be an unobservable global phase. Both Bell states have the same
Z-basis probabilities because probabilities are squared amplitude magnitudes.

## What the APIs show

| Call | Result | Meaning in this exercise |
| --- | --- | --- |
| get_state | Complex amplitudes on a compatible simulator | Exposes the relative phase |
| sample | Counts of measurement bitstrings | Samples the Z-basis probabilities |
| observe | Expectation of the supplied operator | Reveals correlations in a chosen basis |

Each API executes the preparation kernel for its own result. The kernel
leaves out mz so get_state and observe receive coherent state preparation.
For this circuit, sample supplies final Z-basis measurements automatically.

CUDA-Q's two-qubit statevector index order is |00>, |10>, |01>, |11> with
these |q0 q1> labels: q0 is the least-significant index bit. Both middle
amplitudes vanish for these Bell states.

## What the expectation values mean

The expression often shortened to an expectation of Z0 Z1 means:

$$
\langle Z_0Z_1\rangle_{\Phi^+}
=\langle\Phi^+|Z_0Z_1|\Phi^+\rangle.
$$

It is a dimensionless correlation, not an energy. For each qubit's Z
measurement, assign +1 to outcome 0 and -1 to outcome 1, then multiply the
two values. Equal outcomes contribute +1; opposite outcomes contribute -1.
The expectation is the average product. X0 X1 measures the corresponding
correlation in the X basis.

Z0 Z1 gives +1 on both |00> and |11>. X0 X1 exchanges |00> and |11>, so
Phi+ is its +1 eigenstate and Phi- is its -1 eigenstate.

| State | Z-basis probabilities | X0 X1 expectation | Z0 Z1 expectation |
| --- | --- | ---: | ---: |
| Phi+ | P(00)=P(11)=1/2 | +1 | +1 |
| Phi- | P(00)=P(11)=1/2 | -1 | +1 |

Thus the Z correlation confirms matching outcomes, while the X correlation
distinguishes these two relative phases.

## Sampling and recorded checks

Try a smaller sample:

    python bell_phase.py --shots 100

Counts vary between runs. The --shots option controls sample only.
The observe calls request no finite shot budget and return simulator
expectations up to numerical rounding for these observables.

See [VALIDATION.md](VALIDATION.md) for actual GPU and CPU runs with both phase
settings, including commands, output, versions, and source hashes.

## Why this matters

In superdense coding, the phase choice can encode information that a later
decoder converts into a definite measured bit. An unwanted phase flip can
corrupt the decoded message while leaving the initial Z histogram unchanged.

[The application notes](APPLICATION.md) connect this circuit to the protocol,
include a proposed decoding exercise, and link to an optical-fiber research
demonstration. Decoder execution is a separate, pending exercise.

## Try it yourself

- [ ] Predict the baseline statevector, counts, and correlations, then run it.
- [ ] Add --phase-flip and identify precisely what changes.
- [ ] Reduce the shot count and explain the sampling variation.
- [ ] Explain why Z counts alone cannot identify this relative phase.
- [ ] Explain the X0 X1 expectation as a correlation.
- [ ] Try the optional decoder in the application notes.

Record your own commands, observations, and explanation. The saved assistant
checks do not mark these exercises complete.

Next: [VQE practice](../vqe_practice/README.md), where the observable is a
Hamiltonian and its expectation is an energy.

## Reference

[NVIDIA: Executing Kernels][executing-kernels]

[executing-kernels]: https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html
