# Application: superdense coding

Documented: September 5, 2026.
Status: application explanation and proposed extension; decoder execution is
pending.

## Why the phase difference matters

The relative sign can carry information. Later gates can turn that sign into a
readable bit, even though immediate Z-basis measurements of Phi+ and Phi- have
identical probabilities. An unintended phase flip could therefore corrupt the
decoded message while leaving the initial 00/11 histogram looking correct.

A concrete application is superdense coding, a quantum communication protocol:

1. Alice and Bob first share a Bell pair, with one qubit each.
2. Alice encodes information by applying gates to her qubit. Our optional Z
gate
   is the operation used to encode the phase bit.
3. Alice sends her qubit to Bob, who then has both qubits.
4. Bob applies a decoder and measures to recover the encoded information.

The full protocol uses both optional Z and optional X operations to select
among
four Bell states. It communicates two classical bits by transmitting one qubit,
using one previously shared Bell pair. Our extension below demonstrates only
the phase-bit part of that protocol.

Source: [IBM Quantum: Superdense coding][ibm-dense].

## Connect the protocol to this exercise

In the current script, argparse's action="store_true" makes the phase flag
False
when --phase-flip is absent and True when it is present. False prepares Phi+;
True prepares Phi-. The flag is the sender's choice of the phase bit.

To decode that choice, append controlled-X from q0 to q1, then H on q0, before
measurement. The first gate separates the two qubits, and H converts the
relative phase into a definite computational-basis value of q0.

Using this document's ket labels |q0 q1>, the ideal transformations are:

$$
|\Phi^+\rangle
\xrightarrow{CX_{0\to1}}
\frac{|00\rangle+|10\rangle}{\sqrt{2}}
\xrightarrow{H_0}
|00\rangle,
$$

$$
|\Phi^-\rangle
\xrightarrow{CX_{0\to1}}
\frac{|00\rangle-|10\rangle}{\sqrt{2}}
\xrightarrow{H_0}
|10\rangle.
$$

| Sender's phase bit | State before decoding | Expected decoded bits q0 q1 |
| --- | --- | --- |
| False / 0 | Phi+ | 00 |
| True / 1 | Phi- | 10 |

Here q0 carries the recovered phase bit and q1 stays 0. IBM's lesson reports
its
two-bit messages in the reverse qubit order (Bob's original qubit, then
Alice's),
so its Phi- example is labeled 01. Always state the bit order when comparing
tables or outputs.

## Optional CUDA-Q extension

Use a separate sampling kernel for decoding so the original preparation kernel
remains available for the statevector and XX/ZZ comparison. The original
expectations describe the state BEFORE decoding.

This snippet is proposed practice code; its expected results below have not
been verified by a run in this exercise.

    import cudaq

    @cudaq.kernel
    def decode_phase_bit(phase_bit: bool):
        q = cudaq.qvector(2)

        # Prepare the shared Bell pair.
        h(q[0])
        x.ctrl(q[0], q[1])

        # Alice encodes the phase bit.
        if phase_bit:
            z(q[0])

        # Bob decodes after receiving Alice's qubit.
        x.ctrl(q[0], q[1])
        h(q[0])
        mz(q)

    cudaq.set_target("nvidia")
    for phase_bit in [False, True]:
        counts = cudaq.sample(decode_phase_bit, phase_bit, shots_count=1000)
        print(f"phase_bit={phase_bit}: {counts}")

Expected ideal counts are 00:1000 for False and 10:1000 for True, using the
q0 q1 measurement-bit order. These are deterministic ideal predictions after
decoding, rather than the roughly 50/50 counts before decoding. Actual hardware
noise could change the observed counts.

## Has this been used outside a simulator?

Williams, Sadlier, and Humble reported an experimental demonstration of
superdense coding over optical fiber, including a hybrid quantum/classical
image-transfer protocol. Their implementation used photonic hardware and
time-polarization hyperentanglement. It is evidence of a research
demonstration;
our proposed CUDA-Q extension simulates the logical circuit.

Source: [Superdense coding over optical fiber links with complete Bell-state
measurements, Physical Review Letters 118, 050501 (2017)][optical-demo].

## What to demonstrate and record

- [ ] Run the decoder for phase bits False and True.
- [ ] Record the exact commands, target, CUDA-Q version, and observed counts.
- [ ] Explain why pre-decoder Z counts look the same but decoded bits differ.
- [ ] Explain the resources: a shared Bell pair and transmission of Alice's qubit.
- [ ] Distinguish this one-bit extension from the full two-bit protocol.

A completed demo should show that the phase choice encodes a bit and that the
decoder recovers that choice. The XX expectation from the original exercise is
a useful phase check; the decoder makes its communication meaning concrete.

[ibm-dense]: https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/entanglement-in-action/superdense-coding
[optical-demo]: https://arxiv.org/abs/1609.00713
