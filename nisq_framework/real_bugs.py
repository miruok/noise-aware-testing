# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Real bugs from Bugs4Q (Zhao et al.; https://github.com/Z-928/Bugs4Q), transcribed
to Qiskit 2.

Inclusion criteria: the buggy and the fixed program build a quantum circuit
whose difference is a change of the circuit itself (gates, operands,
measurement mapping or initial state) and that is executable as a static
circuit ending in computational-basis measurement. Excluded: API, version,
visualization and simulator-configuration bugs, programs whose fix replaces
the whole program, and programs that need mid-circuit measurement, reset or
classical control. Each entry records its Bugs4Q path and the original source.

Transcription rules: cu1 -> cp; Statevector initialization of a fresh qubit to
|0> (a no-op) is dropped and initialization to |1> becomes X; programs that do
not measure are completed with measure_all(); otherwise the original
measurement is kept. In the differential setting the fixed program is the
trusted reference P and the buggy program the program under test P'.
"""
from __future__ import annotations
from math import pi
from typing import Callable, Dict, List
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def _aer_664():
    def build(mapping):
        qc = QuantumCircuit(QuantumRegister(3, "q"), ClassicalRegister(3, "c"))
        qc.x(0); qc.measure([1, 0, 2], mapping)
        return qc
    return build([1, 0, 2]), build([0, 1, 2])


def _so_610_1():
    def build(ctrl_first):
        qc = QuantumCircuit(3, 3)
        qc.x(0); qc.x(1)
        if ctrl_first:
            qc.ccx(0, 1, 2)
        else:
            qc.ccx(2, 1, 0)
        qc.barrier(); qc.measure([0, 1, 2], [0, 1, 2])
        return qc
    return build(True), build(False)


def _so_610_2():
    def build(fixed):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1) if fixed else qc.h(1)
        qc.cx(0, 1); qc.measure_all()
        return qc
    return build(False), build(True)


def _iqft_block(n):
    c = QuantumCircuit(n)
    for i in range(n // 2):
        c.swap(i, n - 1 - i)
    for i in range(n):
        c.h(i)
        for j in range(i + 1, n):
            c.cp(-pi / 2 ** (j - i), j, i)
    return c


def _so_610_3():
    n = 3
    def build(fixed):
        q = QuantumRegister(n, "q"); cr = ClassicalRegister(n, "c")
        qc = QuantumCircuit(q, cr)
        qc.h(q); qc.z(2); qc.s(1); qc.z(0); qc.t(0)
        blk = _iqft_block(n)
        qc.compose(blk.inverse() if fixed else blk, inplace=True)
        qc.measure(q, cr)
        return qc
    return build(False), build(True)


def _se_9():
    def build(mapping):
        qc = QuantumCircuit(3, 3)
        qc.x(0); qc.cx(0, 1); qc.barrier(); qc.measure([0, 1, 2], mapping)
        return qc
    return build([0, 1, 2]), build([2, 1, 0])


def _se_15():
    nanc, theta = 3, 0.78
    def qft_dagger_buggy(c, q, n):
        for i in range(n // 2):
            c.swap(q[i], q[n - i - 1])
        for i in reversed(range(n)):
            c.h(q[i])
            for m in reversed(range(i)):
                c.cp(-2 * pi / 2 ** (i - m + 1), q[i], q[m])
            c.barrier()
    def qft_dagger_fixed(c, q, n):
        for i in range(n - 1, -1, -1):
            for m in range(n - i, 1, -1):
                c.cp(-2 * pi / 2 ** m, q[i + m - 1], q[i])
            c.h(q[i]); c.barrier()
    def build(fixed):
        q = QuantumRegister(nanc, "q"); s = QuantumRegister(1, "s"); cr = ClassicalRegister(nanc, "c")
        qc = QuantumCircuit(q, s, cr)
        qc.x(s[0])
        for i in range(nanc):
            qc.h(q[i])
        for i in range(nanc):
            qc.cp(2 * pi * theta * 2 ** (nanc - i - 1), q[i], s[0])
        (qft_dagger_fixed if fixed else qft_dagger_buggy)(qc, q, nanc)
        for i in range(nanc):
            qc.measure(q[i], cr[i])
        return qc
    return build(False), build(True)


def _terra_5580():
    def build(fixed):
        qc = QuantumCircuit(3)
        if fixed:
            qc.crz(pi / 2, 0, 1); qc.crz(pi / 4, 0, 2)
        else:
            qc.crz(pi / 2, 2, 0); qc.crz(pi / 4, 2, 1)
        qc.measure_all()
        return qc
    return build(False), build(True)


def _terra_5098():
    def build(fixed):
        qc = QuantumCircuit(3)
        qc.cx(0, 1, ctrl_state="0" if fixed else 0)
        qc.ccx(0, 1, 2, ctrl_state="00" if fixed else 1)
        qc.measure_all()
        return qc
    return build(False), build(True)


def _sep_9():
    def build(fixed):
        qc = QuantumCircuit(4, 4)
        if fixed:
            for i in range(4):
                qc.h(i)
        qc.cx(3, 1); qc.cx(1, 0); qc.cx(0, 1); qc.ccx(3, 2, 1); qc.cx(1, 2); qc.cx(3, 2)
        qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
        return qc
    return build(False), build(True)


def _sep_19():
    def build(fixed):
        qc = QuantumCircuit(2)
        if fixed:
            qc.x(1)            # initialize(1, 1): qubit 1 in |1>
        qc.h(0); qc.h(1); qc.cx(0, 1); qc.h(0); qc.h(1)
        qc.measure_all()
        return qc
    return build(False), build(True)


REAL_BUGS: Dict[str, Dict] = {
    "aer_664": {"build": _aer_664, "path": "Aer/bug_7", "source": "qiskit-aer issue #664", "kind": "measurement mapping"},
    "so_610_1": {"build": _so_610_1, "path": "stackoverflow-6-10/bug_1", "source": "Stack Overflow 63283443", "kind": "qubit order"},
    "so_610_2": {"build": _so_610_2, "path": "stackoverflow-6-10/bug_2", "source": "Stack Overflow 62661255", "kind": "wrong gate"},
    "so_610_3": {"build": _so_610_3, "path": "stackoverflow-6-10/bug_3", "source": "Stack Overflow (inverse QFT)", "kind": "wrong subroutine"},
    "se_9": {"build": _se_9, "path": "StackExchange/9", "source": "Quantum Computing Stack Exchange", "kind": "measurement mapping"},
    "se_15": {"build": _se_15, "path": "StackExchange/15", "source": "Quantum Computing Stack Exchange", "kind": "wrong subroutine"},
    "terra_5580": {"build": _terra_5580, "path": "Terra-4001-6000/Bug_5", "source": "qiskit-terra issue #5580", "kind": "qubit order"},
    "terra_5098": {"build": _terra_5098, "path": "Terra-0-4000/3", "source": "qiskit-terra issue #5098", "kind": "control state"},
    "sep_9": {"build": _sep_9, "path": "StackExchange-page-1-25/Bug_9", "source": "Quantum Computing Stack Exchange", "kind": "missing gates"},
    "sep_19": {"build": _sep_19, "path": "StackExchange-page-1-25/Bug_19", "source": "Quantum Computing Stack Exchange", "kind": "wrong initial state"},
}


def real_bug_pairs() -> List[Dict]:
    """Return [{'id', 'buggy', 'fixed', ...}] for every included real bug."""
    out = []
    for k, v in REAL_BUGS.items():
        b, f = v["build"]()
        out.append({"id": k, "buggy": b, "fixed": f, **{x: v[x] for x in ("path", "source", "kind")}})
    return out
