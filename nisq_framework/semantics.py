# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Exact ideal semantics of measured circuits.

Used to label every mutant and injected fault as semantically equivalent or
not *before* any noisy execution, so that detection metrics are computed
against exact ground truth.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from .divergence import jsd


def first_measure_index(qc: QuantumCircuit) -> int:
    for i, ins in enumerate(qc.data):
        if ins.operation.name == "measure":
            return i
    return len(qc.data)


def check_terminal_measurements(qc: QuantumCircuit) -> bool:
    """True iff no gate acts on a qubit after that qubit is measured."""
    measured = set()
    for ins in qc.data:
        qs = {qc.find_bit(q).index for q in ins.qubits}
        if ins.operation.name == "measure":
            measured |= qs
        elif ins.operation.name != "barrier" and qs & measured:
            return False
    return True


def unitary_part(qc: QuantumCircuit, keep_barriers: bool = False) -> QuantumCircuit:
    u = QuantumCircuit(qc.num_qubits)
    for ins in qc.data:
        if ins.operation.name == "measure":
            continue
        if ins.operation.name == "barrier" and not keep_barriers:
            continue
        u.append(ins.operation, [qc.find_bit(q).index for q in ins.qubits])
    return u


def _clbit_map(qc: QuantumCircuit) -> Dict[int, int]:
    m = {}
    for ins in qc.data:
        if ins.operation.name == "measure":
            m[qc.find_bit(ins.qubits[0]).index] = qc.find_bit(ins.clbits[0]).index
    return m


def with_input(qc: QuantumCircuit, bitstring: str) -> QuantumCircuit:
    """Prepend X gates preparing computational-basis input |bitstring>,
    where bitstring[-1-i] is qubit i (Qiskit little-endian)."""
    if not bitstring or set(bitstring) == {"0"}:
        return qc
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    for i, b in enumerate(reversed(bitstring)):
        if b == "1":
            out.x(i)
    out.compose(qc, inplace=True)
    return out


def ideal_distribution(qc: QuantumCircuit) -> Dict[str, float]:
    if not check_terminal_measurements(qc):
        raise ValueError("ideal_distribution requires terminal measurements")
    probs = Statevector(unitary_part(qc)).probabilities_dict()
    cmap = _clbit_map(qc)
    out: Dict[str, float] = {}
    for k, p in probs.items():
        if p < 1e-15:
            continue
        bits = k[::-1]
        cb = ["0"] * qc.num_clbits
        for q, c in cmap.items():
            cb[c] = bits[q]
        key = "".join(cb[::-1])
        out[key] = out.get(key, 0.0) + float(p)
    return out


def ideal_jsd(a: QuantumCircuit, b: QuantumCircuit,
              inputs: Iterable[str] = ("",)) -> float:
    """Max over input states of the exact JSD between ideal outputs."""
    return max(jsd(ideal_distribution(with_input(a, s)),
                   ideal_distribution(with_input(b, s)), smooth=False)
               for s in inputs)


def unitary_equivalent(a: QuantumCircuit, b: QuantumCircuit,
                       max_qubits: int = 10) -> Optional[bool]:
    if a.num_qubits > max_qubits:
        return None
    return bool(Operator(unitary_part(a)).equiv(Operator(unitary_part(b))))


EQUIV_TOL = 1e-10
