# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Fault injectors with exact equivalence labelling.

Eight injectors model common quantum programming faults. Every injected variant carries its exact ideal divergence, so variants that
do not change the program's output (for example swapping the operands of a
CX that acts on |++>) are reported as equivalent and excluded from recall.
Variants whose unitary differs from the original but whose measured output
on the tested input does not (for example a phase, or a gate acting on an
eigenstate) are flagged `basis_invisible`: they are faults of the program
that no computational-basis output oracle can observe on that input.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate
from qiskit.circuit.library import XGate, ZGate

from qiskit.circuit.library import XGate as _X
from .semantics import EQUIV_TOL, first_measure_index, ideal_jsd, unitary_equivalent


def _gates(qc):
    m = first_measure_index(qc)
    return [i for i, ins in enumerate(qc.data[:m]) if isinstance(ins.operation, Gate)]


def wrong_angle(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    out = qc.copy()
    for i in _gates(out):
        op = out.data[i].operation
        if op.name in ("rx", "ry", "rz", "p", "cp", "rzz") and op.params:
            new = op.copy(); new.params = [float(op.params[0]) + np.pi / 3] + list(op.params[1:])
            out.data[i] = out.data[i].replace(operation=new)
            return out
    return None


def extra_x(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    out = qc.copy(); g = _gates(out)
    if not g: return None
    pos = g[len(g) // 2]; q = qc.num_qubits // 2
    out.data.insert(pos, CircuitInstruction(XGate(), [out.qubits[q]], []))
    return out


def wrong_target(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    if qc.num_qubits < 3: return None
    out = qc.copy()
    for i in _gates(out):
        ins = out.data[i]
        if ins.operation.name == "cx":
            c, t = [out.find_bit(q).index for q in ins.qubits]
            alt = [q for q in range(qc.num_qubits) if q not in (c, t)]
            out.data[i] = ins.replace(qubits=(out.qubits[c], out.qubits[alt[0]]))
            return out
    return None


def drop_last_gate(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    out = qc.copy(); g = _gates(out)
    if not g: return None
    del out.data[g[-1]]
    return out


def phase_flip(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    out = qc.copy(); g = _gates(out)
    if not g: return None
    pos = g[len(g) // 2]
    out.data.insert(pos, CircuitInstruction(ZGate(), [out.qubits[0]], []))
    return out


def remove_entanglement(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Delete the last CX gate."""
    out = qc.copy()
    idx = [i for i, ins in enumerate(out.data) if ins.operation.name in ("cx", "cnot")]
    if not idx: return out
    del out.data[idx[-1]]
    return out


def wrong_basis(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Replace the first H gate with X."""
    out = qc.copy()
    idx = [i for i, ins in enumerate(out.data) if ins.operation.name == "h"]
    if idx:
        out.data[idx[0]] = out.data[idx[0]].replace(operation=_X())
    return out


def swap_qubits(qc: QuantumCircuit) -> Optional[QuantumCircuit]:
    """Swap the operands of the first two-qubit gate."""
    out = qc.copy()
    for i, ins in enumerate(out.data):
        if len(ins.qubits) == 2:
            q1, q2 = ins.qubits
            out.data[i] = ins.replace(qubits=(q2, q1))
            break
    return out


INJECTORS: Dict[str, Callable[[QuantumCircuit], Optional[QuantumCircuit]]] = {
    "remove_entanglement": remove_entanglement,
    "wrong_basis": wrong_basis,
    "swap_qubits": swap_qubits,
    "wrong_angle": wrong_angle,
    "extra_x": extra_x,
    "wrong_target": wrong_target,
    "drop_last_gate": drop_last_gate,
    "phase_flip": phase_flip,
}


def inject_all(qc: QuantumCircuit, names=None, inputs=("",)) -> List[Dict]:
    out = []
    for name in (names or INJECTORS):
        b = INJECTORS[name](qc)
        if b is None:
            out.append({"injector": name, "applicable": False}); continue
        d = ideal_jsd(qc, b, inputs)
        ueq = unitary_equivalent(qc, b)
        out.append({"injector": name, "applicable": True, "circuit": b,
                    "ideal_jsd": float(d), "output_equivalent": bool(d < EQUIV_TOL),
                    "unitary_equivalent": ueq,
                    "basis_invisible": bool(d < EQUIV_TOL and ueq is False)})
    return out
