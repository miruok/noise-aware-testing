# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Noise-aware mutation engine with exact semantic labels.

Placement. Inserted gates go to a uniformly random position before the first
measurement (gates placed after measurement cannot affect the measured
output and would yield equivalent mutants by construction).

Ground truth. Every mutant carries its exact ideal divergence from the
original (statevector), so equivalent mutants are known before execution.
Semantic operators resample until they produce a non-equivalent mutant and
record how many equivalent draws were rejected.

Taxonomy (three classes, by semantic and physical effect):

    semantic                ideal output changes. Two operator families:
                            additive (insert CX, T, X, rotation with
                            |theta| in [pi/8, pi/2]) and subtractive (remove a
                            gate, a single-qubit gate, or a rotation).
    noise-amplifying        ideal output identical, physical noise increased
                            (barrier-protected inverse pairs, CX pairs,
                            rotation pairs).
    noise-neutral           ideal output identical, structure rewritten
                            (commuting swap, CX -> H CZ H, X -> H Z H).

The two semantics-preserving classes are negative controls. Whether a rejection of a
noise-amplifying mutant is a noise effect or a semantic fault is
decided by bug-noise attribution (nisq_framework.attribution).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate
from qiskit.circuit.library import (Barrier, CXGate, CZGate, HGate, IGate,
                                    RXGate, RYGate, RZGate, SdgGate, SGate,
                                    TdgGate, TGate, XGate, YGate, ZGate)

from .semantics import (EQUIV_TOL, first_measure_index, ideal_jsd,
                        unitary_equivalent)

NOISE_AMPLIFYING = "noise_amplifying"
NOISE_NEUTRAL = "noise_neutral"
SEM_ADD = "semantic_additive"
SEM_SUB = "semantic_subtractive"
FAMILIES = (NOISE_AMPLIFYING, NOISE_NEUTRAL, SEM_ADD, SEM_SUB)      # generation order
EQUIVALENT_FAMILIES = (NOISE_AMPLIFYING, NOISE_NEUTRAL)
TAXONOMY_CLASS = {NOISE_AMPLIFYING: "noise_amplifying", NOISE_NEUTRAL: "noise_neutral",
                  SEM_ADD: "semantic", SEM_SUB: "semantic"}

OPERATORS = {
    NOISE_AMPLIFYING: ("inverse_pair_1q", "cx_pair", "rotation_pair"),
    NOISE_NEUTRAL: ("commute_swap", "cx_to_h_cz_h", "x_to_h_z_h"),
    SEM_ADD: ("insert_cnot", "insert_t", "insert_x", "insert_rotation"),
    SEM_SUB: ("remove_gate", "remove_1q_gate", "rotation_to_identity"),
}

_PAIRS = [(HGate, HGate), (XGate, XGate), (YGate, YGate), (ZGate, ZGate),
          (SGate, SdgGate), (TGate, TdgGate)]


@dataclass
class Mutant:
    circuit: QuantumCircuit
    family: str
    operator: str
    position: int
    ideal_jsd: float
    output_equivalent: bool
    unitary_equivalent: Optional[bool]
    rejected_equivalent_draws: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def semantic_fault(self) -> bool:
        return not self.output_equivalent

    def record(self) -> Dict[str, Any]:
        return {"family": self.family, "taxonomy_class": TAXONOMY_CLASS[self.family],
                "operator": self.operator,
                "position": self.position, "ideal_jsd": self.ideal_jsd,
                "output_equivalent": self.output_equivalent,
                "unitary_equivalent": self.unitary_equivalent,
                "rejected_equivalent_draws": self.rejected_equivalent_draws,
                **{f"meta_{k}": v for k, v in self.meta.items()}}


def _qidx(qc, ins):
    return [qc.find_bit(q).index for q in ins.qubits]


class MutationEngine:
    def __init__(self, circuit: QuantumCircuit, seed: int = 42,
                 inputs: Sequence[str] = ("",), max_tries: int = 60):
        self.original = circuit
        self.rng = np.random.default_rng(seed)
        self.inputs = tuple(inputs)
        self.max_tries = max_tries
        self.n = circuit.num_qubits
        self.mpos = first_measure_index(circuit)
        self.gate_idx = [i for i, ins in enumerate(circuit.data[: self.mpos])
                         if isinstance(ins.operation, Gate)]

    # -------------------------------------------------------------- helpers
    def _insert(self, qc, pos, ops):
        """Insert a list of (gate, qubits) at position `pos`."""
        for k, (g, qs) in enumerate(ops):
            qc.data.insert(pos + k, CircuitInstruction(g, [qc.qubits[q] for q in qs], []))

    def _barrier(self, qs):
        return (Barrier(len(qs)), qs)

    def _pos(self):
        return int(self.rng.integers(0, self.mpos + 1))

    # ------------------------------------------------------------ operators
    def _apply(self, op: str):
        qc = self.original.copy()
        n = self.n
        meta: Dict[str, Any] = {}
        if op == "inverse_pair_1q":
            q = int(self.rng.integers(n)); a, b = _PAIRS[self.rng.integers(len(_PAIRS))]
            pos = self._pos(); self._insert(qc, pos, [self._barrier([q]), (a(), [q]), self._barrier([q]), (b(), [q]), self._barrier([q])])
            meta = {"pair": a().name + "-" + b().name, "qubit": q}
        elif op == "cx_pair":
            if n < 2: return None
            c, t = map(int, self.rng.choice(n, 2, replace=False)); pos = self._pos()
            self._insert(qc, pos, [self._barrier([c, t]), (CXGate(), [c, t]), self._barrier([c, t]), (CXGate(), [c, t]), self._barrier([c, t])])
            meta = {"control": c, "target": t}
        elif op == "rotation_pair":
            # identity gates are removed by the transpiler even between
            # barriers, so a rotation and its inverse is used instead
            q = int(self.rng.integers(n)); pos = self._pos()
            th = float(self.rng.uniform(np.pi / 8, np.pi / 2))
            G = [RXGate, RYGate][self.rng.integers(2)]
            self._insert(qc, pos, [self._barrier([q]), (G(th), [q]), self._barrier([q]), (G(-th), [q]), self._barrier([q])])
            meta = {"axis": G(th).name, "theta": th, "qubit": q}
        elif op == "commute_swap":
            cands = [i for i in range(self.mpos - 1)
                     if isinstance(qc.data[i].operation, Gate) and isinstance(qc.data[i + 1].operation, Gate)
                     and set(_qidx(qc, qc.data[i])).isdisjoint(_qidx(qc, qc.data[i + 1]))]
            if not cands: return None
            i = int(self.rng.choice(cands)); qc.data[i], qc.data[i + 1] = qc.data[i + 1], qc.data[i]
            pos = i; meta = {"swapped": [i, i + 1]}
        elif op == "cx_to_h_cz_h":
            cands = [i for i in self.gate_idx if qc.data[i].operation.name == "cx"]
            if not cands: return None
            i = int(self.rng.choice(cands)); c, t = _qidx(qc, qc.data[i])
            del qc.data[i]; self._insert(qc, i, [(HGate(), [t]), (CZGate(), [c, t]), (HGate(), [t])]); pos = i
        elif op == "x_to_h_z_h":
            cands = [i for i in self.gate_idx if qc.data[i].operation.name == "x"]
            if not cands: return None
            i = int(self.rng.choice(cands)); (t,) = _qidx(qc, qc.data[i])
            del qc.data[i]; self._insert(qc, i, [(HGate(), [t]), (ZGate(), [t]), (HGate(), [t])]); pos = i
        elif op == "insert_cnot":
            if n < 2: return None
            c, t = map(int, self.rng.choice(n, 2, replace=False)); pos = self._pos()
            self._insert(qc, pos, [(CXGate(), [c, t])]); meta = {"control": c, "target": t}
        elif op == "insert_t":
            q = int(self.rng.integers(n)); pos = self._pos(); self._insert(qc, pos, [(TGate(), [q])]); meta = {"qubit": q}
        elif op == "insert_x":
            q = int(self.rng.integers(n)); pos = self._pos(); self._insert(qc, pos, [(XGate(), [q])]); meta = {"qubit": q}
        elif op == "insert_rotation":
            q = int(self.rng.integers(n)); pos = self._pos()
            th = float(self.rng.uniform(np.pi / 8, np.pi / 2) * self.rng.choice([-1, 1]))
            G = [RXGate, RYGate, RZGate][self.rng.integers(3)]
            self._insert(qc, pos, [(G(th), [q])]); meta = {"axis": G(th).name, "theta": th}
        elif op == "remove_gate":
            if not self.gate_idx: return None
            i = int(self.rng.choice(self.gate_idx)); meta = {"removed": qc.data[i].operation.name}
            del qc.data[i]; pos = i
        elif op == "remove_1q_gate":
            cands = [i for i in self.gate_idx if len(qc.data[i].qubits) == 1]
            if not cands: return None
            i = int(self.rng.choice(cands)); meta = {"removed": qc.data[i].operation.name}
            del qc.data[i]; pos = i
        elif op == "rotation_to_identity":
            cands = [i for i in self.gate_idx if qc.data[i].operation.name in ("rx", "ry", "rz", "p", "u1")]
            if not cands: return None
            i = int(self.rng.choice(cands)); meta = {"removed": qc.data[i].operation.name,
                                                     "theta": float(qc.data[i].operation.params[0])}
            del qc.data[i]; pos = i
        else:
            raise ValueError(op)
        return qc, pos, meta

    # ---------------------------------------------------------------- public
    def generate_one(self, family: str) -> Optional[Mutant]:
        ops = list(OPERATORS[family])
        rejected = 0
        for _ in range(self.max_tries):
            op = str(self.rng.choice(ops))
            res = self._apply(op)
            if res is None:
                continue
            qc, pos, meta = res
            d = ideal_jsd(self.original, qc, self.inputs)
            eq = d < EQUIV_TOL
            if family in EQUIVALENT_FAMILIES:
                if not eq:
                    raise AssertionError(f"{op} changed semantics (JSD={d})")
            elif eq:
                rejected += 1
                continue
            return Mutant(qc, family, op, pos, float(d), bool(eq),
                            unitary_equivalent(self.original, qc), rejected, meta)
        return None

    def generate(self, n: int, family: str) -> List[Mutant]:
        out = []
        for _ in range(n):
            m = self.generate_one(family)
            if m is not None:
                out.append(m)
        return out
