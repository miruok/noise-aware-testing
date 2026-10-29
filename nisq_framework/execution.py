# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Seeded, layout-stable circuit execution.

  * every simulator execution receives an explicit `seed_simulator` drawn
    from a master RNG, so a whole experiment is reproducible from one seed;
  * transpilation uses a fixed `seed_transpiler` and is cached per
    (circuit, backend), so repeated executions of one circuit use one
    physical layout (layout_policy="fixed"); layout_policy="random"
    re-transpiles with a fresh seed on every call, which is used to
    study recompilation variability;
  * a test circuit can be compiled onto the reference circuit's initial
    layout (`layout_from=`), so reference and test share physical qubits;
  * results are integer counts, not probabilities, so shot-noise statistics
    can be computed exactly.
"""
from __future__ import annotations

import hashlib
import io
from collections import Counter
from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile


def circuit_key(qc: QuantumCircuit) -> str:
    buf = io.BytesIO()
    qpy.dump(qc, buf)
    return hashlib.sha1(buf.getvalue()).hexdigest()


def _is_runtime_backend(backend) -> bool:
    cls = type(backend)
    mod = getattr(cls, "__module__", "") or ""
    if "fake_provider" in mod or "fake" in cls.__name__.lower():
        return False
    return "qiskit_ibm_runtime" in mod


def backend_name(backend) -> str:
    n = getattr(backend, "name", None)
    n = n() if callable(n) else n
    return str(n or type(backend).__name__)


class Executor:
    """Run circuits and return integer counts.

    Args:
        seed: master seed; simulator seeds are drawn from it.
        layout_policy: "fixed" (cached, seeded transpilation) or "random".
        optimization_level: transpiler level (barriers inserted by
            mutation operators prevent cancellation of inserted gates).
    """

    def __init__(self, seed: int = 2026, layout_policy: str = "fixed",
                 optimization_level: int = 1, seed_transpiler: int = 11):
        if layout_policy not in ("fixed", "random"):
            raise ValueError(layout_policy)
        self.rng = np.random.default_rng(seed)
        self.layout_policy = layout_policy
        self.optimization_level = optimization_level
        self.seed_transpiler = seed_transpiler
        self._tcache: Dict[tuple, QuantumCircuit] = {}
        self.n_executions = 0
        self.n_shots = 0

    # ------------------------------------------------------------ compile
    def compile(self, qc: QuantumCircuit, backend,
                layout_from: Optional[QuantumCircuit] = None) -> QuantumCircuit:
        initial_layout = None
        if layout_from is not None:
            ref_t = self.compile(layout_from, backend)
            if ref_t.layout is not None:
                initial_layout = ref_t.layout.initial_index_layout()[: qc.num_qubits]
        if self.layout_policy == "random":
            return transpile(qc, backend,
                             optimization_level=self.optimization_level,
                             seed_transpiler=int(self.rng.integers(1 << 30)),
                             initial_layout=initial_layout)
        key = (circuit_key(qc), backend_name(backend),
               None if initial_layout is None else tuple(initial_layout))
        if key not in self._tcache:
            self._tcache[key] = transpile(
                qc, backend, optimization_level=self.optimization_level,
                seed_transpiler=self.seed_transpiler,
                initial_layout=initial_layout)
        return self._tcache[key]

    # ------------------------------------------------------------ execute
    def counts(self, qc: QuantumCircuit, backend, shots: int,
               layout_from: Optional[QuantumCircuit] = None) -> Counter:
        self.n_executions += 1
        self.n_shots += shots
        if _is_runtime_backend(backend):
            return self._counts_runtime(qc, backend, shots)
        t = self.compile(qc, backend, layout_from=layout_from)
        res = backend.run(t, shots=shots,
                          seed_simulator=int(self.rng.integers(1 << 30))).result()
        return Counter({k.replace(" ", ""): int(v)
                        for k, v in res.get_counts().items()})

    def _counts_runtime(self, qc, backend, shots) -> Counter:
        from qiskit_ibm_runtime import SamplerV2
        t = self.compile(qc, backend)
        job = SamplerV2(mode=backend).run([t], shots=shots)
        data = job.result()[0].data
        for name in getattr(data, "__dict__", {}) or {}:
            arr = getattr(data, name)
            if hasattr(arr, "get_counts"):
                return Counter({k.replace(" ", ""): int(v)
                                for k, v in arr.get_counts().items()})
        for name in ("meas", "c", "c0"):
            if hasattr(data, name):
                return Counter(getattr(data, name).get_counts())
        raise RuntimeError("no classical register in SamplerV2 result")

    def counts_many(self, qc, backend, shots, k, layout_from=None):
        return [self.counts(qc, backend, shots, layout_from) for _ in range(k)]
