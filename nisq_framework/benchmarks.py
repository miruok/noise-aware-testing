# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Benchmark programs (2 to 16 qubits). Every builder returns
(QuantumCircuit, metadata); all circuits end with measurement of every qubit."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit


def _meta(name: str, qc: QuantumCircuit, **extras) -> Dict:
    d = {"name": name, "num_qubits": qc.num_qubits, "depth": qc.depth(),
         "size": qc.size()}
    d.update(extras)
    return d


def bell() -> Tuple[QuantumCircuit, Dict]:
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc, _meta("bell", qc)


def ghz(n: int = 3) -> Tuple[QuantumCircuit, Dict]:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(0, i + 1)
    qc.measure(range(n), range(n))
    return qc, _meta("ghz", qc)


def qft(n: int = 3) -> Tuple[QuantumCircuit, Dict]:
    qc = QuantumCircuit(n, n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / float(2 ** (k - j)), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    qc.measure(range(n), range(n))
    return qc, _meta("qft", qc)


def qaoa_small(n: int = 3, gamma: float = 0.5, beta: float = 0.4
               ) -> Tuple[QuantumCircuit, Dict]:
    """1-layer QAOA on a 3-qubit ring."""
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    edges = [(i, (i + 1) % n) for i in range(n)]
    for (i, j) in edges:
        qc.cx(i, j); qc.rz(2 * gamma, j); qc.cx(i, j)
    for i in range(n):
        qc.rx(2 * beta, i)
    qc.measure(range(n), range(n))
    return qc, _meta("qaoa_small", qc, p=1, gamma=gamma, beta=beta)


def vqe_small(n: int = 3, theta: float = 0.7
              ) -> Tuple[QuantumCircuit, Dict]:
    """Tiny VQE-style ansatz at fixed parameter."""
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.ry(theta, i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n):
        qc.ry(theta / 2, i)
    qc.measure(range(n), range(n))
    return qc, _meta("vqe_small", qc, theta=theta)



# -------------------------------------------------------------------- Bell/GHZ


def bell_n(n: int = 8) -> Tuple[QuantumCircuit, Dict]:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)  # chain
    qc.measure(range(n), range(n))
    return qc, _meta("bell_n", qc, topology="chain")


def ghz_n(n: int = 10) -> Tuple[QuantumCircuit, Dict]:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(0, i + 1)  # star
    qc.measure(range(n), range(n))
    return qc, _meta("ghz_n", qc, topology="star")


# ------------------------------------------------------------------ QFT, Grover


def qft_n(n: int = 8) -> Tuple[QuantumCircuit, Dict]:
    qc = QuantumCircuit(n, n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cp(np.pi / float(2 ** (k - j)), k, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    qc.measure(range(n), range(n))
    return qc, _meta("qft_n", qc)


def grover_n(n: int = 6, marked: Optional[int] = None,
             iterations: Optional[int] = None) -> Tuple[QuantumCircuit, Dict]:
    if marked is None:
        marked = (1 << (n - 1)) | 1
    if iterations is None:
        iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(2 ** n))))
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    bits = [(marked >> i) & 1 for i in range(n)]
    for _ in range(iterations):
        # oracle
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        # diffuser
        qc.h(range(n)); qc.x(range(n))
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(range(n)); qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc, _meta("grover_n", qc, marked=marked, iterations=iterations)


# ------------------------------------------------------------- QAOA, HWE-VQE


def qaoa_maxcut_ring(n: int = 8, p: int = 2,
                     gamma: Optional[List[float]] = None,
                     beta: Optional[List[float]] = None,
                     ) -> Tuple[QuantumCircuit, Dict]:
    if gamma is None:
        gamma = [0.4 * (k + 1) for k in range(p)]
    if beta is None:
        beta = [0.6 / (k + 1) for k in range(p)]
    edges = [(i, (i + 1) % n) for i in range(n)]
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for layer in range(p):
        for (i, j) in edges:
            qc.cx(i, j); qc.rz(2.0 * gamma[layer], j); qc.cx(i, j)
        for i in range(n):
            qc.rx(2.0 * beta[layer], i)
    qc.measure(range(n), range(n))
    return qc, _meta("qaoa_maxcut_ring", qc, p=p, gamma=gamma, beta=beta,
                     num_edges=len(edges))


def hwe_vqe(n: int = 8, layers: int = 3, theta_seed: int = 7
            ) -> Tuple[QuantumCircuit, Dict]:
    rng = np.random.default_rng(theta_seed)
    qc = QuantumCircuit(n, n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(-np.pi, np.pi)), q)
            qc.rz(float(rng.uniform(-np.pi, np.pi)), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(float(rng.uniform(-np.pi, np.pi)), q)
    qc.measure(range(n), range(n))
    return qc, _meta("hwe_vqe", qc, layers=layers, parameter_seed=theta_seed)


# ------------------------------------------------------------------ Clifford


def random_clifford(n: int = 8, num_layers: int = 6, seed: int = 13
                    ) -> Tuple[QuantumCircuit, Dict]:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n)
    cliffords = ["h", "s", "x", "y", "z"]
    for _ in range(num_layers):
        for q in range(n):
            getattr(qc, str(rng.choice(cliffords)))(q)
        perm = list(rng.permutation(n))
        for i in range(0, n - 1, 2):
            qc.cx(int(perm[i]), int(perm[i + 1]))
    qc.measure(range(n), range(n))
    return qc, _meta("random_clifford", qc, num_layers=num_layers, seed=seed)



BENCHMARKS = {
    "bell": bell,
    "ghz_3": lambda: ghz(3),
    "qft_3": lambda: qft(3),
    "qaoa_3": lambda: qaoa_small(3),
    "vqe_3": lambda: vqe_small(3),
    "bell_8": lambda: bell_n(8),
    "ghz_8": lambda: ghz_n(8),
    "qaoa_8": lambda: qaoa_maxcut_ring(8, p=1),
    "hwe_6": lambda: hwe_vqe(6, layers=2),
}
PRETTY = {"bell": "Bell", "ghz_3": "GHZ-3", "qft_3": "QFT-3", "qaoa_3": "QAOA-3", "vqe_3": "VQE-3",
          "bell_8": "Bell-8", "ghz_8": "GHZ-8", "qaoa_8": "QAOA-8", "hwe_6": "HWE-6"}


def get(name: str):
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name]()


# ---------------------------------------------------------------- extended suite
QASMBENCH = ["adder_n10", "adder_n4", "basis_change_n3", "basis_trotter_n4", "bell_n4", "cat_state_n4",
             "deutsch_n2", "dnn_n2", "dnn_n8", "error_correctiond3_n5", "fredkin_n3", "grover_n2", "hhl_n7",
             "hs4_n4", "ising_n10", "iswap_n2", "linearsolver_n3", "lpn_n5", "pea_n5", "qaoa_n6", "qec_en_n5",
             "qft_n4", "qpe_n9", "qrng_n4", "quantumwalks_n2", "sat_n7", "simon_n6", "teleportation_n3",
             "toffoli_n3", "variational_n4", "vqe_n4", "wstate_n3"]


def load_qasmbench(name: str):
    """Load a QASMBench (small) program shipped in benchmarks/qasmbench/.
    Barriers are kept; the circuit is returned unchanged otherwise."""
    import os
    from qiskit import QuantumCircuit
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "qasmbench")
    qc = QuantumCircuit.from_qasm_file(os.path.join(root, f"{name}.qasm"))
    qc.name = name
    return qc
