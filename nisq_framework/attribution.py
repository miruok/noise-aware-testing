# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Bug-noise attribution by noise scaling.

Question: when an oracle reports a difference between reference P and test
P', is the difference semantic (BUG) or produced by hardware noise (NOISE)?

Ground truth is exact and non-circular: a case is BUG iff the ideal
(statevector) outputs of P and P' differ; NOISE otherwise (for example a
semantics-preserving mutant that adds noisy gates).

Protocol (hardware-executable, no noiseless simulator needed):
  1. Fold both circuits globally, U -> U (U^dag U)^m, giving noise scale
     s = 2m + 1 in {1, 3, 5}; barriers prevent the transpiler from undoing
     the folds.
  2. At each scale estimate the output distributions of P and P'.
  3. Extrapolate every outcome probability to s = 0 (zero-noise
     extrapolation: linear Richardson on s = 1, 3, quadratic on s = 1, 3, 5,
     or exponential on s = 1, 3 under a global depolarizing model), project
     onto the simplex, and compute the JS distance r_0 between the two
     extrapolated distributions.
  4. Compare r_0 with a parametric-bootstrap null built from the pooled
     counts at each scale (no difference). r_0 above the null (1 - alpha)
     quantile -> BUG; otherwise NOISE.

A semantic fault keeps a non-zero distance as noise is removed; a
noise-induced difference shrinks towards zero.

`reexecution_triage` is a simulator-based baseline: it re-runs P and P'
with 4x shots, with a 0.5x noise model, and on a noiseless simulator, and
labels BUG iff the noiseless comparison exceeds a fixed tolerance (0.1).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from qiskit import QuantumCircuit

from .divergence import align, jsd, jsd_arrays, pooled
from .semantics import unitary_part


def fold(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd integer >= 1")
    u = unitary_part(qc, keep_barriers=True)   # keeps inserted pairs intact
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=f"{qc.name}_fold{scale}")
    out.compose(u, qubits=range(qc.num_qubits), inplace=True)
    for _ in range((scale - 1) // 2):
        out.barrier()
        out.compose(u.inverse(), qubits=range(qc.num_qubits), inplace=True)
        out.barrier()
        out.compose(u, qubits=range(qc.num_qubits), inplace=True)
    out.barrier()
    for ins in qc.data:
        if ins.operation.name == "measure":
            out.measure(qc.find_bit(ins.qubits[0]).index, qc.find_bit(ins.clbits[0]).index)
    return out


def _extrapolate(scales, r, method):
    s = np.asarray(scales, float); r = np.asarray(r, float)
    if method == "linear":
        # Richardson on the first two scales
        s1, s2 = s[0], s[1]
        return (s2 * r[..., 0] - s1 * r[..., 1]) / (s2 - s1)
    if method == "quadratic":
        # exact quadratic through three points evaluated at 0
        s1, s2, s3 = s[:3]
        l1 = (s2 * s3) / ((s1 - s2) * (s1 - s3))
        l2 = (s1 * s3) / ((s2 - s1) * (s2 - s3))
        l3 = (s1 * s2) / ((s3 - s1) * (s3 - s2))
        return l1 * r[..., 0] + l2 * r[..., 1] + l3 * r[..., 2]
    raise ValueError(method)


def _exp_extrapolate(X, scales, n_outcomes):
    """Exponential zero-noise extrapolation under a global depolarizing model,
    p(s) = u + (p_0 - u) f^s with u the uniform distribution over all
    n_outcomes outcomes. f is estimated per circuit from the L2 distance to u
    at the first two scales; p_0 = u + (p(s1) - u) / f^s1.
    X has shape (..., scales, keys) and holds probabilities over observed keys;
    unobserved outcomes have probability 0 and enter the norm through u."""
    s1, s2 = float(scales[0]), float(scales[1])
    u = 1.0 / n_outcomes
    k = X.shape[-1]
    def dist(p):
        return np.sqrt(((p - u) ** 2).sum(-1) + max(n_outcomes - k, 0) * u * u)
    d1, d2 = dist(X[..., 0, :]), dist(X[..., 1, :])
    ratio = np.clip(d2 / np.maximum(d1, 1e-15), 1e-6, 1.0)
    f = ratio ** (1.0 / (s2 - s1))
    return u + (X[..., 0, :] - u) / (f ** s1)[..., None]


def _zne_distance(cPs, cTs, scales, method, keys, n_outcomes=None):
    """Extrapolate each outcome probability of P and T to zero noise
    (Richardson or exponential), project onto the simplex, return the JS
    distance."""
    P = np.stack([cp for cp in cPs], -2); T = np.stack([ct for ct in cTs], -2)
    P = P / P.sum(-1, keepdims=True); T = T / T.sum(-1, keepdims=True)
    if method == "exponential":
        if n_outcomes is None:
            raise ValueError("exponential extrapolation needs n_outcomes")
        P0 = np.clip(_exp_extrapolate(P, scales, n_outcomes), 0, None)
        T0 = np.clip(_exp_extrapolate(T, scales, n_outcomes), 0, None)
    else:
        P0 = np.clip(_extrapolate(scales, np.moveaxis(P, -2, -1), method), 0, None)
        T0 = np.clip(_extrapolate(scales, np.moveaxis(T, -2, -1), method), 0, None)
    return np.sqrt(jsd_arrays(P0, T0, smooth=False))


def attribute_zne(P: QuantumCircuit, T: QuantumCircuit, backend, executor,
                  shots: int = 4096, scales: Sequence[int] = (1, 3, 5),
                  method: str = "linear", alpha: float = 1e-2, B: int = 2000,
                  seed: int = 0) -> Dict:
    """Zero-noise-extrapolated comparison of P and T.

    r0 = JS distance between the zero-noise extrapolated output
    distributions of P and T. Null: at every scale, P and T counts are
    redrawn from their pooled distribution (no difference at any scale);
    r0 above the null (1 - alpha) quantile -> BUG, otherwise NOISE.
    """
    raw = []
    for s_ in scales:
        Ps, Ts = fold(P, s_), fold(T, s_)
        raw.append((executor.counts(Ps, backend, shots),
                    executor.counts(Ts, backend, shots, layout_from=Ps)))
    return attribute_from_counts([a for a, _ in raw], [b for _, b in raw], scales,
                                 method=method, alpha=alpha, B=B, seed=seed,
                                 n_outcomes=2 ** P.num_clbits)


def attribute_from_counts(counts_P, counts_T, scales, method="linear",
                          alpha=1e-2, B=2000, seed=0, n_outcomes=None) -> Dict:
    """Offline ZNE attribution from counts measured at each noise scale
    (used for hardware data collected in batch jobs)."""
    rng = np.random.default_rng(seed)
    keys = sorted(set().union(*[set(a) | set(b) for a, b in zip(counts_P, counts_T)]))
    vec = lambda c: np.array([c.get(k, 0) for k in keys], float)
    cP = [vec(a) for a in counts_P]; cT = [vec(b) for b in counts_T]
    r0 = float(_zne_distance(cP, cT, scales, method, keys, n_outcomes))
    nP, nT = [], []
    for a, b in zip(cP, cT):
        p = (a + b) / (a + b).sum()
        nP.append(rng.multinomial(int(a.sum()), p, size=B).astype(float))
        nT.append(rng.multinomial(int(b.sum()), p, size=B).astype(float))
    null0 = _zne_distance(nP, nT, scales, method, keys, n_outcomes)
    q = float(np.quantile(null0, 1 - alpha))
    r_by = {int(s_): float(np.sqrt(jsd(dict(zip(keys, a)), dict(zip(keys, b)))))
            for s_, a, b in zip(scales, cP, cT)}
    return {"attribution": "bug" if r0 > q else "noise", "r0": r0, "null_q": q,
            "r_by_scale": r_by, "method": method, "executions": 2 * len(scales)}


def reexecution_triage(P, T, noise_level: float, executor, shots: int = 4096,
                          tolerance: float = 0.1) -> Dict:
    from qiskit_aer import AerSimulator
    from .noise import make_noisy_backend
    noisy, reduced, ideal = (make_noisy_backend(noise_level),
                             make_noisy_backend(noise_level * 0.5), AerSimulator())
    d_big = jsd(executor.counts(P, noisy, 4 * shots), executor.counts(T, noisy, 4 * shots))
    d_red = jsd(executor.counts(P, reduced, shots), executor.counts(T, reduced, shots))
    d_id = jsd(executor.counts(P, ideal, shots), executor.counts(T, ideal, shots))
    passed = {"increased_shots": d_big <= tolerance, "reduced_noise": d_red <= tolerance,
              "ideal": d_id <= tolerance}
    if not passed["ideal"]:
        lab = "bug"
    elif passed["increased_shots"] or passed["reduced_noise"]:
        lab = "noise"
    else:
        lab = "inconclusive"
    return {"attribution": lab, "divergences": {"increased_shots": d_big,
            "reduced_noise": d_red, "ideal": d_id}}
