# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Test oracles for NISQ programs.

The noise-aware oracle (NAO) compares a program under test P' with a
trusted reference program P executed on the same backend (differential /
regression setting). It needs no ideal output, noise model or learned model.

  NAO     K calibration runs of P are pooled into the reference; threshold =
           (1 - alpha) quantile of the finite-shot null distribution of the
           JSD; a homogeneity test across the K runs detects drift and only
           then widens the threshold.
  NAO-1   one reference run, same null-quantile threshold, no drift test
           (lowest cost: one extra execution).

Baselines
  SPCEnvelopeOracle   Shewhart-style envelope tau = max(mu + k sigma, tau_min)
                      over pairwise self-divergences of K runs; single-run
                      reference.
  FixedThresholdOracle  ideal (noiseless, sampled) reference, fixed tau.
  QOINStyleOracle     estimate-and-remove noise (per-outcome multiplicative
                      factor from K_c correct runs), ideal reference, fixed tau.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from qiskit import QuantumCircuit

from .divergence import jsd, pooled
from .execution import Executor
from .threshold import compute_threshold, spc_envelope_threshold


class NoiseAwareOracle:
    """mode="pooled" (NAO, default) or mode="single" (NAO-1)."""

    def __init__(self, reference: QuantumCircuit, backend, executor: Executor,
                 shots: int = 4096, mode: str = "pooled", alpha: float = 1e-3,
                 K: int = 8, B: int = 4000, alpha_drift: float = 1e-2, seed: int = 0):
        if mode not in ("pooled", "single"):
            raise ValueError(mode)
        self.P = reference; self.backend = backend; self.ex = executor
        self.shots = shots; self.mode = mode; self.alpha = alpha
        self.K = K if mode == "pooled" else 0
        self.B = B; self.alpha_drift = alpha_drift
        self.rng = np.random.default_rng(seed)
        self.ref_counts = None; self.report = None; self.tau = None

    @property
    def calibration_executions(self) -> int:
        return self.K if self.mode == "pooled" else 1

    def calibrate(self):
        ex, P, b, N = self.ex, self.P, self.backend, self.shots
        if self.mode == "single":
            self.ref_counts = ex.counts(P, b, N)
            self.report = compute_threshold([], N, self.alpha, "single", n_qubits=P.num_qubits,
                                            B=self.B, reference_counts=self.ref_counts, rng=self.rng)
        else:
            cal = ex.counts_many(P, b, N, self.K)
            self.ref_counts = pooled(cal)
            self.report = compute_threshold(cal, N, self.alpha, "pooled", n_qubits=P.num_qubits,
                                            B=self.B, alpha_drift=self.alpha_drift, rng=self.rng)
        self.tau = self.report.tau
        return self

    def evaluate(self, test: QuantumCircuit, counts=None) -> Dict:
        if self.tau is None:
            self.calibrate()
        c = counts if counts is not None else self.ex.counts(test, self.backend, self.shots, layout_from=self.P)
        d = jsd(self.ref_counts, c)
        return {"passed": bool(d <= self.tau), "divergence": float(d), "tau": float(self.tau),
                "drift_detected": bool(self.report.drift_detected)}


class SPCEnvelopeOracle:
    def __init__(self, reference, backend, executor, shots=4096, K=12, k=4.0, tau_min=0.02):
        cal = executor.counts_many(reference, backend, shots, K)
        self.ref = executor.counts(reference, backend, shots)
        self.tau = spc_envelope_threshold(cal, k, tau_min)
        self.P = reference; self.backend = backend; self.ex = executor; self.shots = shots
        self.calibration_executions = K + 1

    def evaluate(self, test, counts=None):
        c = counts if counts is not None else self.ex.counts(test, self.backend, self.shots, layout_from=self.P)
        d = jsd(self.ref, c)
        return {"passed": bool(d <= self.tau), "divergence": float(d), "tau": float(self.tau)}


class FixedThresholdOracle:
    def __init__(self, reference, backend, executor, shots=4096, tau=0.05):
        from qiskit_aer import AerSimulator
        self.ref = executor.counts(reference, AerSimulator(), shots)
        self.backend = backend; self.ex = executor; self.shots = shots; self.tau = tau
        self.P = reference; self.calibration_executions = 0

    def evaluate(self, test, counts=None):
        c = counts if counts is not None else self.ex.counts(test, self.backend, self.shots, layout_from=self.P)
        d = jsd(self.ref, c)
        return {"passed": bool(d <= self.tau), "divergence": float(d), "tau": float(self.tau)}


class QOINStyleOracle:
    def __init__(self, reference, backend, executor, shots=4096, tau=0.05,
                 calibration_runs=5, smoothing=1e-3):
        from qiskit_aer import AerSimulator
        ideal = executor.counts(reference, AerSimulator(), shots)
        n = sum(ideal.values()); self.ideal = {k: v / n for k, v in ideal.items()}
        runs = executor.counts_many(reference, backend, shots, calibration_runs)
        keys = set(self.ideal).union(*[set(r) for r in runs])
        avg = {k: np.mean([r.get(k, 0) / shots for r in runs]) for k in keys}
        self.f = {k: (avg[k] + smoothing) / (self.ideal.get(k, 0.0) + smoothing) for k in keys}
        self.backend = backend; self.ex = executor; self.shots = shots; self.tau = tau
        self.P = reference; self.calibration_executions = calibration_runs

    def evaluate(self, test, counts=None):
        c = counts if counts is not None else self.ex.counts(test, self.backend, self.shots, layout_from=self.P)
        n = sum(c.values())
        corr = {k: (v / n) / max(self.f.get(k, 1.0), 1e-6) for k, v in c.items()}
        t = sum(corr.values()) or 1.0
        corr = {k: v / t for k, v in corr.items()}
        d = jsd(self.ideal, corr)
        return {"passed": bool(d <= self.tau), "divergence": float(d), "tau": float(self.tau)}
