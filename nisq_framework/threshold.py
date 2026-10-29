# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Shot-noise acceptance thresholds and drift detection.

Null model. A correct test run and the reference are independent
multinomial samples (n_test and n_ref shots) from the same outcome
distribution p. For large counts,

    8 ln2 * JSD_2(p_ref, p_test) / (1/n_ref + 1/n_test)  ~  chi2(d - 1),

where d is the number of outcomes with p > 0. Hence a threshold with a
stated false-positive rate alpha is

    tau_chi2(alpha) = chi2_{1-alpha}(d - 1) * (1/n_ref + 1/n_test) / (8 ln2).

Because d is unknown and counts can be sparse, the default threshold is a
parametric bootstrap from a plug-in estimate of p (pooled calibration
counts, with an optional Chao1 correction for outcomes not yet observed).

Drift. Calibration runs taken at different times should be homogeneous if
the backend is stable. A chi-square test of homogeneity on the K x d
contingency table detects over-dispersion (drift or recompilation
changes). NAO widens its threshold only when drift is detected.

The Shewhart-style envelope tau = max(mu + k sigma, tau_min) over pairwise
self-divergences is provided as a baseline (`spc_envelope_threshold`).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from .divergence import jsd_arrays, align, pooled

LN2 = float(np.log(2.0))


def chi2_threshold(d_eff: int, n_ref: int, n_test: int, alpha: float) -> float:
    df = max(int(d_eff) - 1, 1)
    return float(stats.chi2.ppf(1 - alpha, df) * (1 / n_ref + 1 / n_test) / (8 * LN2))


def plugin_distribution(counts: Mapping[str, int], n_qubits: Optional[int] = None,
                        chao1: bool = True):
    """Plug-in estimate of p with an optional Chao1 unseen-outcome correction.

    Returns (keys, probabilities, n_unseen_added)."""
    keys = sorted(counts)
    c = np.array([counts[k] for k in keys], float)
    n = c.sum()
    p = c / n
    added = 0
    if chao1:
        f1 = int((c == 1).sum()); f2 = int((c == 2).sum())
        if f1 > 0:
            s0 = f1 * (f1 - 1) / (2 * (f2 + 1)) if f2 == 0 else f1 * f1 / (2 * f2)
            if n_qubits is not None:
                s0 = min(s0, 2 ** n_qubits - len(keys))
            added = int(np.floor(s0))
            if added > 0:
                m0 = f1 / n                       # Good-Turing unseen mass
                p = np.concatenate([p * (1 - m0), np.full(added, m0 / added)])
                keys = keys + [f"__unseen_{i}" for i in range(added)]
    return keys, p, added


def bootstrap_null(p: np.ndarray, n_ref: int, n_test: int, B: int,
                   rng: np.random.Generator) -> np.ndarray:
    a = rng.multinomial(n_ref, p, size=B)
    b = rng.multinomial(n_test, p, size=B)
    return jsd_arrays(a, b)


@dataclass
class ThresholdReport:
    tau: float
    tau_null: float
    tau_chi2: float
    tau_empirical: Optional[float]
    d_observed: int
    d_unseen_added: int
    drift_detected: bool
    homogeneity_p: Optional[float]
    dispersion_ratio: Optional[float]
    n_ref: int
    n_test: int
    alpha: float
    K: int
    mode: str

    def as_dict(self):
        return asdict(self)


def homogeneity_test(counts_list: Sequence[Mapping[str, int]], min_expected: float = 5.0):
    """Chi-square homogeneity test across K runs; rare outcomes merged."""
    keys, M = align(*counts_list)
    M = M.astype(float)
    tot = M.sum(0)
    exp_min = tot / len(counts_list)
    keep = exp_min >= min_expected
    table = M[:, keep]
    rest = M[:, ~keep].sum(1, keepdims=True)
    if rest.sum() > 0:
        table = np.hstack([table, rest])
    if table.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(table, correction=False)[1])


def compute_threshold(calib_counts: Sequence[Mapping[str, int]],
                      n_test: int, alpha: float = 1e-3, mode: str = "pooled",
                      n_qubits: Optional[int] = None, B: int = 4000,
                      k_emp: float = 4.0, alpha_drift: float = 1e-2,
                      reference_counts: Optional[Mapping[str, int]] = None,
                      rng: Optional[np.random.Generator] = None) -> ThresholdReport:
    """Threshold for comparing an n_test-shot run against a reference.

    mode="single":  reference = `reference_counts` (one run); no calibration
                    runs (NAO-1).
    mode="pooled":  reference = pooled calibration runs; the drift test widens
                    tau to max(tau_null, mu + k_emp*sigma) when triggered (NAO).
    """
    rng = rng or np.random.default_rng(0)
    K = len(calib_counts)
    if mode == "single":
        assert reference_counts is not None
        ref = dict(reference_counts)
    else:
        assert K >= 2, "pooled mode needs >= 2 calibration runs"
        ref = pooled(calib_counts)
    n_ref = int(sum(ref.values()))
    keys, p, added = plugin_distribution(ref, n_qubits=n_qubits)
    null = bootstrap_null(p, n_ref, n_test, B, rng)
    tau_null = float(np.quantile(null, 1 - alpha, method="higher"))
    tau_c = chi2_threshold(len(keys), n_ref, n_test, alpha)

    hom_p = disp = tau_emp = None
    drift = False
    if K >= 2:
        hom_p = homogeneity_test(calib_counts)
        _, M = align(*calib_counts)
        iu = np.triu_indices(K, 1)
        pair = jsd_arrays(M[iu[0]], M[iu[1]])
        n1 = int(M[0].sum())
        exp_pair = bootstrap_null(p, n1, n1, min(B, 2000), rng).mean()
        disp = float(pair.mean() / exp_pair) if exp_pair > 0 else None
        tau_emp = float(pair.mean() + k_emp * pair.std(ddof=1))
        drift = mode == "pooled" and hom_p < alpha_drift
    tau = max(tau_null, tau_emp) if drift else tau_null
    return ThresholdReport(tau=float(tau), tau_null=tau_null, tau_chi2=tau_c,
                           tau_empirical=tau_emp, d_observed=len(ref),
                           d_unseen_added=added, drift_detected=bool(drift),
                           homogeneity_p=hom_p, dispersion_ratio=disp,
                           n_ref=n_ref, n_test=n_test, alpha=alpha, K=K, mode=mode)


def spc_envelope_threshold(calib_counts: Sequence[Mapping[str, int]], k: float = 4.0,
                           tau_min: float = 0.02) -> float:
    """Baseline: Shewhart-style envelope tau = max(mu + k*sigma, tau_min) over
    the pairwise JSDs of the calibration runs."""
    _, M = align(*calib_counts)
    iu = np.triu_indices(len(calib_counts), 1)
    pair = jsd_arrays(M[iu[0]], M[iu[1]])
    return float(max(pair.mean() + k * pair.std(ddof=1), tau_min))
