# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Statistics used by the experiments."""
from __future__ import annotations
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
from scipy import stats


def wilson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + conf / 2); p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def cluster_bootstrap(values_by_cluster: Sequence[Sequence[float]], B: int = 5000,
                      conf: float = 0.95, seed: int = 0) -> Dict:
    """Mean of all observations with a CI that resamples whole clusters
    (e.g. all tests sharing one calibration window)."""
    rng = np.random.default_rng(seed)
    clusters = [np.asarray(v, float) for v in values_by_cluster if len(v)]
    sums = np.array([c.sum() for c in clusters]); ns = np.array([len(c) for c in clusters])
    est = sums.sum() / ns.sum()
    idx = rng.integers(0, len(clusters), size=(B, len(clusters)))
    boot = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.quantile(boot, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return {"estimate": float(est), "ci": [float(lo), float(hi)],
            "n_obs": int(ns.sum()), "n_clusters": len(clusters)}


def confusion(y_true: Iterable[bool], y_pred: Iterable[bool]) -> Dict:
    y_true = list(map(bool, y_true)); y_pred = list(map(bool, y_pred))
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and (prec + rec) else (0.0 if tp + fp + fn else float("nan"))
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
            "recall": rec, "f1": f1, "fpr": fpr,
            "recall_ci": list(wilson(tp, tp + fn)), "fpr_ci": list(wilson(fp, fp + tn))}


def cohen_kappa(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y_true, y_pred)) / n
    pe = sum((y_true.count(l) / n) * (list(y_pred).count(l) / n) for l in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def sobol_factorial(rows: List[Dict], factors: Sequence[str], response: str) -> Dict:
    """First-order and total Sobol indices on a full factorial grid via the
    functional ANOVA decomposition (exact for a complete grid with equal
    weights on each level)."""
    y = np.array([r[response] for r in rows], float)
    V = y.var()
    if V == 0:
        return {f: {"S1": 0.0, "ST": 0.0} for f in factors} | {"_variance": 0.0}
    out = {"_variance": float(V)}
    for f in factors:
        lv = sorted({r[f] for r in rows})
        cond = np.array([y[[r[f] == l for r in rows]].mean() for l in lv])
        S1 = cond.var() / V
        others = [g for g in factors if g != f]
        keys = sorted({tuple(r[g] for g in others) for r in rows})
        cm = np.array([y[[tuple(r[g] for g in others) == k for r in rows]].mean() for k in keys])
        ST = 1 - cm.var() / V
        out[f] = {"S1": float(S1), "ST": float(ST)}
    return out


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta in [-1, 1]; |d| >= 0.147 small, 0.33 medium, 0.474 large."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b)))


def cmh_test(tables: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]) -> float:
    """Cochran-Mantel-Haenszel test (continuity-corrected) for K stratified
    2x2 tables ((a, b), (c, d)); rows = groups, columns = (rejected, accepted).
    H0: common odds ratio 1. Returns the two-sided p-value."""
    num = var = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n < 2:
            continue
        r1, c1 = a + b, a + c
        num += a - r1 * c1 / n
        var += r1 * (c + d) * c1 * (b + d) / (n * n * (n - 1))
    if var == 0:
        return 1.0
    return float(stats.chi2.sf((abs(num) - 0.5) ** 2 / var, 1))


def signflip_test(diffs: Sequence[float], n_mc: int = 200_000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test of H0: per-cluster differences are
    symmetric around 0. Exact for up to 16 clusters, Monte Carlo otherwise."""
    d = np.asarray(diffs, float)
    obs = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(list(product([-1, 1], repeat=len(d))))
        sims = np.abs((signs * d).mean(1))
        return float(np.mean(sims >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    sims = np.abs((rng.choice([-1, 1], size=(n_mc, len(d))) * d).mean(1))
    return float((np.sum(sims >= obs - 1e-12) + 1) / (n_mc + 1))
