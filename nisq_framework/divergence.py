# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Divergence utilities (base-2 Jensen-Shannon divergence)."""
from __future__ import annotations
from typing import Dict, Mapping, Sequence, Tuple
import numpy as np

EPS = 1e-12


def align(*dists: Mapping[str, float]) -> Tuple[list, np.ndarray]:
    keys = sorted(set().union(*[set(d) for d in dists]))
    arr = np.array([[float(d.get(k, 0.0)) for k in keys] for d in dists])
    return keys, arr


def normalise(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return x / s


def jsd_arrays(p: np.ndarray, q: np.ndarray, smooth: bool = True) -> np.ndarray:
    """Base-2 Jensen-Shannon divergence along the last axis (vectorised).

    With smooth=True, eps=1e-12 is added to every aligned component and the
    vectors are renormalised (avoids log(0)).
    """
    p = normalise(np.asarray(p, float)); q = normalise(np.asarray(q, float))
    if smooth:
        p = normalise(p + EPS); q = normalise(q + EPS)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * np.log2(p / m), 0.0).sum(-1)
        kq = np.where(q > 0, q * np.log2(q / m), 0.0).sum(-1)
    return np.clip(0.5 * kp + 0.5 * kq, 0.0, 1.0)


def jsd(p: Mapping[str, float], q: Mapping[str, float], smooth: bool = True) -> float:
    _, a = align(p, q)
    return float(jsd_arrays(a[0], a[1], smooth))


def pooled(counts_list: Sequence[Mapping[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in counts_list:
        for k, v in c.items():
            out[k] = out.get(k, 0) + int(v)
    return out
