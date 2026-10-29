# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Analytic detection power of the shot-noise threshold.

Under a fault, the test distribution q differs from p. With
lambda = 8 ln2 * JSD_2(p, q) / (1/n_ref + 1/n_test), the scaled statistic is
approximately noncentral chi2(d - 1, lambda). This gives recall as a
function of the fault's true divergence, the shot budget and alpha, and
the shots needed to detect a fault of a given size.
"""
from __future__ import annotations
import numpy as np
from scipy import stats, optimize

LN2 = float(np.log(2.0))


def detection_power(delta: float, d_eff: int, n_ref: int, n_test: int,
                    alpha: float) -> float:
    df = max(d_eff - 1, 1)
    scale = (1 / n_ref + 1 / n_test)
    crit = stats.chi2.ppf(1 - alpha, df)
    lam = 8 * LN2 * delta / scale
    return float(stats.ncx2.sf(crit, df, lam)) if lam > 0 else float(alpha)


def min_detectable_delta(d_eff: int, n_ref: int, n_test: int, alpha: float,
                         power: float = 0.8) -> float:
    f = lambda d: detection_power(d, d_eff, n_ref, n_test, alpha) - power
    return float(optimize.brentq(f, 1e-9, 1.0))


def shots_needed(delta: float, d_eff: int, alpha: float, power: float = 0.8,
                 ref_multiple: float = 1.0) -> int:
    """Smallest n_test (with n_ref = ref_multiple * n_test) reaching `power`."""
    f = lambda n: detection_power(delta, d_eff, max(1, int(ref_multiple * n)),
                                  max(1, int(n)), alpha) - power
    hi = 16
    while f(hi) < 0 and hi < 1e9:
        hi *= 2
    return int(np.ceil(optimize.brentq(f, 1, hi)))
