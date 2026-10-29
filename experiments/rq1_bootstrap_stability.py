# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ1 (soundness, part 4): Monte Carlo stability of the bootstrap threshold.

For one calibration window per (benchmark, noise level), the plug-in estimate
of the reference distribution is held fixed and the null quantile is
re-estimated with B = 1000, 4000 and 40000 bootstrap pairs under 30
independent seeds. A reference null of 400000 pairs from the same plug-in
estimate gives the exact Monte Carlo target, so the experiment isolates
the bootstrap's Monte Carlo error from the plug-in error measured elsewhere
in RQ1. For every estimated threshold it reports the conditional
false-positive rate it implies under the reference null, P(D > tau).
Output: results/rq1_bootstrap_stability/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from common import BENCH, NOISE, SHOTS, RESULTS, save, write_meta, get_benchmark, out_dir
from nisq_framework import Executor, make_noisy_backend, pooled
from nisq_framework.threshold import plugin_distribution, bootstrap_null

ALPHA = 1e-3
BS = (1000, 4000, 40000)
LEVELS = ("low", "medium", "high")


def null_chunked(p, n_ref, n_test, B, rng, chunk=20000):
    out = []
    while B > 0:
        b = min(chunk, B); out.append(bootstrap_null(p, n_ref, n_test, b, rng)); B -= b
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--ref-B", type=int, default=400000); ap.add_argument("--seed", type=int, default=2718)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq1_bootstrap_stability", a.out)
    ex = Executor(seed=a.seed); rng0 = np.random.default_rng(a.seed); rows = []
    for nl in LEVELS:
        be = make_noisy_backend(NOISE[nl])
        for b in BENCH:
            qc, _ = get_benchmark(b)
            cal = ex.counts_many(qc, be, SHOTS, 8)
            ref = pooled(cal); n_ref = sum(ref.values())
            _, p, _ = plugin_distribution(ref, n_qubits=qc.num_qubits)
            null_ref = np.sort(null_chunked(p, n_ref, SHOTS, a.ref_B, rng0))
            tau_ref = float(np.quantile(null_ref, 1 - ALPHA, method="higher"))
            row = {"benchmark": b, "noise": nl, "outcomes": int(len(p)), "tau_reference": tau_ref, "by_B": {}}
            for B in BS:
                taus = []
                for s in range(a.seeds):
                    rng = np.random.default_rng([a.seed, B, s, len(rows)])
                    taus.append(float(np.quantile(null_chunked(p, n_ref, SHOTS, B, rng), 1 - ALPHA, method="higher")))
                taus = np.array(taus)
                cond = 1.0 - np.searchsorted(null_ref, taus, side="right") / len(null_ref)
                row["by_B"][str(B)] = {"tau_mean": float(taus.mean()), "tau_rel_sd": float(taus.std(ddof=1) / tau_ref),
                                       "cond_fpr_mean": float(cond.mean()), "cond_fpr_p5": float(np.percentile(cond, 5)),
                                       "cond_fpr_p95": float(np.percentile(cond, 95))}
            rows.append(row)
            x = row["by_B"]["4000"]
            print(f"{nl:6s} {b:7s} d={len(p):4d} B=4000: rel sd {x['tau_rel_sd']:.3f}, conditional FPR mean {x['cond_fpr_mean']:.5f} "
                  f"[{x['cond_fpr_p5']:.5f}, {x['cond_fpr_p95']:.5f}]", flush=True)
    summ = {}
    for B in BS:
        k = str(B)
        summ[k] = {"tau_rel_sd_median": float(np.median([r["by_B"][k]["tau_rel_sd"] for r in rows])),
                   "tau_rel_sd_max": float(max(r["by_B"][k]["tau_rel_sd"] for r in rows)),
                   "cond_fpr_mean": float(np.mean([r["by_B"][k]["cond_fpr_mean"] for r in rows])),
                   "cond_fpr_p5_min": float(min(r["by_B"][k]["cond_fpr_p5"] for r in rows)),
                   "cond_fpr_p95_max": float(max(r["by_B"][k]["cond_fpr_p95"] for r in rows))}
    save({"rows": rows, "summary": summ}, os.path.join(d, "results.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ1 BOOTSTRAP STABILITY at alpha = {ALPHA} ({a.seeds} seeds per B; reference null of {a.ref_B} pairs)\n")
        for k, v in summ.items():
            f.write(f"  B={k:>6s}: relative sd of tau median {v['tau_rel_sd_median']:.3f} (max {v['tau_rel_sd_max']:.3f}); "
                    f"conditional FPR mean {v['cond_fpr_mean']:.5f}, 5th-95th percentile range over cells [{v['cond_fpr_p5_min']:.5f}, {v['cond_fpr_p95_max']:.5f}]\n")
    write_meta(d, vars(a) | {"alpha": ALPHA, "Bs": BS, "levels": LEVELS, "shots": SHOTS, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
