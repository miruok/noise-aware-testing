# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ5 (drift, part 2): recalibration policies under drift between calibration
and testing.

Calibrate NAO (K = 8) at eta0 = 0.005, then test correct runs at
eta1 = eta0 (1 + r), r in {0, 0.1, 0.25, 0.5}. Before testing, each policy
decides whether to recalibrate (8 new runs at eta1):
  none                  never recalibrate
  sentinel(m, a_s)      execute P m times, pool the m runs, and recalibrate if
                        NAO's two-sample test of the pooled sentinel against
                        the reference rejects at level a_s
                        (m in {1, 2, 4}, a_s in {1e-3, 1e-2})
  co-scheduled          always calibrate in the same session (job) as the
                        tests: the reference is built at eta1
All policies are evaluated on the same executions (paired): one calibration
set at eta0, four sentinel runs, one recalibration set at eta1 and five test
runs per window.
Metrics: FPR of the tests, recalibration rate, executions per session.
Output: results/rq5_sentinel/.
"""
from __future__ import annotations
import argparse, itertools, os, time
import numpy as np
from common import SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, compute_threshold, jsd, pooled, make_noisy_backend

BENCH5 = ["ghz_3", "qaoa_3", "bell_8", "qaoa_8"]
RS = [0.0, 0.1, 0.25, 0.5]
ETA0 = 0.005
ALPHA = 1e-3
POLICIES = ["none"] + [f"sentinel_m{m}_a{a:g}" for m, a in itertools.product((1, 2, 4), (1e-3, 1e-2))] + ["co_scheduled"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=25)
    ap.add_argument("--tests", type=int, default=5); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=101); ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq5_sentinel", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
    rows = []
    for b in BENCH5:
        qc, _ = get_benchmark(b); nq = qc.num_qubits; be0 = make_noisy_backend(ETA0)
        for r in RS:
            be1 = make_noisy_backend(ETA0 * (1 + r))
            fp = {p: 0 for p in POLICIES}; recal = {p: 0 for p in POLICIES}; n = 0
            for _ in range(a.windows):
                cal0 = ex.counts_many(qc, be0, SHOTS, 8); ref0 = pooled(cal0)
                tau0 = compute_threshold(cal0, SHOTS, ALPHA, "pooled", n_qubits=nq, B=a.B, rng=rng).tau
                sent = ex.counts_many(qc, be1, SHOTS, 4)
                cal1 = ex.counts_many(qc, be1, SHOTS, 8); ref1 = pooled(cal1)
                tau1 = compute_threshold(cal1, SHOTS, ALPHA, "pooled", n_qubits=nq, B=a.B, rng=rng).tau
                use = {"none": False, "co_scheduled": True}
                for m, al in itertools.product((1, 2, 4), (1e-3, 1e-2)):
                    ts = compute_threshold(cal0, m * SHOTS, al, "pooled", n_qubits=nq, B=a.B, rng=rng).tau
                    use[f"sentinel_m{m}_a{al:g}"] = jsd(ref0, pooled(sent[:m])) > ts
                tests = ex.counts_many(qc, be1, SHOTS, a.tests); n += len(tests)
                for p in POLICIES:
                    recal[p] += use[p]
                    ref, tau = (ref1, tau1) if use[p] else (ref0, tau0)
                    fp[p] += sum(jsd(ref, c) > tau for c in tests)
            row = {"benchmark": b, "r": r, "tests": n, "windows": a.windows,
                   "fpr": {p: fp[p] / n for p in POLICIES}, "recalibration_rate": {p: recal[p] / a.windows for p in POLICIES}}
            row["executions_per_session"] = {p: 8 + (int(p.split("_m")[1].split("_")[0]) if p.startswith("sentinel") else 0)
                                             + 8 * row["recalibration_rate"][p] for p in POLICIES}
            row["executions_per_session"]["none"] = 8.0
            row["executions_per_session"]["co_scheduled"] = 8.0
            rows.append(row)
            print(b, r, {p: round(v, 3) for p, v in row["fpr"].items()}, flush=True)
    save(rows, os.path.join(d, "results.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ5 RECALIBRATION POLICIES (eta0={ETA0}, alpha={ALPHA}, K=8)\n\nFPR (recalibration rate)\n")
        for x in rows:
            f.write(f"  {x['benchmark']:7s} r={x['r']:<5} " + "  ".join(f"{p}={x['fpr'][p]:.3f}({x['recalibration_rate'][p]:.2f})" for p in POLICIES) + "\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "eta0": ETA0, "alpha": ALPHA, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
