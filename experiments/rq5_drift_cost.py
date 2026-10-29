# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ5 (drift and cost).

(a) Drift between calibration and testing: calibrate at eta0 = 0.005, test
    correct runs at eta1 = eta0 (1 + r), r in {0, 0.1, 0.25, 0.5, 1.0}.
    FPR of each oracle without and with a sentinel policy (one run of P at test
    time; if NAO rejects it, recalibrate before testing).
(b) Drift inside the calibration window: K = 8 calibration runs with eta
    drifting linearly from eta0 to eta0 (1 + r); detection rate of NAO's
    homogeneity test.
(c) Cost model: executions per test campaign of M tests.
Output: results/rq5_drift_cost/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from common import SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, compute_threshold, spc_envelope_threshold, jsd, pooled, make_noisy_backend
from nisq_framework.stats import wilson

BENCH5 = ["ghz_3", "qaoa_3", "bell_8", "qaoa_8"]
RS = [0.0, 0.1, 0.25, 0.5, 1.0]
ETA0 = 0.005
ALPHA = 1e-3
ORACLES = ["nao1", "nao", "spc_envelope"]


def build(qc, cal12, ref1, rng, B):
    nq = qc.num_qubits
    return {"nao1": (ref1, compute_threshold([], SHOTS, ALPHA, "single", n_qubits=nq, B=B, reference_counts=ref1, rng=rng).tau),
            "nao": (pooled(cal12[:8]), compute_threshold(cal12[:8], SHOTS, ALPHA, "pooled", n_qubits=nq, B=B, rng=rng).tau),
            "spc_envelope": (ref1, spc_envelope_threshold(cal12))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=25)
    ap.add_argument("--tests", type=int, default=5); ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=99); ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq5_drift_cost", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
    A, Bw = [], []
    for b in BENCH5:
        qc, _ = get_benchmark(b); be0 = make_noisy_backend(ETA0)
        for r in RS:
            be1 = make_noisy_backend(ETA0 * (1 + r))
            fp = {o: 0 for o in ORACLES}; fps = dict(fp); n = 0; recal = 0
            for _ in range(a.windows):
                cal = ex.counts_many(qc, be0, SHOTS, 12); ref1 = ex.counts(qc, be0, SHOTS)
                O = build(qc, cal, ref1, rng, a.B)
                sent = ex.counts(qc, be1, SHOTS)
                if jsd(O["nao"][0], sent) > O["nao"][1]:
                    recal += 1
                    cal2 = ex.counts_many(qc, be1, SHOTS, 12); ref2 = ex.counts(qc, be1, SHOTS)
                    O2 = build(qc, cal2, ref2, rng, a.B)
                else:
                    O2 = O
                for _ in range(a.tests):
                    c = ex.counts(qc, be1, SHOTS); n += 1
                    for o in ORACLES:
                        fp[o] += jsd(O[o][0], c) > O[o][1]
                        fps[o] += jsd(O2[o][0], c) > O2[o][1]
            A.append({"benchmark": b, "r": r, "tests": n, "fpr": {o: v / n for o, v in fp.items()},
                      "fpr_with_sentinel": {o: v / n for o, v in fps.items()}, "recalibration_rate": recal / a.windows})
            print(b, r, A[-1]["fpr"], A[-1]["fpr_with_sentinel"], flush=True)
        for r in RS:
            det = 0
            for _ in range(a.windows):
                cal = [ex.counts(qc, make_noisy_backend(ETA0 * (1 + r * i / 7)), SHOTS) for i in range(8)]
                det += compute_threshold(cal, SHOTS, ALPHA, "pooled", n_qubits=qc.num_qubits, B=1000, rng=rng).drift_detected
            Bw.append({"benchmark": b, "r": r, "detection_rate": det / a.windows, "ci": list(wilson(det, a.windows))})
    save({"between_window": A, "within_window": Bw}, os.path.join(d, "results.json"))
    cost = [{"M_tests": M, "NAO-1": 1 + M, "NAO (K=8)": 8 + M, "NAO (K=8) + two-run sentinel every 50 tests": 8 + M + 2 * int(np.ceil(M / 50)),
             "SPC envelope (K=12)": 13 + M, "QOIN-style (K_c=5)": 6 + M, "fixed threshold": M} for M in (10, 100, 1000)]
    attribution_cost = {"zne_linear (s = 1, 3)": 4, "zne_quadratic (s = 1, 3, 5)": 6, "reexecution_triage (simulator)": 6}
    save({"campaign_executions": cost, "attribution_executions_per_case": attribution_cost}, os.path.join(d, "cost_model.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ5 DRIFT AND COST (eta0={ETA0}, alpha={ALPHA})\n\n(a) FPR at eta0(1+r) after calibration at eta0\n")
        for x in A:
            f.write(f"  {x['benchmark']:7s} r={x['r']:<5} no sentinel " + " ".join(f"{o}={v:.3f}" for o, v in x["fpr"].items())
                    + "   sentinel " + " ".join(f"{o}={v:.3f}" for o, v in x["fpr_with_sentinel"].items()) + f"   recal={x['recalibration_rate']:.2f}\n")
        f.write("\n(b) drift detection inside the calibration window (K = 8)\n")
        for x in Bw:
            f.write(f"  {x['benchmark']:7s} r={x['r']:<5} detected={x['detection_rate']:.2f}\n")
        f.write("\n(c) executions per campaign\n")
        for c in cost:
            f.write("  " + "  ".join(f"{k}={v}" for k, v in c.items()) + "\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "eta0": ETA0, "alpha": ALPHA, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
