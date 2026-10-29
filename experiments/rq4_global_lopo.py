# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ4 (baseline generalization): leave-one-program-out evaluation of the
noise-calibrated global threshold (after Fortz et al.).

The global threshold T = Q87.5(mu_i) + Q87.5(SE_i) is computed from the
programs it is later applied to. This driver measures how it performs on a
program that did not contribute to it: for every noise level and program b,
T_{-b} is computed from the calibration statistics of the other programs of
the suite and applied to the evaluation runs of b. The in-sample threshold
T (all programs) is applied to the same runs for comparison.

Per program and noise level: R = 30 calibration runs (JS distance to the exact
ideal output), and, on separate executions, 50 correct runs and 10 runs of
every applicable non-equivalent injected fault. The noise-aware oracle does
not appear here: it calibrates each program on its own runs and uses no
cross-program statistic.
Suites: core (nine benchmarks), qasmbench (32 programs).
Output: results/rq4_global_lopo/<suite>/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from common import BENCH, NOISE, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, jsd, inject_all, make_noisy_backend, ideal_distribution
from nisq_framework.benchmarks import QASMBENCH, load_qasmbench
from nisq_framework.stats import confusion

R_CAL, N_CORRECT, N_FAULT, Q = 30, 50, 10, 87.5


def js_distance(p, q):
    return float(np.sqrt(jsd(p, q)))


def summarise(recs, key):
    cm = confusion([r["fault"] for r in recs], [r[key] for r in recs])
    return {k: cm[k] for k in ("tp", "fn", "fp", "tn", "recall", "precision", "f1", "fpr")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--suite", choices=["core", "qasmbench"], required=True)
    ap.add_argument("--seed", type=int, default=909); ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir(os.path.join("rq4_global_lopo", a.suite), a.out)
    ex = Executor(seed=a.seed + (a.suite == "qasmbench"))
    circ = {b: get_benchmark(b)[0] for b in BENCH} if a.suite == "core" else {b: load_qasmbench(b) for b in QASMBENCH}
    ideal = {b: ideal_distribution(qc) for b, qc in circ.items()}
    faults = {b: [r for r in inject_all(qc) if r["applicable"] and not r["output_equivalent"]] for b, qc in circ.items()}
    calib, evals = {}, []
    for nl, lv in NOISE.items():
        be = make_noisy_backend(lv); calib[nl] = {}
        for b, qc in circ.items():
            ds = [js_distance(ideal[b], c) for c in ex.counts_many(qc, be, SHOTS, R_CAL)]
            calib[nl][b] = {"mu": float(np.mean(ds)), "se": float(np.std(ds, ddof=1) / np.sqrt(R_CAL))}
            for c in ex.counts_many(qc, be, SHOTS, N_CORRECT):
                evals.append({"benchmark": b, "noise": nl, "kind": "correct", "fault": False, "dist": js_distance(ideal[b], c)})
            for r in faults[b]:
                for c in ex.counts_many(r["circuit"], be, SHOTS, N_FAULT, layout_from=qc):
                    evals.append({"benchmark": b, "noise": nl, "kind": r["injector"], "fault": True, "dist": js_distance(ideal[b], c)})
        print(a.suite, nl, flush=True)
    thr = {}
    for nl in NOISE:
        progs = list(calib[nl])
        T = lambda ps: float(np.percentile([calib[nl][p]["mu"] for p in ps], Q) + np.percentile([calib[nl][p]["se"] for p in ps], Q))
        thr[nl] = {"in_sample": T(progs), "leave_one_out": {b: T([p for p in progs if p != b]) for b in progs}}
    for r in evals:
        t = thr[r["noise"]]
        r["in_sample"] = bool(r["dist"] > t["in_sample"]); r["leave_one_out"] = bool(r["dist"] > t["leave_one_out"][r["benchmark"]])
    summ = {"overall": {k: summarise(evals, k) for k in ("in_sample", "leave_one_out")},
            "by_noise": {nl: {k: summarise([r for r in evals if r["noise"] == nl], k) for k in ("in_sample", "leave_one_out")} for nl in NOISE},
            "unsound_programs": {k: sum(np.mean([r[k] for r in evals if r["benchmark"] == b and not r["fault"]]) > 0.05 for b in circ)
                                 for k in ("in_sample", "leave_one_out")},
            "programs": len(circ), "thresholds": thr}
    save({"calibration": calib, "records": evals}, os.path.join(d, "records.json"))
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ4 GLOBAL THRESHOLD, LEAVE-ONE-PROGRAM-OUT ({a.suite}, {len(circ)} programs)\n")
        for k, x in summ["overall"].items():
            f.write(f"  {k:13s} FPR={x['fpr']:.4f} recall={x['recall']:.3f} precision={x['precision']:.3f} F1={x['f1']:.3f}  "
                    f"programs with FPR > 0.05: {summ['unsound_programs'][k]}/{len(circ)}\n")
        f.write("\nBy noise level (FPR / recall)\n")
        for nl, v in summ["by_noise"].items():
            f.write(f"  {nl:9s} " + "  ".join(f"{k}={x['fpr']:.3f}/{x['recall']:.3f}" for k, x in v.items()) + "\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "r_cal": R_CAL, "n_correct": N_CORRECT, "n_fault": N_FAULT, "percentile": Q,
                             "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
