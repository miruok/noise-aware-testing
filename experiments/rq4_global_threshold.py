# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ4 (fault detection, part 3): comparison with a noise-model-calibrated global
threshold in the style of Fortz et al. (arXiv:2605.13279).

Baseline ("noise-calibrated threshold"): for each noise model, every correct
benchmark is executed R = 30 times; for each run the Jensen-Shannon distance
(square root of the base-2 divergence) to the exact ideal output distribution
is computed; per program, the mean mu_i and standard error s_i / sqrt(R) are
taken; the threshold is T = Q_87.5(mu_i) + Q_87.5(SE_i) over programs. A run
is rejected if its JS distance to the exact ideal distribution exceeds T.
The baseline therefore requires a noise model of the device and the ideal
output, and uses one threshold per noise model for all programs.

The threshold is calibrated on executions separate from the evaluation. The
evaluation uses the design of rq4_faults.py (10 windows per cell; per window
3 correct runs and one run of every applicable injected fault), and NAO
(K = 8, alpha = 1e-3) judges the same executions (paired comparison).
Output: results/rq4_global_threshold/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from common import BENCH, NOISE, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, compute_threshold, jsd, pooled, inject_all, make_noisy_backend, ideal_distribution
from nisq_framework.stats import confusion

ALPHA = 1e-3
R_CAL = 30
Q = 87.5


def js_distance(p, q):
    return float(np.sqrt(jsd(p, q)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--B", type=int, default=3000); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq4_global_threshold", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
    circ = {b: get_benchmark(b)[0] for b in BENCH}
    ideal = {b: ideal_distribution(qc) for b, qc in circ.items()}
    thresholds = {}
    for nl, lv in NOISE.items():
        be = make_noisy_backend(lv); mus, ses = [], []
        for b, qc in circ.items():
            ds = [js_distance(ideal[b], c) for c in ex.counts_many(qc, be, SHOTS, R_CAL)]
            mus.append(np.mean(ds)); ses.append(np.std(ds, ddof=1) / np.sqrt(R_CAL))
        thresholds[nl] = float(np.percentile(mus, Q) + np.percentile(ses, Q))
        print("threshold", nl, round(thresholds[nl], 4), flush=True)
    recs = []
    for b, qc in circ.items():
        inj = [r for r in inject_all(qc) if r["applicable"]]
        for nl, lv in NOISE.items():
            be = make_noisy_backend(lv)
            for w in range(a.windows):
                cal = ex.counts_many(qc, be, SHOTS, 8); ref = pooled(cal)
                tau = compute_threshold(cal, SHOTS, ALPHA, "pooled", n_qubits=qc.num_qubits, B=a.B, rng=rng).tau
                tests = [("correct", False, None, ex.counts(qc, be, SHOTS)) for _ in range(3)]
                tests += [(r["injector"], not r["output_equivalent"], r["output_equivalent"], ex.counts(r["circuit"], be, SHOTS, layout_from=qc)) for r in inj]
                for kind, fault, equiv, c in tests:
                    recs.append({"benchmark": b, "noise": nl, "window": w, "kind": kind, "fault": fault,
                                 "equivalent_injection": bool(equiv) if equiv is not None else False,
                                 "nao": int(jsd(ref, c) > tau), "noise_calibrated": int(js_distance(ideal[b], c) > thresholds[nl])})
            print(b, nl, flush=True)
    save(recs, os.path.join(d, "records.json"))
    summ = {"thresholds": thresholds, "overall": {}, "by_noise": {}, "by_benchmark_fpr": {}}
    ev = [r for r in recs if r["kind"] == "correct" or r["fault"]]
    for o in ("nao", "noise_calibrated"):
        cm = confusion([r["fault"] for r in ev], [bool(r[o]) for r in ev])
        summ["overall"][o] = {k: cm[k] for k in ("tp", "fn", "fp", "tn", "recall", "precision", "f1", "fpr")}
        summ["overall"][o]["equivalent_rejected"] = [sum(r[o] for r in recs if r["equivalent_injection"]),
                                                    sum(r["equivalent_injection"] for r in recs)]
        summ["by_noise"][o] = {nl: {"fpr": float(np.mean([r[o] for r in recs if r["noise"] == nl and r["kind"] == "correct"])),
                                    "recall": float(np.mean([r[o] for r in recs if r["noise"] == nl and r["fault"]]))} for nl in NOISE}
        summ["by_benchmark_fpr"][o] = {b: float(np.mean([r[o] for r in recs if r["benchmark"] == b and r["kind"] == "correct"])) for b in BENCH}
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ4 NOISE-CALIBRATED GLOBAL THRESHOLD (after Fortz et al.) VS NAO, paired executions\n\n")
        f.write("Thresholds (JS distance to exact ideal): " + "  ".join(f"{k}={v:.4f}" for k, v in thresholds.items()) + "\n\n")
        for o, x in summ["overall"].items():
            f.write(f"  {o:17s} TP={x['tp']} FN={x['fn']} FP={x['fp']} TN={x['tn']} recall={x['recall']:.3f} precision={x['precision']:.3f} "
                    f"F1={x['f1']:.3f} FPR={x['fpr']:.4f} equivalent rejected={x['equivalent_rejected'][0]}/{x['equivalent_rejected'][1]}\n")
        f.write("\nBy noise level (FPR / recall)\n")
        for o, v in summ["by_noise"].items():
            f.write(f"  {o:17s} " + "  ".join(f"{nl}={x['fpr']:.3f}/{x['recall']:.3f}" for nl, x in v.items()) + "\n")
        f.write("\nFPR by benchmark\n")
        for o, v in summ["by_benchmark_fpr"].items():
            f.write(f"  {o:17s} " + "  ".join(f"{b}={x:.3f}" for b, x in v.items()) + "\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha": ALPHA, "r_cal": R_CAL, "percentile": Q, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
