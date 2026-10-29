# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ4 (fault detection, part 5): ablation of the noise-aware oracle.

One component is added at a time; all configurations judge the same
executions (paired design).
  A  ideal reference          + fixed tau = 0.02        (current practice)
  B  one self-baseline run    + fixed tau = 0.02        (adds: self-baseline reference)
  C  one self-baseline run    + SPC envelope threshold  (adds: data-driven heuristic threshold)
  D  one self-baseline run    + null-quantile threshold (adds: statistical threshold; = NAO-1)
  E  8 pooled runs            + null-quantile threshold, no drift test (adds: pooling)
  F  8 pooled runs            + null-quantile threshold + drift test   (adds: drift test; = NAO)

--part synthetic: 9 core benchmarks x 5 noise levels x 10 windows.
--part snapshots: 5 small benchmarks x 3 device snapshots x 10 windows under
    the random compilation policy (recompilation between runs), where the drift
    test is designed to matter.
Each window: 13 executions of P (runs 1-8 pooled reference; run 13 single
reference; runs 1-12 SPC calibration), one noiseless execution (ideal
reference), 5 correct runs and one run of every applicable injected fault.
--part summary merges both parts.
Output: results/rq4_ablation/.
"""
from __future__ import annotations
import argparse, glob, json, os, time
import numpy as np
from qiskit_aer import AerSimulator
from common import BENCH, NOISE, SHOTS, RESULTS, save, write_meta, get_benchmark
from nisq_framework import (Executor, compute_threshold, spc_envelope_threshold, jsd, pooled, inject_all, make_noisy_backend)
from nisq_framework.stats import confusion, cluster_bootstrap

ALPHA = 1e-3
TAU = 0.02
RUNGS = ["A", "B", "C", "D", "E", "F"]
NAMES = {"A": "ideal reference, fixed tau", "B": "self-baseline (1 run), fixed tau", "C": "self-baseline (1 run), SPC envelope",
         "D": "self-baseline (1 run), null quantile (NAO-1)", "E": "pooled (8 runs), null quantile, no drift test",
         "F": "pooled (8 runs), null quantile, drift test (NAO)"}
SMALL = ["bell", "ghz_3", "qft_3", "qaoa_3", "vqe_3"]


def window(qc, be, ex, rng, B, ideal_b):
    nq = qc.num_qubits
    runs = ex.counts_many(qc, be, SHOTS, 13); idl = ex.counts(qc, ideal_b, SHOTS)
    pooled_ref = pooled(runs[:8])
    rep = compute_threshold(runs[:8], SHOTS, ALPHA, "pooled", n_qubits=nq, B=B, rng=rng)
    tau_single = compute_threshold([], SHOTS, ALPHA, "single", n_qubits=nq, B=B, reference_counts=runs[12], rng=rng).tau
    O = {"A": (idl, TAU), "B": (runs[12], TAU), "C": (runs[12], spc_envelope_threshold(runs[:12])),
         "D": (runs[12], tau_single), "E": (pooled_ref, rep.tau_null), "F": (pooled_ref, rep.tau)}
    return O, bool(rep.drift_detected)


def run_env(env, be, benches, ex, rng, a, recs):
    ideal_b = AerSimulator()
    for b in benches:
        qc, _ = get_benchmark(b)
        inj = [r for r in inject_all(qc) if r["applicable"]]
        for w in range(a.windows):
            O, drift = window(qc, be, ex, rng, a.B, ideal_b)
            judge = lambda c: {k: int(jsd(ref, c) > tau) for k, (ref, tau) in O.items()}
            for _ in range(5):
                recs.append({"env": env, "benchmark": b, "window": w, "kind": "correct", "fault": False, "equivalent": False,
                             "drift": drift, **judge(ex.counts(qc, be, SHOTS))})
            for r in inj:
                recs.append({"env": env, "benchmark": b, "window": w, "kind": r["injector"], "fault": not r["output_equivalent"],
                             "equivalent": bool(r["output_equivalent"]), "drift": drift,
                             **judge(ex.counts(r["circuit"], be, SHOTS, layout_from=qc))})
        print(env, b, flush=True)


def summarize(d):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "records_*.json"))):
        recs += json.load(open(f))
    groups = {"synthetic": [r for r in recs if r["env"] in NOISE], "snapshots_random_compilation": [r for r in recs if r["env"] not in NOISE]}
    summ = {"rungs": NAMES, "results": {}}
    for g, R in groups.items():
        if not R:
            continue
        summ["results"][g] = {}
        for k in RUNGS:
            ev = [r for r in R if r["kind"] == "correct" or r["fault"]]
            cm = confusion([r["fault"] for r in ev], [bool(r[k]) for r in ev])
            cl = {}
            for r in R:
                if r["kind"] == "correct":
                    cl.setdefault((r["env"], r["benchmark"], r["window"]), []).append(r[k])
            cb = cluster_bootstrap(list(cl.values()), B=2000, seed=1)
            summ["results"][g][k] = {"fpr": cm["fpr"], "fpr_ci": cb["ci"], "fp": cm["fp"], "negatives": cm["fp"] + cm["tn"],
                                     "recall": cm["recall"], "precision": cm["precision"], "f1": cm["f1"],
                                     "equivalent_rejected": [int(sum(r[k] for r in R if r["equivalent"])), sum(r["equivalent"] for r in R)]}
        summ["results"][g]["n_correct"] = sum(r["kind"] == "correct" for r in R)
        summ["results"][g]["n_fault_tests"] = sum(r["fault"] for r in R)
        summ["results"][g]["drift_flag_rate"] = float(np.mean([r["drift"] for r in R if r["kind"] == "correct"]))
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ4 ABLATION OF THE NOISE-AWARE ORACLE (paired executions; alpha = 1e-3, fixed tau = 0.02)\n")
        for g, v in summ["results"].items():
            f.write(f"\n{g}: {v['n_correct']} correct runs, {v['n_fault_tests']} fault tests, drift flagged in {v['drift_flag_rate']:.2f} of windows\n")
            for k in RUNGS:
                x = v[k]
                f.write(f"  {k} {NAMES[k]:48s} FPR={x['fpr']:.4f} CI=[{x['fpr_ci'][0]:.4f}, {x['fpr_ci'][1]:.4f}] recall={x['recall']:.3f} "
                        f"precision={x['precision']:.3f} F1={x['f1']:.3f} equivalent rejected={x['equivalent_rejected'][0]}/{x['equivalent_rejected'][1]}\n")
    print(open(os.path.join(d, "REPORT.txt")).read())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", choices=["synthetic", "snapshots", "summary"], required=True)
    ap.add_argument("--windows", type=int, default=10); ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1618); ap.add_argument("--out"); a = ap.parse_args()
    pub = os.path.join(RESULTS, "rq4_ablation")
    d = a.out or (os.path.join(RESULTS, "reruns", "rq4_ablation") if os.path.exists(os.path.join(pub, "summary.json")) else pub)
    os.makedirs(d, exist_ok=True)
    if a.part == "summary":
        return summarize(d)
    t0 = time.time(); recs = []
    if a.part == "synthetic":
        ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
        for nl, lv in NOISE.items():
            run_env(nl, make_noisy_backend(lv), BENCH, ex, rng, a, recs)
    else:
        from qiskit_ibm_runtime.fake_provider import FakeKolkataV2, FakeLagosV2, FakeManilaV2
        ex = Executor(seed=a.seed + 1, layout_policy="random"); rng = np.random.default_rng(a.seed + 1)
        for be in (FakeManilaV2(), FakeLagosV2(), FakeKolkataV2()):
            run_env(be.name, be, SMALL, ex, rng, a, recs)
    save(recs, os.path.join(d, f"records_{a.part}.json"))
    os.makedirs(os.path.join(d, f"meta_{a.part}"), exist_ok=True)
    write_meta(os.path.join(d, f"meta_{a.part}"), vars(a) | {"shots": SHOTS, "alpha": ALPHA, "tau": TAU, "executions": ex.n_executions}, t0)


if __name__ == "__main__":
    main()
