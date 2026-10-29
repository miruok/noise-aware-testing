# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
External validity: replication of RQ1, RQ2 and RQ4 on an extended suite of
32 QASMBench programs (all static programs of the QASMBench "small" category
with at most 10 qubits; benchmarks/qasmbench/).

--part detection (RQ1 and RQ4): 5 noise levels x 10 calibration windows per
    program; in each window every oracle builds its reference and threshold
    and judges 5 correct runs and one run of every applicable injected fault
    (paired). Oracles: NAO (K = 8, alpha = 1e-3), NAO-1, SPC envelope, fixed
    thresholds vs ideal reference, QOIN-style filter, and a noise-calibrated
    global threshold (after Fortz et al.), whose thresholds are computed per
    noise level from 30 separate runs of every program.
--part mutation (RQ2): low, medium, high noise; 15 mutants per family per
    program (seed 42); one calibration per cell; NAO and SPC envelope.
--part thresholds: global thresholds (run first); --part detection runs in
    --chunks chunks (programs QASMBENCH[chunk::chunks]).
--part summary: merge all parts into summary.json and REPORT.txt.
Output: results/ext_suite/.
"""
from __future__ import annotations
import argparse, itertools, json, os, time
import numpy as np
from qiskit_aer import AerSimulator
from common import NOISE, SHOTS, RESULTS, out_dir, save, write_meta
from nisq_framework import (Executor, MutationEngine, FAMILIES, compute_threshold, spc_envelope_threshold, jsd, pooled,
                            inject_all, make_noisy_backend, ideal_distribution)
from nisq_framework.benchmarks import QASMBENCH, load_qasmbench
from nisq_framework.stats import confusion, wilson, cluster_bootstrap, cmh_test, signflip_test, cliffs_delta
from scipy import stats

ALPHA = 1e-3
FIXED = [0.01, 0.05, 0.1]
NOISE3 = {"low": 0.001, "medium": 0.005, "high": 0.02}
ORACLES = ["nao", "nao1", "spc_envelope"] + [f"fixed_{t:g}" for t in FIXED] + ["qoin_style", "global_noise_calibrated"]
CLASSES = ["semantic", "noise_amplifying", "noise_neutral"]


def js_distance(p, q):
    return float(np.sqrt(jsd(p, q)))


def global_thresholds(circ, ideal, ex, R=30, Q=87.5):
    th = {}
    for nl, lv in NOISE.items():
        be = make_noisy_backend(lv); mus, ses = [], []
        for b, qc in circ.items():
            ds = [js_distance(ideal[b], c) for c in ex.counts_many(qc, be, SHOTS, R)]
            mus.append(np.mean(ds)); ses.append(np.std(ds, ddof=1) / np.sqrt(R))
        th[nl] = float(np.percentile(mus, Q) + np.percentile(ses, Q))
        print("global threshold", nl, round(th[nl], 4), flush=True)
    return th


def part_detection(a, d):
    t0 = time.time()
    ex = Executor(seed=a.seed + 100 * a.chunk); rng = np.random.default_rng(a.seed + 100 * a.chunk); ideal_b = AerSimulator()
    progs = QASMBENCH[a.chunk::a.chunks]
    circ = {b: load_qasmbench(b) for b in progs}
    ideal = {b: ideal_distribution(qc) for b, qc in circ.items()}
    gth = json.load(open(os.path.join(d, "global_thresholds.json")))
    recs = []
    for b, qc in circ.items():
        inj = [r for r in inject_all(qc) if r["applicable"]]
        nq = qc.num_qubits
        for nl, lv in NOISE.items():
            be = make_noisy_backend(lv)
            for w in range(a.windows):
                runs = ex.counts_many(qc, be, SHOTS, 13); idl = ex.counts(qc, ideal_b, SHOTS)
                O = {"nao": (pooled(runs[:8]), compute_threshold(runs[:8], SHOTS, ALPHA, "pooled", n_qubits=nq, B=a.B, rng=rng).tau),
                     "nao1": (runs[12], compute_threshold([], SHOTS, ALPHA, "single", n_qubits=nq, B=a.B, reference_counts=runs[12], rng=rng).tau),
                     "spc_envelope": (runs[12], spc_envelope_threshold(runs[:12]))}
                for t in FIXED:
                    O[f"fixed_{t:g}"] = (idl, t)
                n_id = sum(idl.values()); pid = {k: v / n_id for k, v in idl.items()}
                keys = set(pid).union(*[set(r) for r in runs[:5]])
                avg = {k: np.mean([r.get(k, 0) / SHOTS for r in runs[:5]]) for k in keys}
                fac = {k: (avg[k] + 1e-3) / (pid.get(k, 0) + 1e-3) for k in keys}

                def judge(c):
                    out = {o: int(jsd(ref, c) > tau) for o, (ref, tau) in O.items()}
                    corr = {k: (v / SHOTS) / max(fac.get(k, 1.0), 1e-6) for k, v in c.items()}
                    s = sum(corr.values()); corr = {k: v / s for k, v in corr.items()}
                    out["qoin_style"] = int(jsd(pid, corr) > 0.05)
                    out["global_noise_calibrated"] = int(js_distance(ideal[b], c) > gth[nl])
                    return out
                for _ in range(5):
                    recs.append({"benchmark": b, "noise": nl, "window": w, "kind": "correct", "fault": False, "equivalent_injection": False,
                                 **judge(ex.counts(qc, be, SHOTS))})
                for r in inj:
                    recs.append({"benchmark": b, "noise": nl, "window": w, "kind": r["injector"], "fault": not r["output_equivalent"],
                                 "equivalent_injection": bool(r["output_equivalent"]),
                                 **judge(ex.counts(r["circuit"], be, SHOTS, layout_from=qc))})
            print(b, nl, flush=True)
    save({"records": recs, "global_thresholds": gth}, os.path.join(d, f"detection_records_{a.chunk}.json"))
    os.makedirs(os.path.join(d, f"meta_detection_{a.chunk}"), exist_ok=True)
    write_meta(os.path.join(d, f"meta_detection_{a.chunk}"), vars(a) | {"part": "detection", "shots": SHOTS, "alpha": ALPHA, "executions": ex.n_executions}, t0)


def part_thresholds(a, d):
    t0 = time.time()
    ex = Executor(seed=a.seed + 7)
    circ = {b: load_qasmbench(b) for b in QASMBENCH}
    ideal = {b: ideal_distribution(qc) for b, qc in circ.items()}
    save(global_thresholds(circ, ideal, ex), os.path.join(d, "global_thresholds.json"))
    os.makedirs(os.path.join(d, "meta_thresholds"), exist_ok=True)
    write_meta(os.path.join(d, "meta_thresholds"), vars(a) | {"part": "thresholds", "shots": SHOTS, "executions": ex.n_executions}, t0)


def part_mutation(a, d):
    t0 = time.time()
    ex = Executor(seed=a.seed + 1); rng = np.random.default_rng(a.seed + 1)
    recs = []
    for b in QASMBENCH:
        qc = load_qasmbench(b); nq = qc.num_qubits
        eng = MutationEngine(qc, seed=42)
        muts = {f: eng.generate(15, f) for f in FAMILIES}
        for nl, lv in NOISE3.items():
            be = make_noisy_backend(lv)
            runs = ex.counts_many(qc, be, SHOTS, 13)
            tau = {"nao": compute_threshold(runs[:8], SHOTS, ALPHA, "pooled", n_qubits=nq, B=a.B, rng=rng).tau,
                   "spc_envelope": spc_envelope_threshold(runs[:12])}
            refs = {"nao": pooled(runs[:8]), "spc_envelope": runs[12]}
            orig = ex.counts(qc, be, SHOTS)
            op = {o: jsd(refs[o], orig) <= tau[o] for o in tau}
            for fam, ms in muts.items():
                for m in ms:
                    c = ex.counts(m.circuit, be, SHOTS, layout_from=qc)
                    rec = {"benchmark": b, "noise": nl, **{k: v for k, v in m.record().items()}}
                    for o in tau:
                        dv = jsd(refs[o], c)
                        rec[f"div_{o}"] = dv; rec[f"tau_{o}"] = tau[o]; rec[f"original_passed_{o}"] = bool(op[o])
                        rec[f"rejected_{o}"] = bool(op[o] and dv > tau[o])
                    recs.append(rec)
            print("mutation", b, nl, flush=True)
    save(recs, os.path.join(d, "mutation_records.json"))
    os.makedirs(os.path.join(d, "meta_mutation"), exist_ok=True)
    write_meta(os.path.join(d, "meta_mutation"), vars(a) | {"part": "mutation", "shots": SHOTS, "alpha": ALPHA, "executions": ex.n_executions}, t0)


def part_summary(d):
    import glob
    recs = []
    for fp in sorted(glob.glob(os.path.join(d, "detection_records_*.json"))):
        recs += json.load(open(fp))["records"]
    gth = json.load(open(os.path.join(d, "global_thresholds.json")))
    summ = {"programs": len({r["benchmark"] for r in recs}), "global_thresholds": gth, "soundness": {}, "detection": {}, "by_noise": {}}
    for o in ORACLES:
        cl = {}
        for r in recs:
            if r["kind"] == "correct":
                cl.setdefault((r["benchmark"], r["noise"], r["window"]), []).append(r[o])
        vals = [v for c in cl.values() for v in c]
        lo, hi = cluster_bootstrap(list(cl.values()), B=2000, seed=1)["ci"]
        fp_prog = {}
        for (b, _, _), v in cl.items():
            fp_prog.setdefault(b, []).extend(v)
        summ["soundness"][o] = {"fp": int(sum(vals)), "tests": len(vals), "fpr": float(np.mean(vals)), "ci": [lo, hi],
                                "programs_with_fpr_above_0.05": sum(np.mean(v) > 0.05 for v in fp_prog.values())}
        ev = [r for r in recs if r["kind"] == "correct" or r["fault"]]
        cm = confusion([r["fault"] for r in ev], [bool(r[o]) for r in ev])
        summ["detection"][o] = {k: cm[k] for k in ("tp", "fn", "fp", "tn", "recall", "precision", "f1", "fpr")}
        summ["detection"][o]["equivalent_rejected"] = [int(sum(r[o] for r in recs if r["equivalent_injection"])), sum(r["equivalent_injection"] for r in recs)]
        summ["by_noise"][o] = {nl: {"fpr": float(np.mean([r[o] for r in recs if r["noise"] == nl and r["kind"] == "correct"])),
                                    "recall": float(np.mean([r[o] for r in recs if r["noise"] == nl and r["fault"]]))} for nl in NOISE}
    summ["n_faults"] = len({(r["benchmark"], r["kind"]) for r in recs if r["fault"]})
    summ["n_equivalent_faults"] = len({(r["benchmark"], r["kind"]) for r in recs if r["equivalent_injection"]})
    mp = os.path.join(d, "mutation_records.json")
    if os.path.exists(mp):
        M = json.load(open(mp))
        summ["mutation"] = {}
        for o in ("nao", "spc_envelope"):
            # cells in which the original program fails the oracle are excluded (Definition of NAMS)
            Mo = [r for r in M if r[f"original_passed_{o}"]]
            summ["mutation"][o] = {c: {"k": int(sum(r[f"rejected_{o}"] for r in Mo if r["taxonomy_class"] == c)),
                                       "n": sum(r["taxonomy_class"] == c for r in Mo)} for c in CLASSES}
        summ["mutation"]["original_failed_cells_nao"] = len({(r["benchmark"], r["noise"]) for r in M if not r["original_passed_nao"]})
        tests = {}
        for scope in ["pooled"] + list(NOISE3):
            S = [r for r in M if scope == "pooled" or r["noise"] == scope]
            for a_, b_ in itertools.combinations(CLASSES, 2):
                tabs, diffs = [], []
                for cell in sorted({(r["benchmark"], r["noise"]) for r in S}):
                    C = [r for r in S if (r["benchmark"], r["noise"]) == cell and r["original_passed_nao"]]
                    ka = sum(r["rejected_nao"] for r in C if r["taxonomy_class"] == a_); na = sum(r["taxonomy_class"] == a_ for r in C)
                    kb = sum(r["rejected_nao"] for r in C if r["taxonomy_class"] == b_); nb = sum(r["taxonomy_class"] == b_ for r in C)
                    if na and nb:
                        tabs.append(((ka, na - ka), (kb, nb - kb))); diffs.append(ka / na - kb / nb)
                diffs = np.array(diffs)
                xa = [r["div_nao"] / r["tau_nao"] for r in S if r["taxonomy_class"] == a_]
                xb = [r["div_nao"] / r["tau_nao"] for r in S if r["taxonomy_class"] == b_]
                tests[f"{scope}|{a_}_vs_{b_}"] = {"cmh_p": cmh_test(tabs), "signflip_p": signflip_test(diffs), "cells": len(diffs),
                                                   "higher": int((diffs > 0).sum()), "equal": int((diffs == 0).sum()), "lower": int((diffs < 0).sum()),
                                                   "rate_a": float(np.mean([r["rejected_nao"] for r in S if r["taxonomy_class"] == a_])),
                                                   "rate_b": float(np.mean([r["rejected_nao"] for r in S if r["taxonomy_class"] == b_])),
                                                   "cliffs_delta": cliffs_delta(xa, xb)}
        summ["taxonomy_tests"] = tests
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"EXTENDED SUITE: {summ['programs']} QASMBench programs, {summ['n_faults']} non-equivalent faults, {summ['n_equivalent_faults']} equivalent injections\n\n")
        f.write("Global thresholds: " + "  ".join(f"{k}={v:.4f}" for k, v in gth.items()) + "\n\nSoundness and detection\n")
        for o in ORACLES:
            s, x = summ["soundness"][o], summ["detection"][o]
            f.write(f"  {o:24s} FPR={s['fpr']:.4f} ({s['fp']}/{s['tests']}) CI={s['ci']} programs FPR>0.05: {s['programs_with_fpr_above_0.05']}  "
                    f"recall={x['recall']:.3f} precision={x['precision']:.3f} F1={x['f1']:.3f} equivalent rejected={x['equivalent_rejected']}\n")
        f.write("\nBy noise (FPR / recall)\n")
        for o in ORACLES:
            f.write(f"  {o:24s} " + "  ".join(f"{nl}={v['fpr']:.3f}/{v['recall']:.3f}" for nl, v in summ["by_noise"][o].items()) + "\n")
        if "mutation" in summ:
            f.write("\nMutation (rejected/n)\n")
            for o in ("nao", "spc_envelope"):
                f.write(f"  {o:13s} " + "  ".join(f"{c}={v['k']}/{v['n']}={v['k']/max(v['n'],1):.3f}" for c, v in summ["mutation"][o].items()) + "\n")
            f.write(f"  cells where the original failed NAO: {summ['mutation']['original_failed_cells_nao']}\n\nTaxonomy tests (NAO)\n")
            for k, v in summ["taxonomy_tests"].items():
                f.write(f"  {k:52s} rates {v['rate_a']:.3f}/{v['rate_b']:.3f} CMH p={v['cmh_p']:.2e} cells {v['higher']}/{v['equal']}/{v['lower']} sign-flip p={v['signflip_p']:.2e} delta={v['cliffs_delta']:+.2f}\n")
    print(open(os.path.join(d, "REPORT.txt")).read())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", choices=["thresholds", "detection", "mutation", "summary"], required=True)
    ap.add_argument("--windows", type=int, default=10); ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=314); ap.add_argument("--out")
    ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--chunks", type=int, default=1); a = ap.parse_args()
    # published results are never overwritten: reruns of all parts share results/reruns/ext_suite/
    pub = os.path.join(RESULTS, "ext_suite")
    d = a.out or (os.path.join(RESULTS, "reruns", "ext_suite") if os.path.exists(os.path.join(pub, "summary.json")) else pub)
    os.makedirs(d, exist_ok=True)
    {"thresholds": lambda: part_thresholds(a, d), "detection": lambda: part_detection(a, d), "mutation": lambda: part_mutation(a, d), "summary": lambda: part_summary(d)}[a.part]()


if __name__ == "__main__":
    main()
