# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ4 (fault detection, part 4): real bugs from Bugs4Q.

Each real bug (nisq_framework/real_bugs.py) is a pair (fixed, buggy). In the
differential setting the fixed program is the trusted reference P and the
buggy program the program under test P'. Exact labels: the bug is
observable if the ideal outputs differ on |0...0> (and over four inputs).

For each bug, environment and window: every oracle builds its reference from
P, then judges 3 runs of P (negatives) and one run of P' (positive).
Environments: 5 synthetic noise levels and two device snapshots
(FakeManilaV2, FakeLagosV2), 10 windows each.
Rejected buggy programs are attributed by linear ZNE (bug if r0 exceeds the
null quantile).
Output: results/rq4_real_bugs/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from qiskit_aer import AerSimulator
from common import NOISE, SHOTS, out_dir, save, write_meta
from nisq_framework import (Executor, compute_threshold, spc_envelope_threshold, jsd, pooled, make_noisy_backend, ideal_jsd)
from nisq_framework.real_bugs import real_bug_pairs
from nisq_framework.attribution import attribute_zne
from nisq_framework.semantics import EQUIV_TOL
from nisq_framework.stats import confusion

ALPHA = 1e-3
FIXED = [0.01, 0.05, 0.1]
ORACLES = ["nao", "nao1", "spc_envelope"] + [f"fixed_{t:g}" for t in FIXED]


def inputs4(qc, rng):
    return ("",) + tuple(format(int(x), f"0{qc.num_qubits}b") for x in rng.integers(1, 2 ** qc.num_qubits, 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--B", type=int, default=3000); ap.add_argument("--seed", type=int, default=2718)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq4_real_bugs", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed); ideal_b = AerSimulator()
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2, FakeManilaV2
    envs = [(nl, make_noisy_backend(lv)) for nl, lv in NOISE.items()] + [("fake_manila", FakeManilaV2()), ("fake_lagos", FakeLagosV2())]
    bugs = real_bug_pairs(); catalogue, recs = [], []
    for bug in bugs:
        P, T = bug["fixed"], bug["buggy"]
        d0 = ideal_jsd(P, T); dm = ideal_jsd(P, T, inputs4(P, np.random.default_rng(5)))
        observable = d0 >= EQUIV_TOL
        catalogue.append({k: bug[k] for k in ("id", "path", "source", "kind")} | {"qubits": P.num_qubits, "ideal_jsd": float(d0),
                         "observable_zero_input": bool(observable), "observable_4_inputs": bool(dm >= EQUIV_TOL)})
        for env, be in envs:
            for w in range(a.windows):
                runs = ex.counts_many(P, be, SHOTS, 13); idl = ex.counts(P, ideal_b, SHOTS); nq = P.num_qubits
                O = {"nao": (pooled(runs[:8]), compute_threshold(runs[:8], SHOTS, ALPHA, "pooled", n_qubits=nq, B=a.B, rng=rng).tau),
                     "nao1": (runs[12], compute_threshold([], SHOTS, ALPHA, "single", n_qubits=nq, B=a.B, reference_counts=runs[12], rng=rng).tau),
                     "spc_envelope": (runs[12], spc_envelope_threshold(runs[:12]))}
                for t in FIXED:
                    O[f"fixed_{t:g}"] = (idl, t)
                judge = lambda c: {o: int(jsd(ref, c) > tau) for o, (ref, tau) in O.items()}
                for _ in range(3):
                    recs.append({"bug": bug["id"], "env": env, "window": w, "program": "fixed", "positive": False, **judge(ex.counts(P, be, SHOTS))})
                v = judge(ex.counts(T, be, SHOTS, layout_from=P))
                rec = {"bug": bug["id"], "env": env, "window": w, "program": "buggy", "positive": bool(observable), **v}
                if v["nao"] and w < 3:
                    att = attribute_zne(P, T, be, ex, SHOTS, scales=(1, 3), method="linear", alpha=1e-2, B=1000, seed=int(rng.integers(1 << 30)))
                    rec["attribution"] = att["attribution"]
                recs.append(rec)
            print(bug["id"], env, flush=True)
    save(catalogue, os.path.join(d, "catalogue.json")); save(recs, os.path.join(d, "records.json"))
    summ = {"overall": {}, "by_bug": {}, "by_env": {}}
    for o in ORACLES:
        ev = [r for r in recs if r["program"] == "fixed" or r["positive"]]
        cm = confusion([r["positive"] for r in ev], [bool(r[o]) for r in ev])
        summ["overall"][o] = {k: cm[k] for k in ("tp", "fn", "fp", "tn", "recall", "precision", "f1", "fpr")}
        summ["overall"][o]["unobservable_rejected"] = [int(sum(r[o] for r in recs if r["program"] == "buggy" and not r["positive"])),
                                                      sum(r["program"] == "buggy" and not r["positive"] for r in recs)]
        summ["by_env"][o] = {e: {"recall": float(np.mean([r[o] for r in recs if r["env"] == e and r["positive"]])),
                                 "fpr": float(np.mean([r[o] for r in recs if r["env"] == e and r["program"] == "fixed"]))} for e, _ in envs}
    for c in catalogue:
        S = [r for r in recs if r["bug"] == c["id"] and r["program"] == "buggy"]
        summ["by_bug"][c["id"]] = {o: float(np.mean([r[o] for r in S])) for o in ORACLES}
        att = [r["attribution"] for r in S if "attribution" in r]
        summ["by_bug"][c["id"]]["attributed_bug"] = [att.count("bug"), len(att)]
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ4 REAL BUGS (Bugs4Q): {len(catalogue)} bugs, {sum(c['observable_zero_input'] for c in catalogue)} observable on |0...0>\n\nCatalogue\n")
        for c in catalogue:
            f.write(f"  {c['id']:11s} {c['kind']:20s} q={c['qubits']} ideal JSD={c['ideal_jsd']:.3f} observable={c['observable_zero_input']} (4 inputs: {c['observable_4_inputs']})  {c['path']}\n")
        f.write("\nOverall (positives: observable bugs; negatives: runs of the fixed program)\n")
        for o, x in summ["overall"].items():
            f.write(f"  {o:13s} recall={x['recall']:.3f} precision={x['precision']:.3f} F1={x['f1']:.3f} FPR={x['fpr']:.4f} unobservable rejected={x['unobservable_rejected']}\n")
        f.write("\nRecall / FPR by environment\n")
        for o, v in summ["by_env"].items():
            f.write(f"  {o:13s} " + "  ".join(f"{e}={x['recall']:.2f}/{x['fpr']:.3f}" for e, x in v.items()) + "\n")
        f.write("\nDetection rate by bug (NAO, SPC, fixed 0.05) and ZNE attribution of NAO rejections\n")
        for b, v in summ["by_bug"].items():
            f.write(f"  {b:11s} NAO={v['nao']:.2f} SPC={v['spc_envelope']:.2f} fixed0.05={v['fixed_0.05']:.2f} attributed bug {v['attributed_bug'][0]}/{v['attributed_bug'][1]}\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha": ALPHA, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
