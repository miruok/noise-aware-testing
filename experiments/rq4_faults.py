# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ4 (fault detection, part 1): detection of injected faults, analytic power, and
shot budget.

(a) Power validation on synthetic distributions: empirical detection rate of
    the null-quantile threshold vs the noncentral chi-square prediction across
    fault sizes and support sizes d (numpy only).
(b) Recall, precision and F1 on injected faults: every applicable injector
    (8 kinds) on 9 benchmarks x 5 noise levels with exact labels. Equivalent
    injections are reported separately and excluded from recall. Each window
    builds fresh oracles and also judges 3 correct runs (matched negatives).
(c) Predicted vs observed recall per fault (fault size estimated from
    200,000-shot runs).
(d) Shots needed for 80% power and minimum detectable JSD at 4096 shots.
Output: results/rq4_faults/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from qiskit_aer import AerSimulator
from common import BENCH, NOISE, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import (Executor, compute_threshold, spc_envelope_threshold, jsd, pooled, inject_all,
                  detection_power, shots_needed, min_detectable_delta, make_noisy_backend)
from nisq_framework.divergence import jsd_arrays
from nisq_framework.threshold import bootstrap_null
from nisq_framework.stats import confusion, wilson

ALPHA = 1e-3
FIXED = [0.01, 0.02, 0.05, 0.10]
ORACLES = ["nao", "nao1", "spc_envelope"] + [f"fixed_{t:g}" for t in FIXED]


def part_a(rng, trials=300, N=SHOTS):
    rows = []
    for d in (4, 16, 64, 256):
        p = rng.dirichlet(np.ones(d))
        for eps in np.geomspace(2e-3, 0.4, 14):
            r = rng.dirichlet(np.ones(d)); q = (1 - eps) * p + eps * r
            delta = float(jsd_arrays(p, q, smooth=False)); det = 0
            for _ in range(trials):
                ref = rng.multinomial(N, p); test = rng.multinomial(N, q)
                ph = ref / ref.sum()
                tau = np.quantile(bootstrap_null(ph[ph > 0], N, N, 1000, rng), 1 - ALPHA, method="higher")
                det += jsd_arrays(ref, test) > tau
            rows.append({"d": d, "eps": float(eps), "delta": delta, "empirical": det / trials,
                         "empirical_ci": list(wilson(det, trials)), "predicted": detection_power(delta, d, N, N, ALPHA)})
    return rows


def build_oracles(qc, backend, ex, ideal_b, rng, B):
    runs = ex.counts_many(qc, backend, SHOTS, 13)
    ideal = ex.counts(qc, ideal_b, SHOTS)
    o = {}
    r = compute_threshold([], SHOTS, ALPHA, "single", n_qubits=qc.num_qubits, B=B, reference_counts=runs[12], rng=rng)
    o["nao1"] = (runs[12], r.tau)
    r = compute_threshold(runs[:8], SHOTS, ALPHA, "pooled", n_qubits=qc.num_qubits, B=B, rng=rng)
    o["nao"] = (pooled(runs[:8]), r.tau)
    o["spc_envelope"] = (runs[12], spc_envelope_threshold(runs[:12]))
    for t in FIXED:
        o[f"fixed_{t:g}"] = (ideal, t)
    return o


def judge(oracles, counts):
    return {name: int(jsd(ref, counts) > tau) for name, (ref, tau) in oracles.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=10); ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq4_faults", a.out); rng = np.random.default_rng(a.seed)
    ex = Executor(seed=a.seed); ideal_b = AerSimulator()
    if not a.skip_a:
        save(part_a(rng), os.path.join(d, "power_validation.json")); print("power validation done", flush=True)
    faults, records = [], []
    for b in BENCH:
        qc, _ = get_benchmark(b)
        inj = [r for r in inject_all(qc) if r["applicable"]]
        for r in inj:
            faults.append({"benchmark": b, **{k: v for k, v in r.items() if k != "circuit"}})
        real = [r for r in inj if not r["output_equivalent"]]
        for nl, lv in NOISE.items():
            backend = make_noisy_backend(lv)
            big_P = ex.counts(qc, backend, 200_000)
            sizes = {}
            for r in real:
                big_F = ex.counts(r["circuit"], backend, 200_000, layout_from=qc)
                sizes[r["injector"]] = (jsd(big_P, big_F), len(set(big_P) | set(big_F)))
            for w in range(a.windows):
                orc = build_oracles(qc, backend, ex, ideal_b, rng, a.B)
                for _ in range(3):
                    records.append({"benchmark": b, "noise": nl, "window": w, "kind": "correct", "fault": False,
                                    **judge(orc, ex.counts(qc, backend, SHOTS))})
                for r in inj:
                    c = ex.counts(r["circuit"], backend, SHOTS, layout_from=qc)
                    rec = {"benchmark": b, "noise": nl, "window": w, "kind": r["injector"], "fault": not r["output_equivalent"],
                           "unitary_changed_output_unchanged": r["basis_invisible"], "ideal_jsd": r["ideal_jsd"], **judge(orc, c)}
                    if rec["fault"]:
                        dn, deff = sizes[r["injector"]]
                        rec.update(noisy_delta=dn, d_eff=deff,
                                   pred_power_nao1=detection_power(dn, deff, SHOTS, SHOTS, ALPHA),
                                   pred_power_nao=detection_power(dn, deff, 8 * SHOTS, SHOTS, ALPHA))
                    records.append(rec)
            print(f"{b:8s} {nl:9s} done", flush=True)
    save(faults, os.path.join(d, "fault_catalogue.json"))
    save(records, os.path.join(d, "records.json"))
    evalset = [r for r in records if r["kind"] == "correct" or r["fault"]]
    summ = {"overall": {}, "by_noise": {}, "by_injector": {}, "equivalent_injections": {}}
    for o in ORACLES:
        summ["overall"][o] = confusion([r["fault"] for r in evalset], [r[o] for r in evalset])
        for nl in NOISE:
            s = [r for r in evalset if r["noise"] == nl]
            summ["by_noise"].setdefault(o, {})[nl] = confusion([r["fault"] for r in s], [r[o] for r in s])
        for inj in sorted({r["kind"] for r in records if r["kind"] != "correct"}):
            s = [r for r in records if r["kind"] == inj and r["fault"]]
            if s:
                summ["by_injector"].setdefault(o, {})[inj] = {"recall": float(np.mean([r[o] for r in s])), "n": len(s)}
        eqs = [r for r in records if r["kind"] != "correct" and not r["fault"]]
        summ["equivalent_injections"][o] = {"n": len(eqs), "rejected": int(sum(r[o] for r in eqs))}
    pc = [r for r in records if r.get("fault")]
    bins = np.linspace(0, 1, 11); cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        for mode in ("nao1", "nao"):
            s = [r for r in pc if lo <= r[f"pred_power_{mode}"] < hi + (hi == 1)]
            if s:
                cal.append({"oracle": mode, "bin": [lo, hi], "n": len(s),
                            "mean_predicted": float(np.mean([r[f"pred_power_{mode}"] for r in s])),
                            "observed": float(np.mean([r[mode] for r in s]))})
    summ["power_calibration"] = cal
    summ["shots_needed_80pct_alpha1e-3"] = [
        {"delta": dl, "d": dd, "nao1": shots_needed(dl, dd, ALPHA, 0.8, 1.0), "nao_K8": shots_needed(dl, dd, ALPHA, 0.8, 8.0)}
        for dl in (1e-3, 3e-3, 1e-2, 3e-2) for dd in (4, 16, 64, 256)]
    summ["min_detectable_delta_4096"] = {str(dd): {"nao1": min_detectable_delta(dd, SHOTS, SHOTS, ALPHA),
                                                   "nao_K8": min_detectable_delta(dd, 8 * SHOTS, SHOTS, ALPHA)}
                                         for dd in (4, 16, 64, 256)}
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ4 FAULT DETECTION\n\nConfusion (faults = non-equivalent injections; negatives = correct runs)\n")
        f.write(f"{'oracle':14s} {'TP':>5s} {'FN':>5s} {'FP':>5s} {'TN':>5s} {'recall':>8s} {'precision':>10s} {'F1':>7s} {'FPR':>8s}\n")
        for o, v in summ["overall"].items():
            f.write(f"{o:14s} {v['tp']:5d} {v['fn']:5d} {v['fp']:5d} {v['tn']:5d} {v['recall']:8.3f} {v['precision']:10.3f} {v['f1']:7.3f} {v['fpr']:8.4f}\n")
        f.write("\nRecall by noise level\n" + f"{'oracle':14s}" + "".join(f"{n:>11s}" for n in NOISE) + "\n")
        for o, v in summ["by_noise"].items():
            f.write(f"{o:14s}" + "".join(f"{v[n]['recall']:11.3f}" for n in NOISE) + "\n")
        f.write("\nRecall by injector\n")
        for o, v in summ["by_injector"].items():
            f.write(f"  {o:14s} " + "  ".join(f"{k}={x['recall']:.2f}(n={x['n']})" for k, x in v.items()) + "\n")
        f.write("\nEquivalent injections rejected\n")
        for o, v in summ["equivalent_injections"].items():
            f.write(f"  {o:14s} {v['rejected']}/{v['n']}\n")
        f.write("\nPredicted vs observed recall\n")
        for c in cal:
            f.write(f"  {c['oracle']:6s} bin {c['bin'][0]:.1f}-{c['bin'][1]:.1f} n={c['n']:4d} predicted={c['mean_predicted']:.3f} observed={c['observed']:.3f}\n")
        f.write("\nMinimum detectable JSD at 4096 shots, 80% power, alpha 1e-3\n")
        for k, v in summ["min_detectable_delta_4096"].items():
            f.write(f"  d={k:>4s} NAO-1={v['nao1']:.5f} NAO(K=8)={v['nao_K8']:.5f}\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha": ALPHA, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read()[:4000])


if __name__ == "__main__":
    main()
