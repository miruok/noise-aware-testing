# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Supplementary analyses of existing raw records (no new executions).

(a) RQ1, per program and per window: false-positive rate of NAO and NAO-1 for
    each core benchmark (1,000 correct runs per benchmark and level) with
    95% cluster-bootstrap intervals over calibration windows and Wilson
    intervals (the cluster bootstrap degenerates to [0, 0] without events), and the
    distribution of false positives per window (0, 1, 2 or more of five runs)
    against the binomial expectation for independent runs at the nominal level.
    Source: results/rq1_soundness/per_window.json.
(b) RQ3, by noise level: confusion matrices of linear and quadratic ZNE
    attribution on the cases that NAO rejects (positive class: Bug).
    Source: results/rq3_attribution/records.json.
(c) RQ4, power model by noise level and by number of occupied outcomes:
    mean predicted detection probability, observed detection rate and Brier
    score of the noncentral chi-square prediction for NAO on the
    non-equivalent injected faults.
    Source: results/rq4_faults/records.json.
(d) RQ2, effect sizes with uncertainty: Cliff's delta on D/tau for every pair of
    taxonomy classes, per noise level and pooled, with a 95% cluster-bootstrap
    interval that resamples cells (one cell shares one calibration window),
    for the core benchmarks and the QASMBench programs.
    Sources: results/rq2_mutation/records.json, results/ext_suite/mutation_records.json.
Output: results/supplementary/.
"""
from __future__ import annotations
import argparse, os, time
import json
import numpy as np
from scipy import stats as st
from common import RESULTS, BENCH, out_dir, save, write_meta
from nisq_framework.stats import cluster_bootstrap, wilson

LEVELS = ["very_low", "low", "medium", "high", "very_high"]


def part_a():
    W = json.load(open(os.path.join(RESULTS, "rq1_soundness", "per_window.json")))
    out = {"per_benchmark": {}, "per_window": {}}
    for key, alpha in (("nao_a0.001", 1e-3), ("nao1_a0.001", 1e-3), ("nao_a0.01", 1e-2), ("nao1_a0.01", 1e-2)):
        pb = {}
        for b in BENCH:
            clusters = [w["fp"][key] for c in W if c["benchmark"] == b for w in c["windows"]]
            cb = cluster_bootstrap(clusters, B=2000, seed=7)
            k = int(sum(map(sum, clusters))); n = cb["n_obs"]
            pb[b] = {"fp": k, "tests": n, "fpr": cb["estimate"], "ci": cb["ci"], "wilson": list(wilson(k, n)),
                     "binomial_p_two_sided": float(st.binomtest(k, n, alpha).pvalue)}
        out["per_benchmark"][key] = pb
        per = [sum(w["fp"][key]) for c in W for w in c["windows"]]
        nwin = len(per); m = len(W[0]["windows"][0]["fp"][key])
        hist = {"0": per.count(0), "1": per.count(1), "2+": sum(x >= 2 for x in per)}
        exp = {"0": nwin * st.binom.pmf(0, m, alpha), "1": nwin * st.binom.pmf(1, m, alpha),
               "2+": nwin * st.binom.sf(1, m, alpha)}
        out["per_window"][key] = {"windows": nwin, "runs_per_window": m, "observed": hist, "expected_independent": exp}
    return out


def part_b():
    R = json.load(open(os.path.join(RESULTS, "rq3_attribution", "records.json")))
    out = {}
    for method in ("zne_linear", "zne_quadratic"):
        out[method] = {}
        for nl in ("low", "medium", "high"):
            S = [r for r in R if r["triggered"] and r["noise"] == nl]
            tp = sum(r["truth"] == "bug" and r[method] == "bug" for r in S)
            fn = sum(r["truth"] == "bug" and r[method] == "noise" for r in S)
            fp = sum(r["truth"] == "noise" and r[method] == "bug" for r in S)
            tn = sum(r["truth"] == "noise" and r[method] == "noise" for r in S)
            out[method][nl] = {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "cases": len(S),
                               "bug_recall": tp / (tp + fn) if tp + fn else None,
                               "bug_precision": tp / (tp + fp) if tp + fp else None,
                               "noise_recall": tn / (tn + fp) if tn + fp else None}
    return out


def part_c():
    R = [r for r in json.load(open(os.path.join(RESULTS, "rq4_faults", "records.json"))) if r["fault"]]

    def summarise(S):
        p = np.array([r["pred_power_nao"] for r in S]); y = np.array([r["nao"] for r in S], float)
        return {"tests": len(S), "mean_predicted": float(p.mean()), "observed": float(y.mean()),
                "difference": float(y.mean() - p.mean()), "brier": float(np.mean((p - y) ** 2)),
                "brier_constant": float(np.mean((y.mean() - y) ** 2))}
    out = {"by_noise": {nl: summarise([r for r in R if r["noise"] == nl]) for nl in LEVELS}}
    bins = [(1, 16), (17, 64), (65, 128), (129, 256)]
    out["by_occupied_outcomes"] = {f"{lo}-{hi}": summarise([r for r in R if lo <= r["d_eff"] <= hi]) for lo, hi in bins
                                   if any(lo <= r["d_eff"] <= hi for r in R)}
    out["all"] = summarise(R)
    return out


def fast_delta(a, b):
    """Cliff's delta via the Mann-Whitney U statistic (ties count one half)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    r = st.rankdata(np.concatenate([a, b]))
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


def part_d(B=2000, seed=11):
    CL = ["semantic", "noise_amplifying", "noise_neutral"]
    srcs = {"core": os.path.join(RESULTS, "rq2_mutation", "records.json"),
            "qasmbench": os.path.join(RESULTS, "ext_suite", "mutation_records.json")}
    out = {}
    rng = np.random.default_rng(seed)
    for suite, path in srcs.items():
        R = [r for r in json.load(open(path)) if r["original_passed_nao"]]
        out[suite] = {}
        for scope in ("low", "medium", "high", "pooled"):
            S = R if scope == "pooled" else [r for r in R if r["noise"] == scope]
            cells = sorted({(r["benchmark"], r["noise"]) for r in S})
            by = {c: {k: np.array([r["div_nao"] / r["tau_nao"] for r in S if (r["benchmark"], r["noise"]) == c and r["taxonomy_class"] == k])
                      for k in CL} for c in cells}
            res = {}
            for i in range(3):
                for j in range(i + 1, 3):
                    a_, b_ = CL[i], CL[j]
                    est = fast_delta(np.concatenate([by[c][a_] for c in cells]), np.concatenate([by[c][b_] for c in cells]))
                    boots = []
                    for _ in range(B):
                        idx = rng.integers(0, len(cells), len(cells))
                        A = np.concatenate([by[cells[k]][a_] for k in idx]); Bv = np.concatenate([by[cells[k]][b_] for k in idx])
                        if len(A) and len(Bv):
                            boots.append(fast_delta(A, Bv))
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    res[f"{a_}_vs_{b_}"] = {"delta": est, "ci": [float(lo), float(hi)], "cells": len(cells)}
            out[suite][scope] = res
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("supplementary", a.out)
    A, B, C, Dd = part_a(), part_b(), part_c(), part_d()
    save({"rq1_per_program": A, "rq3_triggered_by_noise": B, "rq4_power_by_stratum": C, "rq2_effect_sizes": Dd}, os.path.join(d, "results.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("(a) RQ1 false-positive rate per core benchmark (95% cluster bootstrap over windows)\n")
        for key, pb in A["per_benchmark"].items():
            f.write(f"  {key}\n")
            for b, v in pb.items():
                f.write(f"    {b:7s} {v['fp']:3d}/{v['tests']}  {v['fpr']:.4f}  cluster [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]  Wilson [{v['wilson'][0]:.4f}, {v['wilson'][1]:.4f}]  binomial p={v['binomial_p_two_sided']:.3f}\n")
        f.write("\n    false positives per window (observed / expected for independent runs at the nominal level)\n")
        for key, v in A["per_window"].items():
            f.write(f"    {key:12s} windows={v['windows']}: " + "  ".join(f"{k}: {v['observed'][k]} / {v['expected_independent'][k]:.2f}" for k in ("0", "1", "2+")) + "\n")
        f.write("\n(b) RQ3 attribution on triggered cases by noise level (positive class: Bug)\n")
        for m, v in B.items():
            for nl, c in v.items():
                f.write(f"  {m:13s} {nl:6s} cases={c['cases']:3d} TP={c['TP']:3d} FN={c['FN']:3d} FP={c['FP']:3d} TN={c['TN']:3d}  "
                        f"bug recall {c['bug_recall']:.3f}  bug precision {c['bug_precision']:.3f}  noise recall {c['noise_recall']:.3f}\n")
        f.write("\n(c) RQ4 power model for NAO on non-equivalent faults\n")
        for grp in ("by_noise", "by_occupied_outcomes"):
            for k, v in C[grp].items():
                f.write(f"  {grp:20s} {k:9s} tests={v['tests']:4d} predicted {v['mean_predicted']:.3f} observed {v['observed']:.3f} "
                        f"diff {v['difference']:+.3f}  Brier {v['brier']:.3f} (constant {v['brier_constant']:.3f})\n")
        v = C["all"]
        f.write(f"  all                            tests={v['tests']:4d} predicted {v['mean_predicted']:.3f} observed {v['observed']:.3f} "
                f"diff {v['difference']:+.3f}  Brier {v['brier']:.3f} (constant {v['brier_constant']:.3f})\n")
        f.write("\n(d) RQ2 Cliff's delta on D/tau with 95% cluster-bootstrap intervals over cells\n")
        for suite, v in Dd.items():
            for scope, res in v.items():
                for k, x in res.items():
                    f.write(f"  {suite:9s} {scope:6s} {k:36s} delta={x['delta']:+.3f}  [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]  cells={x['cells']}\n")
    write_meta(d, vars(a), t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
