# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ1 (soundness, part 3): calibration of the null-quantile threshold when the
output support is large relative to the shot budget.

Pure sampling experiment, no circuits: a correct test run and its reference
are multinomial samples from the same distribution p, so every rejection is a
false positive. p ranges over supports of d = 8 to 16384 outcomes (shots per
outcome N/d from 512 to 0.25 at N = 4096) and two shapes:
  flat    p ~ Dirichlet(1, ..., 1): every outcome occupied, near-uniform spread
  skewed  Zipf, p_i proportional to i^-1.1: a few heavy outcomes and a long tail
  headtail four heavy outcomes carrying 80% of the mass (weights 1, 1/2, 1/3, 1/4)
          and a nearly uniform tail carrying the remaining 20%
For every window, the thresholds of NAO (K = 8 pooled runs, drift test) and
NAO-1 (one reference run) are computed exactly as in nisq_framework.threshold
(plug-in estimate with Chao1 / Good-Turing unseen correction, parametric
bootstrap), together with the closed-form chi-square threshold on the
observed support; five correct test runs are judged against each.
Per window, two diagnostics that a tester can compute from the reference counts
alone are recorded with the number of false positives: the Good-Turing unseen
mass f1/n (fraction of reference shots on outcomes observed exactly once) and
the shots per observed outcome (both for the pooled reference and for one
run of P). They do not consume random numbers, so the
false-positive counts are identical to runs without them.
Output: results/rq1_sparsity/.
"""
from __future__ import annotations
import argparse, itertools, os, time
import numpy as np
from common import RESULTS, SHOTS, save, write_meta
from nisq_framework.threshold import plugin_distribution, bootstrap_null, homogeneity_test, chi2_threshold
from nisq_framework.divergence import jsd_arrays

DS = [8, 64, 256, 1024, 4096, 16384]
ALPHAS = [1e-3, 1e-2, 5e-2]
K, TESTS, K_EMP, A_DRIFT = 8, 5, 4.0, 1e-2


def make_p(shape, d, rng):
    if shape == "flat":
        return rng.dirichlet(np.ones(d))
    if shape == "headtail":
        h = 4
        head = 1.0 / np.arange(1, h + 1); head = 0.8 * head / head.sum()
        tail = np.full(d - h, 0.2 / (d - h))
        return rng.permutation(np.concatenate([head, tail]))
    w = np.arange(1, d + 1, dtype=float) ** -1.1
    return rng.permutation(w / w.sum())


def as_counts(v):
    return {str(i): int(c) for i, c in enumerate(v) if c > 0}


def thresholds(ref_counts, n_ref, B, rng, cal=None):
    """Null quantiles at every alpha plus the drift-widened value (pooled mode)."""
    keys, p, added = plugin_distribution(ref_counts, n_qubits=None)
    null = bootstrap_null(p, n_ref, SHOTS, B, rng)
    tn = {a: float(np.quantile(null, 1 - a, method="higher")) for a in ALPHAS}
    tc = {a: chi2_threshold(len(keys), n_ref, SHOTS, a) for a in ALPHAS}
    drift, temp = False, None
    if cal is not None:
        hp = homogeneity_test([as_counts(c) for c in cal])
        iu = np.triu_indices(len(cal), 1)
        pair = jsd_arrays(cal[iu[0]], cal[iu[1]])
        temp = float(pair.mean() + K_EMP * pair.std(ddof=1)); drift = hp < A_DRIFT
    tau = {a: (max(tn[a], temp) if drift else tn[a]) for a in ALPHAS}
    return tau, tc, len(ref_counts), added, drift


def diagnostics(v):
    v = np.asarray(v); n = int(v.sum())
    return {"good_turing_unseen_mass": float((v == 1).sum() / n), "shots_per_observed_outcome": float(n / (v > 0).sum())}


def cell(shape, d, windows, B, rng, per_window=None):
    fp = {o: {a: 0 for a in ALPHAS} for o in ("nao", "nao1", "chi2_pooled", "chi2_single")}
    obs, unseen, drifts, occupied = [], [], 0, []
    for _ in range(windows):
        p = make_p(shape, d, rng)
        occupied.append(float(1 / np.sum(p ** 2)))           # effective number of outcomes
        cal = rng.multinomial(SHOTS, p, size=K); single = rng.multinomial(SHOTS, p)
        pooled = cal.sum(0)
        tau, tcp, dobs, add, dr = thresholds(as_counts(pooled), K * SHOTS, B, rng, cal=cal)
        tau1, tc1, _, _, _ = thresholds(as_counts(single), SHOTS, B, rng)
        obs.append(dobs); unseen.append(add); drifts += dr
        tests = rng.multinomial(SHOTS, p, size=TESTS)
        dp = jsd_arrays(np.broadcast_to(pooled, tests.shape), tests)
        ds = jsd_arrays(np.broadcast_to(single, tests.shape), tests)
        for a in ALPHAS:
            fp["nao"][a] += int((dp > tau[a]).sum()); fp["chi2_pooled"][a] += int((dp > tcp[a]).sum())
            fp["nao1"][a] += int((ds > tau1[a]).sum()); fp["chi2_single"][a] += int((ds > tc1[a]).sum())
        if per_window is not None:
            per_window.append({"shape": shape, "d": d, "tests": TESTS,
                               "pooled": diagnostics(pooled), "single": diagnostics(single),
                               "fp_nao": {str(a): int((dp > tau[a]).sum()) for a in ALPHAS},
                               "fp_nao1": {str(a): int((ds > tau1[a]).sum()) for a in ALPHAS}})
    n = windows * TESTS
    return {"shape": shape, "d": d, "shots_per_outcome": SHOTS / d, "tests": n, "windows": windows,
            "effective_outcomes": float(np.mean(occupied)), "observed_outcomes_pooled": float(np.mean(obs)),
            "unseen_added_pooled": float(np.mean(unseen)), "drift_flag_rate": drifts / windows,
            "fpr": {o: {str(a): v / n for a, v in x.items()} for o, x in fp.items()},
            "fp": {o: {str(a): v for a, v in x.items()} for o, x in fp.items()}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--shape", choices=["flat", "skewed", "headtail", "summary"], required=True)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time()
    d = a.out or os.path.join(RESULTS, "rq1_sparsity"); os.makedirs(d, exist_ok=True)
    if a.shape == "summary":
        return summarize(d)
    rng = np.random.default_rng(a.seed + {"flat": 0, "skewed": 1, "headtail": 2}[a.shape]); rows = []; pw = []
    for dd in DS:
        rows.append(cell(a.shape, dd, a.windows, a.B, rng, pw)); r = rows[-1]
        print(f"{a.shape:6s} d={dd:6d} N/d={SHOTS/dd:7.2f} NAO FPR@1e-2={r['fpr']['nao']['0.01']:.4f} NAO-1={r['fpr']['nao1']['0.01']:.4f}", flush=True)
    save(rows, os.path.join(d, f"results_{a.shape}.json"))
    save(pw, os.path.join(d, f"windows_{a.shape}.json"))
    md = os.path.join(d, f"meta_{a.shape}"); os.makedirs(md, exist_ok=True)
    write_meta(md, vars(a) | {"shots": SHOTS, "K": K, "tests_per_window": TESTS}, t0)


def summarize(d):
    import json
    rows, wins = [], []
    for sh in ("flat", "skewed", "headtail"):
        f = os.path.join(d, f"results_{sh}.json")
        if os.path.exists(f):
            rows += json.load(open(f))
        f = os.path.join(d, f"windows_{sh}.json")
        if os.path.exists(f):
            wins += json.load(open(f))
    save(rows, os.path.join(d, "results.json"))
    bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0001]
    by_diag = {}
    for mode, ref in (("nao", "pooled"), ("nao1", "single")):
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = [w for w in wins if lo <= w[ref]["good_turing_unseen_mass"] < hi]
            n = sum(w["tests"] for w in s)
            out.append({"unseen_mass": [lo, min(hi, 1.0)], "windows": len(s), "tests": n,
                        "fpr": {a: (sum(w[f"fp_{mode}"][a] for w in s) / n if n else None) for a in ("0.001", "0.01", "0.05")}})
        by_diag[mode] = out
    spo_bins = [0, 2, 4, 8, 16, 32, 1e9]
    by_spo = {}
    for mode in ("nao", "nao1"):
        out = []
        for lo, hi in zip(spo_bins[:-1], spo_bins[1:]):
            s = [w for w in wins if lo <= w["single"]["shots_per_observed_outcome"] < hi]
            n = sum(w["tests"] for w in s)
            out.append({"shots_per_observed_outcome": [lo, hi if hi < 1e9 else None], "windows": len(s), "tests": n,
                        "shapes": sorted({w["shape"] for w in s}),
                        "fpr": {a: (sum(w[f"fp_{mode}"][a] for w in s) / n if n else None) for a in ("0.001", "0.01", "0.05")}})
        by_spo[mode] = out
    save(by_diag, os.path.join(d, "by_unseen_mass.json"))
    save(by_spo, os.path.join(d, "by_shots_per_observed_outcome.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ1 SPARSITY: observed false-positive rate vs nominal alpha (N = {SHOTS}; K = {K} pooled runs for NAO)\n")
        for r in rows:
            f.write(f"\n{r['shape']} d={r['d']} shots/outcome={r['shots_per_outcome']:g} tests={r['tests']} effective outcomes={r['effective_outcomes']:.0f} "
                    f"observed (pooled)={r['observed_outcomes_pooled']:.0f} unseen added={r['unseen_added_pooled']:.0f} drift flagged={r['drift_flag_rate']:.3f}\n")
            for o in ("nao", "nao1", "chi2_pooled", "chi2_single"):
                f.write(f"  {o:12s} " + "  ".join(f"alpha={k}: {v:.4f}" for k, v in r["fpr"][o].items()) + "\n")
        f.write("\nFalse-positive rate by the Good-Turing unseen mass f1/n of the reference (all shapes pooled)\n")
        for mode, out in by_diag.items():
            for b in out:
                if b["tests"]:
                    f.write(f"  {mode:5s} f1/n in [{b['unseen_mass'][0]:.2f}, {b['unseen_mass'][1]:.2f}) windows={b['windows']:4d} "
                            + "  ".join(f"alpha={k}: {v:.4f}" for k, v in b["fpr"].items()) + "\n")
        f.write("\nFalse-positive rate by the shots per observed outcome of one run of P, N / (outcomes observed in that run) (all shapes pooled)\n")
        for mode, out in by_spo.items():
            for b in out:
                if b["tests"]:
                    hi = "inf" if b["shots_per_observed_outcome"][1] is None else f"{b['shots_per_observed_outcome'][1]:g}"
                    f.write(f"  {mode:5s} N/observed in [{b['shots_per_observed_outcome'][0]:g}, {hi}) windows={b['windows']:4d} "
                            + "  ".join(f"alpha={k}: {v:.4f}" for k, v in b["fpr"].items()) + f"  shapes={','.join(b['shapes'])}\n")
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
