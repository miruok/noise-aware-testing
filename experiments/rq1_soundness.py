# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ1 (soundness): false-positive rate of each oracle on correct programs, and
agreement of the observed rate with the nominal alpha.

For every (benchmark, noise) cell, W independent calibration windows are
drawn. In each window every oracle builds its own reference and threshold,
and the same T fresh runs of the correct program are judged by all oracles
(paired design). Any rejection is a false positive.

Oracles: NAO (K=8; alpha 1e-2, 1e-3), NAO-1 (alpha 1e-2, 1e-3), SPC envelope
(K=12, k=4, tau_min=0.02), fixed threshold vs ideal reference (tau in
{0.005, 0.01, 0.02, 0.05, 0.10}), QOIN-style filter (K_c=5, tau=0.05).
Output: results/rq1_soundness/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from qiskit_aer import AerSimulator
from common import BENCH, NOISE, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, compute_threshold, spc_envelope_threshold, jsd, pooled, make_noisy_backend
from nisq_framework.stats import cluster_bootstrap, wilson

FIXED = [0.005, 0.01, 0.02, 0.05, 0.10]
ALPHAS = [1e-2, 1e-3]


def run_window(qc, backend, ex, ideal_backend, rng, T, B):
    runs = ex.counts_many(qc, backend, SHOTS, 13)
    tests = ex.counts_many(qc, backend, SHOTS, T)
    ideal = ex.counts(qc, ideal_backend, SHOTS)
    nq = qc.num_qubits
    taus, refs, info = {}, {}, {}
    for a in ALPHAS:
        r = compute_threshold([], SHOTS, a, "single", n_qubits=nq, B=B, reference_counts=runs[12], rng=rng)
        taus[f"nao1_a{a:g}"] = r.tau; refs[f"nao1_a{a:g}"] = runs[12]
        info[f"nao1_a{a:g}"] = {"tau_chi2": r.tau_chi2, "d_obs": r.d_observed, "unseen": r.d_unseen_added}
        r = compute_threshold(runs[:8], SHOTS, a, "pooled", n_qubits=nq, B=B, rng=rng)
        taus[f"nao_a{a:g}"] = r.tau; refs[f"nao_a{a:g}"] = pooled(runs[:8])
        info[f"nao_a{a:g}"] = {"tau_null": r.tau_null, "tau_emp": r.tau_empirical, "drift": r.drift_detected,
                                "homogeneity_p": r.homogeneity_p, "dispersion": r.dispersion_ratio}
    taus["spc_envelope"] = spc_envelope_threshold(runs[:12]); refs["spc_envelope"] = runs[12]
    for t in FIXED:
        taus[f"fixed_{t:g}"] = t; refs[f"fixed_{t:g}"] = ideal
    n_id = sum(ideal.values()); pid = {k: v / n_id for k, v in ideal.items()}
    keys = set(pid).union(*[set(r) for r in runs[:5]])
    avg = {k: np.mean([r.get(k, 0) / SHOTS for r in runs[:5]]) for k in keys}
    f = {k: (avg[k] + 1e-3) / (pid.get(k, 0) + 1e-3) for k in keys}
    out = {name: [] for name in taus}; out["qoin_style"] = []
    for c in tests:
        for name, tau in taus.items():
            out[name].append(int(jsd(refs[name], c) > tau))
        corr = {k: (v / SHOTS) / max(f.get(k, 1.0), 1e-6) for k, v in c.items()}
        s = sum(corr.values()); corr = {k: v / s for k, v in corr.items()}
        out["qoin_style"].append(int(jsd(pid, corr) > 0.05))
    taus["qoin_style"] = 0.05
    return out, taus, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=40); ap.add_argument("--tests", type=int, default=5)
    ap.add_argument("--B", type=int, default=4000); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bench", nargs="*", default=BENCH); ap.add_argument("--out")
    a = ap.parse_args(); t0 = time.time(); d = out_dir("rq1_soundness", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed + 1); ideal_b = AerSimulator()
    cells = []
    for b in a.bench:
        qc, _ = get_benchmark(b)
        for nl, lv in NOISE.items():
            backend = make_noisy_backend(lv); W = []
            for _ in range(a.windows):
                fp, taus, info = run_window(qc, backend, ex, ideal_b, rng, a.tests, a.B)
                W.append({"fp": fp, "tau": taus, "info": info})
            cells.append({"benchmark": b, "noise": nl, "eta": lv, "windows": W})
            print(f"{b:8s} {nl:9s} " + " ".join(f"{k}={sum(sum(x['fp'][k]) for x in W)}" for k in ("nao_a0.001", "nao1_a0.001", "spc_envelope", "fixed_0.05", "qoin_style")), flush=True)
    save(cells, os.path.join(d, "per_window.json"))
    oracles = list(cells[0]["windows"][0]["fp"].keys())
    summary = {"by_oracle": {}, "by_oracle_noise": {}, "thresholds": {}, "drift": {}}
    for o in oracles:
        clusters = [w["fp"][o] for c in cells for w in c["windows"]]
        cb = cluster_bootstrap(clusters, seed=1); k = sum(map(sum, clusters)); n = sum(map(len, clusters))
        wk = sum(any(x) for x in clusters)
        summary["by_oracle"][o] = {"fp": k, "tests": n, "fpr": k / n, "fpr_cluster_ci": cb["ci"],
                                   "fpr_wilson_ci_naive": list(wilson(k, n)), "windows_with_fp": wk,
                                   "windows": len(clusters), "window_fpr": wk / len(clusters)}
        for nl in NOISE:
            cl = [w["fp"][o] for c in cells if c["noise"] == nl for w in c["windows"]]
            kk = sum(map(sum, cl)); nn = sum(map(len, cl))
            summary["by_oracle_noise"].setdefault(o, {})[nl] = {"fp": kk, "tests": nn, "fpr": kk / nn}
    for c in cells:
        key = f"{c['benchmark']}|{c['noise']}"
        th = {o: float(np.median([w["tau"][o] for w in c["windows"]])) for o in ("nao1_a0.001", "nao_a0.001", "spc_envelope")}
        th["spc_floor_bound_fraction"] = float(np.mean([w["tau"]["spc_envelope"] <= 0.02 + 1e-12 for w in c["windows"]]))
        th["tau_chi2_a0.001"] = float(np.median([w["info"]["nao1_a0.001"]["tau_chi2"] for w in c["windows"]]))
        summary["thresholds"][key] = th
        summary["drift"][key] = {"drift_flag_rate": float(np.mean([w["info"]["nao_a0.001"]["drift"] for w in c["windows"]])),
                                 "median_dispersion_ratio": float(np.median([w["info"]["nao_a0.001"]["dispersion"] for w in c["windows"]])),
                                 "median_homogeneity_p": float(np.median([w["info"]["nao_a0.001"]["homogeneity_p"] for w in c["windows"]]))}
    save(summary, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ1 SOUNDNESS: {len(a.bench)} benchmarks x {len(NOISE)} noise levels x {a.windows} windows x {a.tests} tests\n\n")
        f.write(f"{'oracle':16s} {'FP/tests':>12s} {'FPR':>8s} {'95% CI (cluster)':>20s} {'windows w/ FP':>14s}\n")
        for o, v in summary["by_oracle"].items():
            f.write(f"{o:16s} {v['fp']:>5d}/{v['tests']:<6d} {v['fpr']:8.4f}   [{v['fpr_cluster_ci'][0]:.4f}, {v['fpr_cluster_ci'][1]:.4f}]   {v['windows_with_fp']:>4d}/{v['windows']}\n")
        f.write("\nFPR by noise level\n" + f"{'oracle':16s}" + "".join(f"{n:>11s}" for n in NOISE) + "\n")
        for o, v in summary["by_oracle_noise"].items():
            f.write(f"{o:16s}" + "".join(f"{v[n]['fpr']:11.4f}" for n in NOISE) + "\n")
        f.write("\nMedian thresholds (alpha = 1e-3)\n")
        for k, v in summary["thresholds"].items():
            f.write(f"  {k:18s} NAO-1={v['nao1_a0.001']:.4f} chi2={v['tau_chi2_a0.001']:.4f} NAO={v['nao_a0.001']:.4f} SPC={v['spc_envelope']:.4f} SPC_floor_bound={v['spc_floor_bound_fraction']:.2f}  drift_flag_rate={summary['drift'][k]['drift_flag_rate']:.3f} dispersion={summary['drift'][k]['median_dispersion_ratio']:.2f}\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alphas": ALPHAS, "fixed_taus": FIXED, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read()[:2500])


if __name__ == "__main__":
    main()
