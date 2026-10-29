# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ2 (taxonomy, part 2): dose-response of noise-amplifying mutants.

A noise-amplifying mutant preserves the program's semantics and adds physical
operations, so it changes the noisy output distribution by an amount that
depends on how much gate error it adds. This experiment tests whether the
oracle's rejections of such mutants are explained by that amount.

Mutants: k inverse pairs inserted at independent uniformly random positions
before the first measurement, each pair separated by barriers:
  cx_pairs  k in {1, 2, 4}   (2k added CNOTs)
  1q_pairs  k in {1, 4, 16}  (2k added single-qubit gates G G^dag)
10 mutants per (benchmark, kind, k), exact equivalence verified; 9 benchmarks
at low, medium and high noise. Per mutant: one 4096-shot run judged by NAO
(K = 8, alpha = 1e-3) and a 200,000-shot run that estimates the true noisy
divergence delta from the original; the noncentral chi-square model gives the
predicted rejection probability.
Output: results/rq2_amplification/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import Barrier, CXGate, HGate, SGate, SdgGate, TGate, TdgGate, XGate, YGate, ZGate
from common import BENCH, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, compute_threshold, jsd, pooled, make_noisy_backend, detection_power
from nisq_framework.semantics import first_measure_index, ideal_jsd, EQUIV_TOL
from nisq_framework.stats import wilson

NOISE3 = {"low": 0.001, "medium": 0.005, "high": 0.02}
DOSES = {"cx_pairs": (1, 2, 4), "1q_pairs": (1, 4, 16)}
PAIRS_1Q = [(HGate, HGate), (XGate, XGate), (YGate, YGate), (ZGate, ZGate), (SGate, SdgGate), (TGate, TdgGate)]
ALPHA = 1e-3
N_BIG = 200_000


def amplify(qc, kind, k, rng):
    out = qc.copy(); n = qc.num_qubits
    for _ in range(k):
        pos = int(rng.integers(0, first_measure_index(out) + 1))
        if kind == "cx_pairs":
            c, t = map(int, rng.choice(n, 2, replace=False)); qs = [c, t]
            ops = [(Barrier(2), qs), (CXGate(), qs), (Barrier(2), qs), (CXGate(), qs), (Barrier(2), qs)]
        else:
            q = int(rng.integers(n)); a, b = PAIRS_1Q[rng.integers(len(PAIRS_1Q))]
            ops = [(Barrier(1), [q]), (a(), [q]), (Barrier(1), [q]), (b(), [q]), (Barrier(1), [q])]
        for j, (g, qs) in enumerate(ops):
            out.data.insert(pos + j, CircuitInstruction(g, [out.qubits[x] for x in qs], []))
    assert ideal_jsd(qc, out) < EQUIV_TOL
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-dose", type=int, default=10)
    ap.add_argument("--seed", type=int, default=77); ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq2_amplification", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
    recs = []
    for b in BENCH:
        qc, _ = get_benchmark(b)
        muts = [(kind, k, amplify(qc, kind, k, rng)) for kind, ks in DOSES.items() for k in ks for _ in range(a.per_dose)]
        for nl, eta in NOISE3.items():
            be = make_noisy_backend(eta)
            cal = ex.counts_many(qc, be, SHOTS, 8); ref = pooled(cal)
            tau = compute_threshold(cal, SHOTS, ALPHA, "pooled", n_qubits=qc.num_qubits, B=a.B, rng=rng).tau
            big = ex.counts(qc, be, N_BIG)
            e1, e2 = eta, min(10 * eta, 0.5)
            for kind, k, m in muts:
                c = ex.counts(m, be, SHOTS, layout_from=qc)
                bigm = ex.counts(m, be, N_BIG, layout_from=qc)
                delta = jsd(big, bigm); deff = len(set(big) | set(bigm))
                n2, n1 = (2 * k, 0) if kind == "cx_pairs" else (0, 2 * k)
                recs.append({"benchmark": b, "noise": nl, "kind": kind, "dose": k, "added_2q": n2, "added_1q": n1,
                             "added_error": float(1 - (1 - e2) ** n2 * (1 - e1) ** n1),
                             "rejected": bool(jsd(ref, c) > tau), "delta": delta, "d_eff": deff,
                             "predicted": detection_power(delta, deff, 8 * SHOTS, SHOTS, ALPHA)})
            print(b, nl, " ".join(f"{kind}{k}={np.mean([r['rejected'] for r in recs if r['benchmark']==b and r['noise']==nl and r['kind']==kind and r['dose']==k]):.1f}"
                                  for kind, ks in DOSES.items() for k in ks), flush=True)
    save(recs, os.path.join(d, "records.json"))
    summ = {"by_dose": [], "by_added_error": [], "calibration": []}
    for nl in NOISE3:
        for kind, ks in DOSES.items():
            for k in ks:
                S = [r for r in recs if r["noise"] == nl and r["kind"] == kind and r["dose"] == k]
                kk = sum(r["rejected"] for r in S)
                summ["by_dose"].append({"noise": nl, "kind": kind, "dose": k, "n": len(S), "rejected": kk, "rate": kk / len(S),
                                        "ci": list(wilson(kk, len(S))), "mean_predicted": float(np.mean([r["predicted"] for r in S]))})
    edges = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        S = [r for r in recs if lo <= r["added_error"] < hi or (hi == 1.0 and r["added_error"] == 1.0)]
        if S:
            kk = sum(r["rejected"] for r in S)
            summ["by_added_error"].append({"bin": [lo, hi], "n": len(S), "rate": kk / len(S), "ci": list(wilson(kk, len(S))),
                                           "mean_predicted": float(np.mean([r["predicted"] for r in S]))})
    for lo, hi in [(0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0001)]:
        S = [r for r in recs if lo <= r["predicted"] < hi]
        if S:
            summ["calibration"].append({"bin": [lo, min(hi, 1.0)], "n": len(S), "mean_predicted": float(np.mean([r["predicted"] for r in S])),
                                        "observed": float(np.mean([r["rejected"] for r in S]))})
    pr = np.array([r["predicted"] for r in recs]); ob = np.array([r["rejected"] for r in recs], float)
    summ["brier"] = float(np.mean((pr - ob) ** 2)); summ["brier_constant"] = float(np.mean((ob.mean() - ob) ** 2))
    from scipy import stats
    summ["spearman_added_error_vs_delta"] = float(stats.spearmanr([r["added_error"] for r in recs], [r["delta"] for r in recs])[0])
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ2 NOISE-AMPLIFICATION DOSE-RESPONSE (NAO, K = 8, alpha = 1e-3)\n\nRejection rate by dose\n")
        for x in summ["by_dose"]:
            f.write(f"  {x['noise']:7s} {x['kind']:9s} k={x['dose']:<3d} rejected {x['rejected']}/{x['n']} = {x['rate']:.3f}  predicted {x['mean_predicted']:.3f}\n")
        f.write("\nRejection rate by added gate error\n")
        for x in summ["by_added_error"]:
            f.write(f"  [{x['bin'][0]:.2f}, {x['bin'][1]:.2f})  n={x['n']:4d}  rate={x['rate']:.3f}  predicted={x['mean_predicted']:.3f}\n")
        f.write("\nPredicted vs observed\n")
        for x in summ["calibration"]:
            f.write(f"  predicted [{x['bin'][0]:.1f}, {x['bin'][1]:.1f})  n={x['n']:4d}  mean predicted={x['mean_predicted']:.3f}  observed={x['observed']:.3f}\n")
        f.write(f"\nBrier score {summ['brier']:.4f} (constant predictor {summ['brier_constant']:.4f}); "
                f"Spearman(added error, delta) = {summ['spearman_added_error_vs_delta']:.3f}\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha": ALPHA, "n_big": N_BIG, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
