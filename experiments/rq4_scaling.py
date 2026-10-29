# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ4 (fault detection, part 2): scaling to 16 qubits; full-distribution vs
marginal comparison.

Circuits: GHZ (star) n in {8, 10, 12, 14, 16}; QAOA-MaxCut ring p = 1,
n in {8, 10, 12}; eta = 0.005; 4096 shots.
Oracles:
  nao           NAO on the full output distribution
  nao_marginal  NAO applied to every 2-qubit marginal with a chi-square
                 null threshold at alpha / m (Bonferroni over m = n(n-1)/2);
                 the test fails if any marginal exceeds its threshold
  spc_envelope   SPC envelope on the full distribution
Output: results/rq4_scaling/.
"""
from __future__ import annotations
import argparse, itertools, os, time
import numpy as np
from common import SHOTS, out_dir, save, write_meta
from nisq_framework.benchmarks import ghz_n, qaoa_maxcut_ring
from nisq_framework import (Executor, compute_threshold, chi2_threshold, spc_envelope_threshold, jsd, pooled, inject_all, make_noisy_backend)

ALPHA = 1e-3
ETA = 0.005
ORACLES = ["nao", "nao_marginal", "spc_envelope"]


def marginals(counts, pairs):
    out = []
    for i, j in pairs:
        m = {}
        for k, v in counts.items():
            s = k[::-1]; key = s[i] + s[j]
            m[key] = m.get(key, 0) + v
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--tests", type=int, default=4); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17); ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq4_scaling", a.out); ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
    circuits = [(f"ghz_{n}", ghz_n(n)[0]) for n in (8, 10, 12, 14, 16)] + [(f"qaoa_{n}", qaoa_maxcut_ring(n, p=1)[0]) for n in (8, 10, 12)]
    be = make_noisy_backend(ETA); rows = []
    for name, qc in circuits:
        n = qc.num_qubits; pairs = list(itertools.combinations(range(n), 2)); m = len(pairs)
        faults = [r for r in inject_all(qc, names=["remove_entanglement", "wrong_basis", "extra_x", "wrong_angle", "drop_last_gate"])
                  if r["applicable"] and not r["output_equivalent"]]
        fp = {k: 0 for k in ORACLES}; det = {k: 0 for k in ORACLES}; nt = nf = 0; taus = []
        for _ in range(a.windows):
            cal = ex.counts_many(qc, be, SHOTS, 12)
            ref = pooled(cal[:8])
            tau = compute_threshold(cal[:8], SHOTS, ALPHA, "pooled", n_qubits=n, B=a.B, rng=rng).tau
            tspc = spc_envelope_threshold(cal); r1 = cal[11]
            mref = marginals(ref, pairs); mt = chi2_threshold(4, sum(ref.values()), SHOTS, ALPHA / m)
            taus.append((tau, tspc))

            def judge(c):
                mc = marginals(c, pairs)
                return {"nao": jsd(ref, c) > tau, "spc_envelope": jsd(r1, c) > tspc,
                        "nao_marginal": any(jsd(x, y) > mt for x, y in zip(mref, mc))}
            for _ in range(a.tests):
                j = judge(ex.counts(qc, be, SHOTS)); nt += 1
                for k in ORACLES: fp[k] += j[k]
            for f_ in faults:
                j = judge(ex.counts(f_["circuit"], be, SHOTS)); nf += 1
                for k in ORACLES: det[k] += j[k]
        rows.append({"circuit": name, "qubits": n, "outcomes": 2 ** n, "shots_per_outcome": SHOTS / 2 ** n, "tests": nt,
                     "fault_evals": nf, "faults": [f_["injector"] for f_ in faults],
                     "fpr": {k: v / nt for k, v in fp.items()}, "recall": {k: (v / nf if nf else None) for k, v in det.items()},
                     "median_tau_nao": float(np.median([t[0] for t in taus])), "median_tau_spc": float(np.median([t[1] for t in taus]))})
        print(rows[-1]["circuit"], rows[-1]["fpr"], rows[-1]["recall"], flush=True)
    save(rows, os.path.join(d, "results.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ4 SCALING (eta={ETA}, {SHOTS} shots, alpha={ALPHA})\n\n")
        f.write(f"{'circuit':8s} {'q':>3s} {'shots/outcome':>14s} {'tau_NAO':>9s} {'tau_SPC':>8s}  FPR NAO/marginal/SPC    recall NAO/marginal/SPC\n")
        for r in rows:
            f.write(f"{r['circuit']:8s} {r['qubits']:3d} {r['shots_per_outcome']:14.3f} {r['median_tau_nao']:9.4f} {r['median_tau_spc']:8.4f}  "
                    f"{r['fpr']['nao']:.3f}/{r['fpr']['nao_marginal']:.3f}/{r['fpr']['spc_envelope']:.3f}       "
                    f"{r['recall']['nao']:.3f}/{r['recall']['nao_marginal']:.3f}/{r['recall']['spc_envelope']:.3f}\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "eta": ETA, "alpha": ALPHA, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
