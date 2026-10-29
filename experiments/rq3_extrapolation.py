# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ3 (bug-noise attribution, part 2): choice of the zero-noise extrapolation.

The attribution cases of rq3_attribution.py (same seed, same mutants and
faults) are executed once at noise scale factors s = 1, 3, 5 and the counts
are stored, so that every extrapolation is evaluated on the same executions:
  linear       Richardson on s = 1, 3          (4 executions per case)
  quadratic    Richardson on s = 1, 3, 5       (6 executions per case)
  exponential  global-depolarizing exponential on s = 1, 3 (4 executions)
The exponential model p(s) = u + (p_0 - u) f^s is the standard alternative to
Richardson extrapolation in zero-noise extrapolation; it removes the bias that
a linear fit incurs when P and P' decay at different rates, which is the case
for noise-amplifying mutants (P' has more gates than P).

(a) synthetic noise: 9 benchmarks x low, medium, high noise;
(b) device snapshots (FakeManilaV2, FakeLagosV2; small benchmarks), whose
    noise is not depolarizing, as a check that the exponential model does not
    depend on the synthetic noise model.
Output: results/rq3_extrapolation/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from common import BENCH, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import (Executor, MutationEngine, NOISE_AMPLIFYING, NOISE_NEUTRAL, SEM_ADD, SEM_SUB,
                            compute_threshold, jsd, pooled, inject_all, make_noisy_backend)
from nisq_framework.attribution import fold, attribute_from_counts
from nisq_framework.stats import cohen_kappa, confusion

NOISE3 = {"low": 0.001, "medium": 0.005, "high": 0.02}
SMALL = ["bell", "ghz_3", "qft_3", "qaoa_3", "vqe_3"]
ALPHA = 1e-3
PLAN = {NOISE_AMPLIFYING: 10, NOISE_NEUTRAL: 5, SEM_ADD: 8, SEM_SUB: 8}
METHODS = {"linear": (1, 3), "quadratic": (1, 3, 5), "exponential": (1, 3)}


def cases_for(qc, seed):
    eng = MutationEngine(qc, seed=seed)
    cases = [("unmodified", "unmodified", qc, "noise")]
    for fam, n in PLAN.items():
        for m in eng.generate(n, fam):
            cases.append((fam, m.operator, m.circuit, "noise" if m.output_equivalent else "bug"))
    for r in inject_all(qc):
        if r["applicable"]:
            cases.append(("injected_fault", r["injector"], r["circuit"], "noise" if r["output_equivalent"] else "bug"))
    return cases


def run_cell(qc, be, cases, ex, rng, B, env, b, nl):
    out = []
    cal = ex.counts_many(qc, be, SHOTS, 8)
    tau = compute_threshold(cal, SHOTS, ALPHA, "pooled", n_qubits=qc.num_qubits, B=B, rng=rng).tau
    ref = pooled(cal)
    folded_P = {s: fold(qc, s) for s in (1, 3, 5)}
    for kind, op, T, truth in cases:
        triggered = jsd(ref, ex.counts(T, be, SHOTS, layout_from=qc)) > tau
        cP, cT = {}, {}
        for s in (1, 3, 5):
            cP[s] = ex.counts(folded_P[s], be, SHOTS)
            cT[s] = ex.counts(fold(T, s), be, SHOTS, layout_from=folded_P[s])
        rec = {"environment": env, "benchmark": b, "noise": nl, "case_type": kind, "operator": op, "truth": truth,
               "triggered": bool(triggered), "counts_P": {str(s): dict(cP[s]) for s in cP}, "counts_T": {str(s): dict(cT[s]) for s in cT}}
        for m, sc in METHODS.items():
            r = attribute_from_counts([cP[s] for s in sc], [cT[s] for s in sc], sc, method=m, alpha=1e-2, B=B,
                                      seed=int(rng.integers(1 << 30)), n_outcomes=2 ** qc.num_clbits)
            rec[m] = r["attribution"]; rec[f"r0_{m}"] = r["r0"]; rec[f"null_{m}"] = r["null_q"]
        out.append(rec)
    return out


def summarize(S):
    res = {"n": len(S), "n_bug": sum(r["truth"] == "bug" for r in S), "n_noise": sum(r["truth"] == "noise" for r in S)}
    for m in METHODS:
        yt = [r["truth"] for r in S]; yp = [r[m] for r in S]
        cm = confusion([t == "bug" for t in yt], [p == "bug" for p in yp])
        res[m] = {"accuracy": float(np.mean([t == p for t, p in zip(yt, yp)])), "kappa": cohen_kappa(yt, yp),
                  "bug_precision": cm["precision"], "bug_recall": cm["recall"],
                  "noise_recall": 1 - cm["fpr"] if cm["fpr"] == cm["fpr"] else None,
                  "confusion": {k: cm[k] for k in ("tp", "fp", "fn", "tn")},
                  "by_noise": {nl: float(np.mean([r["truth"] == r[m] for r in S if r["noise"] == nl]))
                               for nl in sorted({r["noise"] for r in S})}}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq3_extrapolation", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
    recs = []
    for b in BENCH:
        qc, _ = get_benchmark(b); cases = cases_for(qc, a.seed)
        for nl, lv in NOISE3.items():
            recs += run_cell(qc, make_noisy_backend(lv), cases, ex, rng, a.B, "synthetic", b, nl)
            print("synthetic", b, nl, flush=True)
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2, FakeManilaV2
    for be in (FakeManilaV2(), FakeLagosV2()):
        name = be.name
        for b in SMALL:
            qc, _ = get_benchmark(b)
            recs += run_cell(qc, be, cases_for(qc, a.seed), ex, rng, a.B, "snapshot", b, name)
            print("snapshot", name, b, flush=True)
    save(recs, os.path.join(d, "records.json"))
    summ = {}
    for env in ("synthetic", "snapshot"):
        E = [r for r in recs if r["environment"] == env]
        summ[env] = {"all_cases": summarize(E), "triggered_only": summarize([r for r in E if r["triggered"]])}
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ3 ATTRIBUTION: CHOICE OF EXTRAPOLATION (same executions for all methods)\n")
        for env, v in summ.items():
            for scope, x in v.items():
                f.write(f"\n{env} / {scope}: n={x['n']} (bug {x['n_bug']}, noise {x['n_noise']})\n")
                for m in METHODS:
                    y = x[m]
                    f.write(f"  {m:12s} acc={y['accuracy']:.3f} kappa={y['kappa']:.3f} bugP={y['bug_precision']:.3f} "
                            f"bugR={y['bug_recall']:.3f} noiseR={(y['noise_recall'] if y['noise_recall'] is not None else float('nan')):.3f} "
                            f"by_noise={ {k: round(z, 3) for k, z in y['by_noise'].items()} }\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha_trigger": ALPHA, "plan": {k: v for k, v in PLAN.items()},
                             "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
