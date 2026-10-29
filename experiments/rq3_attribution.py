# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ3 (bug-noise attribution): when an oracle reports a difference between P
and P', is it a semantic fault or a noise effect?

Cases per (benchmark, noise level):
  noise truth: noise-amplifying mutants (10), noise-neutral
               mutants (5), and the unmodified program;
  bug truth:   semantic mutants (8 additive, 8 subtractive) and every
               non-equivalent injected fault.
Label = bug iff the exact ideal outputs of P and P' differ.

Each case is first judged by NAO (trigger). Attribution is evaluated on all
cases (classifier validation) and on triggered cases (operational use).
Methods:
  zne_linear          folding at s = 1, 3; linear zero-noise extrapolation
  zne_quadratic       folding at s = 1, 3, 5; quadratic extrapolation
  reexecution_triage  baseline: 4x shots, 0.5x noise model, noiseless simulator
Output: results/rq3_attribution/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from common import BENCH, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import (Executor, MutationEngine, NOISE_AMPLIFYING, NOISE_NEUTRAL, SEM_ADD, SEM_SUB, TAXONOMY_CLASS,
                  compute_threshold, jsd, pooled, inject_all, attribute_zne, reexecution_triage, make_noisy_backend)
from nisq_framework.stats import cohen_kappa, confusion

NOISE3 = {"low": 0.001, "medium": 0.005, "high": 0.02}
ALPHA = 1e-3
PLAN = {NOISE_AMPLIFYING: 10, NOISE_NEUTRAL: 5, SEM_ADD: 8, SEM_SUB: 8}
METHODS = ["zne_linear", "zne_quadratic", "reexecution_triage"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--bench", nargs="*", default=BENCH)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq3_attribution", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
    recs = []
    for b in a.bench:
        qc, _ = get_benchmark(b)
        eng = MutationEngine(qc, seed=a.seed)
        cases = [("unmodified", "unmodified", qc, "noise")]
        for fam, n in PLAN.items():
            for m in eng.generate(n, fam):
                cases.append((fam, m.operator, m.circuit, "noise" if m.output_equivalent else "bug"))
        for r in inject_all(qc):
            if r["applicable"]:
                cases.append(("injected_fault", r["injector"], r["circuit"], "noise" if r["output_equivalent"] else "bug"))
        for nl, lv in NOISE3.items():
            be = make_noisy_backend(lv)
            cal = ex.counts_many(qc, be, SHOTS, 8)
            tau = compute_threshold(cal, SHOTS, ALPHA, "pooled", n_qubits=qc.num_qubits, B=a.B, rng=rng).tau
            ref = pooled(cal)
            for kind, op, T, truth in cases:
                c = ex.counts(T, be, SHOTS, layout_from=qc)
                triggered = jsd(ref, c) > tau
                rl = attribute_zne(qc, T, be, ex, SHOTS, scales=(1, 3), method="linear", alpha=1e-2, B=a.B, seed=int(rng.integers(1 << 30)))
                rq = attribute_zne(qc, T, be, ex, SHOTS, scales=(1, 3, 5), method="quadratic", alpha=1e-2, B=a.B, seed=int(rng.integers(1 << 30)))
                re = reexecution_triage(qc, T, lv, ex, SHOTS)
                recs.append({"benchmark": b, "noise": nl, "case_type": kind, "operator": op, "truth": truth,
                             "triggered": bool(triggered), "zne_linear": rl["attribution"], "zne_quadratic": rq["attribution"],
                             "reexecution_triage": re["attribution"], "r0_linear": rl["r0"], "null_linear": rl["null_q"],
                             "r0_quadratic": rq["r0"], "null_quadratic": rq["null_q"], "r_by_scale": rq["r_by_scale"]})
            print(f"{b:8s} {nl:7s} cases={len(cases)} triggered={sum(r['triggered'] for r in recs if r['benchmark']==b and r['noise']==nl)}", flush=True)
    save(recs, os.path.join(d, "records.json"))
    summ = {}
    for scope in ("all_cases", "triggered_only"):
        S = recs if scope == "all_cases" else [r for r in recs if r["triggered"]]
        summ[scope] = {"n": len(S), "n_bug": sum(r["truth"] == "bug" for r in S), "n_noise": sum(r["truth"] == "noise" for r in S)}
        for m in METHODS:
            yt = [r["truth"] for r in S]; yp = [r[m] for r in S]
            cm = confusion([t == "bug" for t in yt], [p == "bug" for p in yp])
            summ[scope][m] = {"accuracy": float(np.mean([t == p for t, p in zip(yt, yp)])), "kappa": cohen_kappa(yt, yp),
                              "bug_precision": cm["precision"], "bug_recall": cm["recall"],
                              "noise_recall": 1 - cm["fpr"] if cm["fpr"] == cm["fpr"] else None,
                              "confusion": {k: cm[k] for k in ("tp", "fp", "fn", "tn")},
                              "by_noise": {nl: float(np.mean([r["truth"] == r[m] for r in S if r["noise"] == nl])) for nl in NOISE3 if any(r["noise"] == nl for r in S)},
                              "by_case_type": {c: float(np.mean([r["truth"] == r[m] for r in S if r["case_type"] == c])) for c in sorted({r["case_type"] for r in S})}}
    summ["trigger"] = {c: {"triggered": sum(r["triggered"] for r in recs if r["case_type"] == c), "n": sum(r["case_type"] == c for r in recs)}
                       for c in sorted({r["case_type"] for r in recs})}
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ3 ATTRIBUTION (exact labels)\n\nNAO trigger rate by case type\n")
        for c, v in summ["trigger"].items():
            f.write(f"  {c:24s} {v['triggered']}/{v['n']}\n")
        for scope in ("all_cases", "triggered_only"):
            s = summ[scope]
            f.write(f"\n{scope}: n={s['n']} (bug {s['n_bug']}, noise {s['n_noise']})\n")
            for m in METHODS:
                x = s[m]
                f.write(f"  {m:19s} acc={x['accuracy']:.3f} kappa={x['kappa']:.3f} bugP={x['bug_precision']:.3f} bugR={x['bug_recall']:.3f} noiseR={x['noise_recall']:.3f} {x['confusion']}\n")
                f.write(f"  {'':19s} by noise {x['by_noise']}\n  {'':19s} by case type {x['by_case_type']}\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha_trigger": ALPHA, "plan": PLAN, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
