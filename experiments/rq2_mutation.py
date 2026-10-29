# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ2 (mutation score and taxonomy validation): mutation analysis under
noise with exact labels.

15 mutants per operator family (4 families, seed 42) on 9 benchmarks x 3
noise levels (low, medium, high). Each mutant is executed once and judged by
NAO, NAO-1 and the SPC envelope on the same counts.

Taxonomy validation (three classes: semantic, noise-amplifying,
noise-neutral). For each pair of classes and each noise level:
  H0: the rejection probability is equal in both classes (Fisher exact test
      on rejection counts, Bonferroni over 3 pairs);
  H0: the normalised divergence d/tau has the same distribution (Mann-Whitney
      U, two-sided) with Cliff's delta as effect size;
  H0: common odds ratio 1 within every cell (Cochran-Mantel-Haenszel test
      stratified by cell, conditioning on the shared calibration window);
  H0: per-cell rate differences symmetric around 0 (cell-level sign-flip test).
The two semantic families (additive, subtractive) are tested the same way
to check whether they should be separate classes.
Output: results/rq2_mutation/.
"""
from __future__ import annotations
import argparse, itertools, os, time
import numpy as np
from scipy import stats
from common import BENCH, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import (Executor, MutationEngine, FAMILIES, TAXONOMY_CLASS, compute_threshold,
                  spc_envelope_threshold, jsd, pooled, make_noisy_backend)
from nisq_framework.stats import wilson, cliffs_delta, cmh_test, signflip_test

NOISE3 = {"low": 0.001, "medium": 0.005, "high": 0.02}
ALPHA = 1e-3
ORACLES = ["nao", "nao1", "spc_envelope"]
CLASSES = ["semantic", "noise_amplifying", "noise_neutral"]


def cell_tests(S, key, a, b):
    """Tests that account for the calibration window shared by all mutants of a
    cell: CMH stratified by cell, and a cell-level sign-flip test."""
    tabs, diffs = [], []
    for cell in sorted({(r["benchmark"], r["noise"]) for r in S}):
        C = [r for r in S if (r["benchmark"], r["noise"]) == cell and r["original_passed_nao"]]
        ka = sum(r["rejected_nao"] for r in C if r[key] == a); na = sum(r[key] == a for r in C)
        kb = sum(r["rejected_nao"] for r in C if r[key] == b); nb = sum(r[key] == b for r in C)
        if na and nb:
            tabs.append(((ka, na - ka), (kb, nb - kb))); diffs.append(ka / na - kb / nb)
    diffs = np.array(diffs)
    return {"cmh_p": cmh_test(tabs), "signflip_p": signflip_test(diffs), "cells": len(diffs),
            "cells_higher": int((diffs > 0).sum()), "cells_equal": int((diffs == 0).sum()), "cells_lower": int((diffs < 0).sum())}


def pair_tests(groups_k, groups_x):
    out = {}
    pairs = list(itertools.combinations(groups_k, 2))
    for a, b in pairs:
        ka, na = groups_k[a]; kb, nb = groups_k[b]
        p_f = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
        p_mw = stats.mannwhitneyu(groups_x[a], groups_x[b], alternative="two-sided").pvalue
        out[f"{a}_vs_{b}"] = {"rate_a": ka / na, "rate_b": kb / nb, "fisher_p": float(p_f),
                              "fisher_significant_bonferroni": bool(p_f < 0.05 / len(pairs)),
                              "mw_p": float(p_mw), "cliffs_delta": cliffs_delta(groups_x[a], groups_x[b])}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-family", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq2_mutation", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed)
    recs = []
    for b in BENCH:
        qc, _ = get_benchmark(b)
        eng = MutationEngine(qc, seed=a.seed)
        muts = {f: eng.generate(a.per_family, f) for f in FAMILIES}
        for nl, lv in NOISE3.items():
            be = make_noisy_backend(lv)
            runs = ex.counts_many(qc, be, SHOTS, 13)
            tau = {"nao": compute_threshold(runs[:8], SHOTS, ALPHA, "pooled", n_qubits=qc.num_qubits, B=a.B, rng=rng).tau,
                   "nao1": compute_threshold([], SHOTS, ALPHA, "single", n_qubits=qc.num_qubits, B=a.B, reference_counts=runs[12], rng=rng).tau,
                   "spc_envelope": spc_envelope_threshold(runs[:12])}
            refs = {"nao": pooled(runs[:8]), "nao1": runs[12], "spc_envelope": runs[12]}
            orig = ex.counts(qc, be, SHOTS)
            orig_pass = {o: jsd(refs[o], orig) <= tau[o] for o in tau}
            for fam, ms in muts.items():
                for m in ms:
                    c = ex.counts(m.circuit, be, SHOTS, layout_from=qc)
                    div = {o: jsd(refs[o], c) for o in tau}
                    recs.append({"benchmark": b, "noise": nl, **m.record(),
                                 **{f"div_{o}": div[o] for o in tau},
                                 **{f"rejected_{o}": bool(orig_pass[o] and div[o] > tau[o]) for o in tau},
                                 **{f"tau_{o}": tau[o] for o in tau},
                                 **{f"original_passed_{o}": orig_pass[o] for o in tau}})
            print(f"{b:8s} {nl:7s} " + " ".join(f"{f}={np.mean([r['rejected_nao'] for r in recs if r['benchmark']==b and r['noise']==nl and r['family']==f]):.2f}" for f in FAMILIES), flush=True)
    save(recs, os.path.join(d, "records.json"))
    summ = {"rejection_by_class": {}, "rejection_by_family": {}, "by_noise": {}, "by_operator": {}, "taxonomy_tests": {}, "family_tests": {}}
    for o in ORACLES:
        for c in CLASSES:
            s = [r[f"rejected_{o}"] for r in recs if r["taxonomy_class"] == c]
            summ["rejection_by_class"].setdefault(o, {})[c] = {"k": int(sum(s)), "n": len(s), "rate": float(np.mean(s)), "ci": list(wilson(int(sum(s)), len(s)))}
            summ["by_noise"].setdefault(o, {})[c] = {nl: float(np.mean([r[f"rejected_{o}"] for r in recs if r["taxonomy_class"] == c and r["noise"] == nl])) for nl in NOISE3}
        for f in FAMILIES:
            s = [r[f"rejected_{o}"] for r in recs if r["family"] == f]
            summ["rejection_by_family"].setdefault(o, {})[f] = {"k": int(sum(s)), "n": len(s), "rate": float(np.mean(s))}
        for op in sorted({r["operator"] for r in recs}):
            s = [r[f"rejected_{o}"] for r in recs if r["operator"] == op]
            summ["by_operator"].setdefault(o, {})[op] = {"rate": float(np.mean(s)), "n": len(s)}
    for scope in ["pooled"] + list(NOISE3):
        S = recs if scope == "pooled" else [r for r in recs if r["noise"] == scope]
        gk = {c: (sum(r["rejected_nao"] for r in S if r["taxonomy_class"] == c), sum(r["taxonomy_class"] == c for r in S)) for c in CLASSES}
        gx = {c: [r["div_nao"] / r["tau_nao"] for r in S if r["taxonomy_class"] == c] for c in CLASSES}
        summ["taxonomy_tests"][scope] = pair_tests(gk, gx)
        for a_, b_ in itertools.combinations(CLASSES, 2):
            summ["taxonomy_tests"][scope][f"{a_}_vs_{b_}"].update(cell_tests(S, "taxonomy_class", a_, b_))
        fk = {f: (sum(r["rejected_nao"] for r in S if r["family"] == f), sum(r["family"] == f for r in S)) for f in ("semantic_additive", "semantic_subtractive")}
        fx = {f: [r["div_nao"] / r["tau_nao"] for r in S if r["family"] == f] for f in fk}
        summ["family_tests"][scope] = pair_tests(fk, fx)
        summ["family_tests"][scope]["semantic_additive_vs_semantic_subtractive"].update(cell_tests(S, "family", "semantic_additive", "semantic_subtractive"))
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ2 MUTATION ANALYSIS\n\nRejection rate by taxonomy class (semantic: mutation score; equivalent classes: false-rejection rate)\n")
        for o, v in summ["rejection_by_class"].items():
            f.write(f"  {o:13s} " + "  ".join(f"{c}={x['k']}/{x['n']}={x['rate']:.3f}" for c, x in v.items()) + "\n")
        f.write("\nBy family\n")
        for o, v in summ["rejection_by_family"].items():
            f.write(f"  {o:13s} " + "  ".join(f"{c}={x['k']}/{x['n']}={x['rate']:.3f}" for c, x in v.items()) + "\n")
        f.write("\nNAO by noise level\n")
        for c, v in summ["by_noise"]["nao"].items():
            f.write(f"  {c:24s} " + "  ".join(f"{n}={x:.3f}" for n, x in v.items()) + "\n")
        f.write("\nNAO by operator\n")
        for op, x in summ["by_operator"]["nao"].items():
            f.write(f"  {op:22s} {x['rate']:.3f} (n={x['n']})\n")
        for title, key in (("Taxonomy class tests (NAO)", "taxonomy_tests"), ("Semantic family tests (NAO)", "family_tests")):
            f.write(f"\n{title}\n")
            for scope, v in summ[key].items():
                for p, x in v.items():
                    f.write(f"  {scope:7s} {p:58s} rates {x['rate_a']:.3f} vs {x['rate_b']:.3f}  Fisher p={x['fisher_p']:.2e} sig={x['fisher_significant_bonferroni']}  MW p={x['mw_p']:.2e}  delta={x['cliffs_delta']:+.2f}  CMH p={x['cmh_p']:.2e}  cells >/=/< {x['cells_higher']}/{x['cells_equal']}/{x['cells_lower']}  sign-flip p={x['signflip_p']:.2e}\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha": ALPHA, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
