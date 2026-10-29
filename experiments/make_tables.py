# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Build results/SUMMARY.md and paper-ready LaTeX tables (results/tables/*.tex)
from the JSON outputs. Every number is read from a results file."""
from __future__ import annotations
import json, os
from common import RESULTS, PRETTY

T = os.path.join(RESULTS, "tables"); os.makedirs(T, exist_ok=True)
NAME = {"nao": "NAO", "nao1": "NAO-1", "spc_envelope": "SPC envelope", "qoin_style": "QOIN-style filter",
        "nao_a0.001": r"NAO ($\alpha=10^{-3}$)", "nao_a0.01": r"NAO ($\alpha=10^{-2}$)",
        "nao1_a0.001": r"NAO-1 ($\alpha=10^{-3}$)", "nao1_a0.01": r"NAO-1 ($\alpha=10^{-2}$)",
        "nao_marginal": "NAO (2-qubit marginals)",
        "zne_linear": "ZNE attribution (linear)", "zne_quadratic": "ZNE attribution (quadratic)",
        "reexecution_triage": "Re-execution triage"}
for t in ("0.005", "0.01", "0.02", "0.05", "0.1"):
    NAME[f"fixed_{t}"] = rf"Fixed $\tau={t}$"
CLASS = {"semantic": "Semantic", "noise_amplifying": "Noise-amplifying", "noise_neutral": "Noise-neutral",
         "semantic_additive": "Semantic (additive)", "semantic_subtractive": "Semantic (subtractive)"}
def md(s):
    for x, y in ((r"$|0\rangle$", "zero input"), (r"\downarrow", "↓"), (r"\uparrow", "↑"), (r"\alpha", "α"), (r"\tau", "τ"),
                 (r"\kappa", "κ"), (r"\delta", "δ"), (r"\chi^2", "χ²"), ("10^{-3}", "1e-3"), ("10^{-2}", "1e-2"),
                 ("$S_1$", "S1"), ("$S_T$", "ST"), ("$p$", "p"), ("$d$", "d"), ("$r$", "r"), ("\\%", "%"), ("$", "")):
        s = s.replace(x, y)
    return s


def load(*a):
    p = os.path.join(RESULTS, *a)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=3):
    return "--" if x is None or x != x else f"{x:.{n}f}"


def tex(name, caption, label, header, rows, colspec):
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(T, f"{name}.tex"), "w").write("\n".join(lines) + "\n")


def mdtable(header, rows):
    out = ["| " + " | ".join(md(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(md(c) for c in r) + " |" for r in rows]
    return out



def extra_tables(L):
    """Appendix and supplementary tables of the paper."""
    LEV = ["very_low", "low", "medium", "high", "very_high"]
    so = load("rq1_soundness", "summary.json")
    if so:
        order = [("nao_a0.001", r"NAO ($\alpha = 10^{-3}$)"), ("nao_a0.01", r"NAO ($\alpha = 10^{-2}$)"), ("nao1_a0.001", r"NAO-1 ($\alpha = 10^{-3}$)"),
                 ("nao1_a0.01", r"NAO-1 ($\alpha = 10^{-2}$)"), ("spc_envelope", "SPC envelope"), ("qoin_style", "QOIN-style filter")] + \
                [(f"fixed_{t}", rf"Fixed $\tau = {t}$") for t in ("0.005", "0.01", "0.02", "0.05", "0.1")]
        rows = [[n] + [f(so["by_oracle_noise"][k][l]["fpr"], 4) for l in LEV] for k, n in order]
        h = ["Oracle", "Very low", "Low", "Medium", "High", "Very high"]
        tex("app_fpr_noise", "False-positive rate on correct programs by noise level (RQ1).", "tab:app-fpr-noise", h, rows, "lrrrrr")
        L += ["## Appendix: FPR by noise level", ""] + mdtable(h, rows) + [""]
    fa = load("rq4_faults", "summary.json")
    if fa:
        inj = sorted(fa["by_injector"]["nao"])
        rows = [[i.replace("_", r"\_"), str(fa["by_injector"]["nao"][i]["n"])] + [f(fa["by_injector"][o][i]["recall"], 2) for o in ("nao", "nao1", "spc_envelope", "fixed_0.05")] for i in inj]
        h = ["Injector", "$n$", "NAO", "NAO-1", "SPC", r"Fixed $\tau=0.05$"]
        tex("app_injector", "Recall on non-equivalent injected faults by injector (RQ4).", "tab:app-injector", h, rows, "lrrrrr")
        L += ["## Appendix: recall by injector", ""] + mdtable(h, rows) + [""]
    mu = load("rq2_mutation", "summary.json")
    if mu:
        cls = {"insert_cnot": "Semantic", "insert_rotation": "Semantic", "insert_t": "Semantic", "insert_x": "Semantic", "remove_gate": "Semantic",
               "remove_1q_gate": "Semantic", "rotation_to_identity": "Semantic", "cx_pair": "Noise-amplifying", "inverse_pair_1q": "Noise-amplifying",
               "rotation_pair": "Noise-amplifying", "commute_swap": "Noise-neutral", "cx_to_h_cz_h": "Noise-neutral", "x_to_h_z_h": "Noise-neutral"}
        ops = [o for o in cls if o in mu["by_operator"]["nao"]]
        rows = [[cls[o], o.replace("_", r"\_"), str(mu["by_operator"]["nao"][o]["n"]), f(mu["by_operator"]["nao"][o]["rate"])] for o in ops]
        h = ["Class", "Operator", "$n$", "Rejected"]
        tex("app_operator", "Rejection rate of mutants by operator under NAO (RQ2).", "tab:app-operator", h, rows, "llrr")
        L += ["## Appendix: rejection by operator", ""] + mdtable(h, rows) + [""]
    gr = load("design_parameters", "nao_grid.json")
    if gr:
        rows = []
        for al in (0.01, 0.001, 0.0001):
            for K in sorted({g["K"] for g in gr["grid"]}):
                row = [f"{al:g}", str(max(K, 1))]
                for N in (1024, 4096, 16384):
                    g = next(x for x in gr["grid"] if x["alpha"] == al and x["K"] == K and x["shots"] == N)
                    row += [f(g["fpr"], 4), f(g["recall"])]
                rows.append(row)
        h = [r"$\alpha$", "$K$", "FPR 1024", "Recall 1024", "FPR 4096", "Recall 4096", "FPR 16384", "Recall 16384"]
        tex("app_grid", "Parameter study of NAO ($K = 1$ is NAO-1).", "tab:app-grid", h, rows, "llrrrrrr")
        L += ["## Appendix: NAO parameter grid", ""] + mdtable(h, rows) + [""]
    bs = load("rq1_bootstrap_stability", "results.json")
    if bs:
        rows = [[k, f"{f(v['tau_rel_sd_median'])} ({f(v['tau_rel_sd_max'])})", f(v["cond_fpr_mean"], 5),
                 f"{f(v['cond_fpr_p5_min'], 5)}--{f(v['cond_fpr_p95_max'], 5)}"] for k, v in bs["summary"].items()]
        h = ["$B$", "Relative s.d. of threshold, median (max)", "Implied FPR, mean", "Implied FPR, 5th--95th percentile range"]
        tex("rq1_bootstrap", r"Monte Carlo stability of the bootstrap threshold at $\alpha = 10^{-3}$.", "tab:rq1-bootstrap", h, rows, "rrrr")
        L += ["## RQ1 Bootstrap stability", ""] + mdtable(h, rows) + [""]
    sp = load("supplementary", "results.json")
    if sp:
        pb = sp["rq1_per_program"]["per_benchmark"]
        rows = [[PRETTY[b]] + [f"{pb[k][b]['fp']} [{f(pb[k][b]['wilson'][0], 4)}, {f(pb[k][b]['wilson'][1], 4)}]"
                               for k in ("nao_a0.001", "nao1_a0.001", "nao_a0.01", "nao1_a0.01")] for b in pb["nao_a0.001"]]
        h = ["Benchmark", r"NAO, $\alpha = 10^{-3}$", r"NAO-1, $\alpha = 10^{-3}$", r"NAO, $\alpha = 10^{-2}$", r"NAO-1, $\alpha = 10^{-2}$"]
        tex("app_fpr_program", "False positives per core benchmark over 1,000 correct runs, with 95% Wilson intervals (RQ1).", "tab:app-fpr-program", h, rows, "lrrrr")
        L += ["## Appendix: false positives per program", ""] + mdtable(h, rows) + [""]
        rows = []
        for m, mn in (("zne_linear", "ZNE linear"), ("zne_quadratic", "ZNE quadratic")):
            for nl, c in sp["rq3_triggered_by_noise"][m].items():
                rows.append([mn, nl, str(c["TP"]), str(c["FN"]), str(c["FP"]), str(c["TN"]), f(c["bug_recall"]), f(c["bug_precision"]), f(c["noise_recall"])])
        h = ["Method", "Noise", "TP", "FN", "FP", "TN", "Bug rec.", "Bug prec.", "Noise rec."]
        tex("rq3_by_noise", "Attribution of the cases rejected by NAO, by noise level (positive class: Bug).", "tab:rq3-by-noise", h, rows, "llrrrrrrr")
        L += ["## RQ3 Attribution of rejected cases by noise level", ""] + mdtable(h, rows) + [""]
    tc = load("taxonomy_compilation", "summary.json")
    if tc:
        A = tc["all"]; tg = ("synthetic", "cx_native", "cz_native")
        def cell(fam, keys, t):
            ops = A[fam][t]["by_operator"]; n = sum(ops[k]["n"] for k in keys); a = sum(ops[k]["adds_physical_ops"] for k in keys)
            q = sum(ops[k]["adds_2q"] for k in keys)
            return f"{a}/{n}" + (f" ({q})" if fam == "noise_neutral" and a else "")
        groups = [("noise_amplifying", "Noise-amplifying", "cx_pair", ["cx_pair"]), ("noise_amplifying", "", r"inverse\_pair\_1q, $Y$", ["inverse_pair_1q:y"]),
                  ("noise_amplifying", "", r"inverse\_pair\_1q, $Z, S, T$", ["inverse_pair_1q:z", "inverse_pair_1q:s", "inverse_pair_1q:t"]),
                  ("noise_amplifying", "", "rotation_pair", ["rotation_pair"]), ("noise_neutral", "Noise-neutral", "commute_swap", ["commute_swap"]),
                  ("noise_neutral", "", "cx_to_h_cz_h", ["cx_to_h_cz_h"]), ("noise_neutral", "", "x_to_h_z_h", ["x_to_h_z_h"])]
        rows = [[lab, op.replace("_", r"\_") if "$" not in op else op] + [cell(fam, keys, t) for t in tg] for fam, lab, op, keys in groups]
        h = ["Class", "Operator", "Synthetic", "CX-native", "CZ-native"]
        tex("rq2_compiled", "Semantics-preserving mutants whose compiled circuit contains more physical operations than the compiled original (in parentheses: with additional two-qubit gates).", "tab:rq2-compiled", h, rows, "llrrr")
        L += ["## RQ2 Physical criterion after compilation", ""] + mdtable(h, rows) + [""]
    hm = load("hardware_analysis", "real", "hm", "analysis.json")
    if hm and "a" in hm and "c" in hm:
        S, C = hm["a"]["summary"], hm["c"]
        def tot(part, o, kind):
            return sum(v[o][kind] for v in C[part].values()), sum(v[o]["n_" + kind] for v in C[part].values())
        rows = []
        for o, n in (("nao", "NAO"), ("nao1", "NAO-1"), ("spc", "SPC envelope"), ("fixed_0.005", r"Fixed $\tau = 0.005$"), ("fixed_0.02", r"Fixed $\tau = 0.02$"), ("fixed_0.05", r"Fixed $\tau = 0.05$")):
            small = ["%d/%d" % (S[o][k]["k"], S[o][k]["n"]) for k in ("correct", "fault", "noise_amplifying", "noise_neutral")]
            if o in next(iter(C["large"].values())):
                big = ["%d/%d" % tot(p, o, k) for p in ("large", "real_bugs") for k in ("correct", "fault")]
            else:
                big = ["--"] * 4
            rows.append([n] + small + big)
        h = ["Oracle", "Small: correct rejected", "Small: faults detected", "Small: noise-ampl. rejected", "Small: noise-neutral rejected",
             "Larger: correct rejected", "Larger: faults detected", "Real bugs: fixed rejected", "Real bugs: buggy detected"]
        tex("hm", "Mini campaign HM on ibm_kingston: rejected / tested runs (small programs: first job; larger programs and real bugs: third job).", "tab:hm", h, rows, "lrrrrrrrr")
        L += ["## Hardware: mini campaign HM", ""] + mdtable(h, rows) + [""]

def main():
    L = ["# Results summary", "", "Generated by `experiments/make_tables.py` from `results/*/`. LaTeX versions: `results/tables/`.",
         "Oracle names: NAO = noise-aware oracle (key `nao`, K = 8); NAO-1 = single-reference variant (key `nao1`). "
         "The README lists every key used in the result files.", ""]

    v = load("rq2_equivalence", "summary.json")
    if v:
        rows = []
        for k, x in sorted(v["census_by_operator"].items()):
            fam, op = k.split("|")
            rows.append([CLASS.get(fam, fam), op.replace("_", r"\_"), str(x["draws"]), f(x["rate_zero_input"]), f(x["rate_multi_input"])])
        h = ["Family", "Operator", "Draws", r"Equiv. $|0\rangle$ $\downarrow$", r"Equiv. 4 inputs $\downarrow$"]
        tex("rq2_census", "Fraction of raw mutation draws that are semantically equivalent to the original program.", "tab:rq2-census", h, rows, "llrrr")
        s = v["faults"]; p = v["placement"]
        L += ["## RQ2 Equivalent mutants", ""] + mdtable(h, [[c.replace(r"\_", "_") for c in r] for r in rows]) + ["",
              f"Injected faults: {s['equivalent_zero_input']}/{s['applicable']} applicable injections are equivalent on |0...0> "
              f"({s['unitary_changed_output_unchanged']} change the unitary but not the measured output); {s['equivalent_multi_input']} remain equivalent over 4 inputs.",
              f"Placement: inserting CX/T/X after the final measurements yields {p['after_measurement']['equivalent']}/{p['after_measurement']['draws']} equivalent mutants; before them {p['before_measurement']['equivalent']}/{p['before_measurement']['draws']} ({f(p['before_measurement']['rate'])}).", ""]

    s = load("rq1_soundness", "summary.json")
    if s:
        order = ["nao_a0.001", "nao_a0.01", "nao1_a0.001", "nao1_a0.01", "spc_envelope", "qoin_style",
                 "fixed_0.1", "fixed_0.05", "fixed_0.02", "fixed_0.01", "fixed_0.005"]
        rows = [[NAME[o], f"{s['by_oracle'][o]['fp']}/{s['by_oracle'][o]['tests']}", f(s['by_oracle'][o]['fpr'], 4),
                 f"[{f(s['by_oracle'][o]['fpr_cluster_ci'][0], 4)}, {f(s['by_oracle'][o]['fpr_cluster_ci'][1], 4)}]"] for o in order]
        h = ["Oracle", "FP / tests", r"FPR $\downarrow$", "95\\% CI"]
        tex("rq1_soundness", "False-positive rate on correct programs (9 benchmarks, 5 noise levels, 40 calibration windows, 5 runs each); CI by cluster bootstrap over windows.", "tab:rq1-soundness", h, rows, "lrrr")
        L += ["## RQ1 Soundness", ""] + mdtable(h, rows) + [""]
        th = s["thresholds"]; rows = []
        for k in ("bell|medium", "ghz_3|medium", "bell_8|medium", "ghz_8|high", "qaoa_8|medium", "hwe_6|medium"):
            b, n = k.split("|"); x = th[k]
            rows.append([f"{PRETTY[b]}, {n}", f(x["nao1_a0.001"], 4), f(x["tau_chi2_a0.001"], 4), f(x["nao_a0.001"], 4), f(x["spc_envelope"], 4)])
        h = ["Cell", "NAO-1 (bootstrap)", r"$\chi^2$ closed form", "NAO (K = 8)", "SPC envelope"]
        tex("rq1_thresholds", r"Median acceptance thresholds at $\alpha = 10^{-3}$.", "tab:rq1-thresholds", h, rows, "lrrrr")
        disp = sorted(x["median_dispersion_ratio"] for x in s["drift"].values())
        L += mdtable(h, rows) + ["", f"Dispersion ratio (observed / predicted calibration spread), median over 45 cells: {f(disp[len(disp)//2])} (range {f(disp[0])} to {f(disp[-1])}).", ""]

    sn = load("rq1_snapshots", "results.json")
    if sn:
        rows = []
        for bn in sorted({x["backend"] for x in sn["soundness"]}):
            for pol in ("fixed", "random"):
                S = [x for x in sn["soundness"] if x["backend"] == bn and x["policy"] == pol]; n = sum(x["tests"] for x in S)
                fpr = lambda o: sum(x["fpr"][o] * x["tests"] for x in S) / n
                rows.append([bn.replace("_", r"\_"), pol, f(fpr("nao")), f(fpr("nao1")), f(fpr("spc_envelope")), f(fpr("fixed_0.005")), f(fpr("fixed_0.05")),
                             f(fpr("fixed_0.1")), f(sum(x["drift_flag_rate"] for x in S) / len(S), 2)])
        h = ["Backend", "Compilation", r"NAO $\downarrow$", r"NAO-1 $\downarrow$", r"SPC $\downarrow$", r"Fixed 0.005 $\downarrow$", r"Fixed 0.05 $\downarrow$",
             r"Fixed 0.1 $\downarrow$", "Drift flagged"]
        tex("rq1_snapshots", "False-positive rate on device-snapshot backends under fixed and per-run compilation.", "tab:rq1-snapshots", h, rows, "llrrrrrrr")
        L += ["### Device-snapshot backends", ""] + mdtable(h, [[c.replace(r"\_", "_") for c in r] for r in rows]) + [""]

    fa = load("rq4_faults", "summary.json")
    if fa:
        order = ["nao", "nao1", "spc_envelope", "fixed_0.01", "fixed_0.02", "fixed_0.05", "fixed_0.1"]
        rows = [[NAME[o]] + [str(fa["overall"][o][k]) for k in ("tp", "fn", "fp", "tn")] + [f(fa["overall"][o][k]) for k in ("recall", "precision", "f1")]
                + [f(fa["overall"][o]["fpr"], 4), f"{fa['equivalent_injections'][o]['rejected']}/{fa['equivalent_injections'][o]['n']}"] for o in order]
        h = ["Oracle", "TP", "FN", "FP", "TN", r"Recall $\uparrow$", r"Precision $\uparrow$", r"F1 $\uparrow$", r"FPR $\downarrow$", r"Equiv. rejected $\downarrow$"]
        tex("rq4_faults", "Detection of non-equivalent injected faults (negatives: correct runs). Equivalent injections are reported separately.", "tab:rq4-faults", h, rows, "lrrrrrrrrr")
        L += ["## RQ4 Fault detection", "", "### Injected faults", ""] + mdtable(h, rows) + [""]
        noises = list(next(iter(fa["by_noise"].values())).keys())
        rows = [[NAME[o]] + [f(fa["by_noise"][o][n]["recall"]) for n in noises] for o in order]
        h = ["Oracle"] + [n.replace("_", " ") for n in noises]
        tex("rq4_recall_by_noise", "Recall on non-equivalent injected faults by noise level.", "tab:rq4-recall-noise", h, rows, "l" + "r" * len(noises))
        L += mdtable(h, rows) + [""]
        rows = [[k, f(x["nao1"], 5), f(x["nao_K8"], 5)] for k, x in fa["min_detectable_delta_4096"].items()]
        h = ["Outcomes $d$", "NAO-1", "NAO (K = 8)"]
        tex("rq4_mdd", r"Minimum detectable JSD at 4096 shots (80\% power, $\alpha = 10^{-3}$).", "tab:rq4-mdd", h, rows, "rrr")
        L += mdtable(h, rows) + [""]

    mu = load("rq2_mutation", "summary.json")
    if mu:
        rows = []
        for o in ("nao", "nao1", "spc_envelope"):
            r = mu["rejection_by_class"][o]; fm = mu["rejection_by_family"][o]
            rows.append([NAME[o]] + [f"{r[c]['k']}/{r[c]['n']} ({f(r[c]['rate'])})" for c in ("semantic", "noise_amplifying", "noise_neutral")]
                        + [f(fm["semantic_additive"]["rate"]), f(fm["semantic_subtractive"]["rate"])])
        h = ["Oracle", r"Semantic $\uparrow$", "Noise-amplifying", r"Noise-neutral $\downarrow$", "Sem. additive", "Sem. subtractive"]
        tex("rq2_mutation", "Rejection rate by taxonomy class (9 benchmarks, 3 noise levels, 15 mutants per family).", "tab:rq2-mutation", h, rows, "lrrrrr")
        L += ["## RQ2 Mutation analysis and taxonomy", ""] + mdtable(h, rows) + [""]
        rows = []
        es = (load("supplementary", "results.json") or {}).get("rq2_effect_sizes", {}).get("core", {})
        for scope, v in mu["taxonomy_tests"].items():
            for p, x in v.items():
                a, b = p.split("_vs_")
                ci = es.get(scope, {}).get(p)
                dl = f(x["cliffs_delta"], 2) + (f" [{f(ci['ci'][0], 2)}, {f(ci['ci'][1], 2)}]" if ci else "")
                rows.append([scope, f"{CLASS[a]} vs {CLASS[b]}", f"{f(x['rate_a'])} vs {f(x['rate_b'])}", f"{x['fisher_p']:.1e}", f"{x['cmh_p']:.1e}", f"{x['cells_higher']}/{x['cells_equal']}/{x['cells_lower']}", f"{x['signflip_p']:.1e}", dl])
        for scope, v in mu["family_tests"].items():
            for p, x in v.items():
                rows.append([scope, "Additive vs subtractive", f"{f(x['rate_a'])} vs {f(x['rate_b'])}", f"{x['fisher_p']:.1e}", f"{x['cmh_p']:.1e}", f"{x['cells_higher']}/{x['cells_equal']}/{x['cells_lower']}", f"{x['signflip_p']:.1e}", f(x["cliffs_delta"], 2)])
        h = ["Noise", "Comparison", "Rejection rates", "Fisher $p$", "CMH $p$", "Cells $>$/$=$/$<$", "Sign-flip $p$", r"Cliff's $\delta$"]
        tex("rq2_taxonomy_tests", r"Taxonomy validation. $H_0$: equal rejection probability. Fisher exact test per mutant; Cochran-Mantel-Haenszel test stratified by cell (conditions on the shared calibration window); cell-level sign-flip test. Cliff's $\delta$ on $d/\tau$. Bonferroni over 3 class pairs.", "tab:rq2-taxonomy", h, rows, "llrrrrrr")
        L += mdtable(h, rows) + [""]

    sc = load("rq4_scaling", "results.json")
    if sc:
        rows = [[r["circuit"].replace("_", "-").upper(), str(r["qubits"]), f(r["shots_per_outcome"], 3)]
                + [f(r["fpr"][k]) for k in ("nao", "nao_marginal", "spc_envelope")] + [f(r["recall"][k]) for k in ("nao", "nao_marginal", "spc_envelope")] for r in sc]
        h = ["Circuit", "Qubits", "Shots/outcome", "FPR NAO", "FPR marg.", "FPR SPC", "Recall NAO", "Recall marg.", "Recall SPC"]
        tex("rq4_scaling", r"Scaling to 16 qubits ($\eta = 0.005$, 4096 shots).", "tab:rq4-scaling", h, rows, "lrrrrrrrr")
        L += ["## RQ4 Scaling", ""] + mdtable(h, rows) + [""]

    pr = load("design_parameters", "spc_envelope_grid.json"); pn = load("design_parameters", "nao_grid.json")
    if pr and pn:
        rows = []
        for lab, S in (("SPC envelope", pr["sobol"]["config_level"]), ("NAO", pn["sobol"])):
            for r, s in S.items():
                for k, x in s.items():
                    if k != "_variance":
                        rows.append([lab, r.upper() if r == "fpr" else r, k.replace("_", r"\_"), f(x["S1"]), f(x["ST"])])
        h = ["Oracle", "Response", "Parameter", "$S_1$", "$S_T$"]
        tex("design_sobol", "First-order ($S_1$) and total ($S_T$) Sobol indices of oracle parameters.", "tab:design-sobol", h, rows, "lllrr")
        sound = [r for r in pr["configs"] if r["fpr"] == 0]
        L += ["## Parameter selection study (design)", ""] + mdtable(h, [[c.replace(r"\_", "_") for c in r] for r in rows]) + ["",
              f"SPC envelope: {len(sound)}/{len(pr['configs'])} configurations have zero false positives.", ""]

    at = load("rq3_attribution", "summary.json")
    if at:
        rows = []
        for scope in ("all_cases", "triggered_only"):
            for m in ("zne_linear", "zne_quadratic", "reexecution_triage"):
                x = at[scope][m]
                rows.append([scope.replace("_", " "), NAME[m], f(x["accuracy"]), f(x["kappa"]), f(x["bug_precision"]), f(x["bug_recall"]), f(x["noise_recall"])])
        h = ["Cases", "Method", r"Accuracy $\uparrow$", r"$\kappa$ $\uparrow$", r"Bug prec. $\uparrow$", r"Bug recall $\uparrow$", r"Noise recall $\uparrow$"]
        tex("rq3_attribution", f"Bug-noise attribution with exact labels (all cases n = {at['all_cases']['n']}; triggered by NAO n = {at['triggered_only']['n']}).", "tab:rq3-attribution", h, rows, "llrrrrr")
        L += ["## RQ3 Attribution", ""] + mdtable(h, rows) + [""]

    dr = load("rq5_drift_cost", "results.json"); co = load("rq5_drift_cost", "cost_model.json")
    if dr:
        rows = [[PRETTY[x["benchmark"]], f(x["r"], 2), f(x["fpr"]["nao"]), f(x["fpr_with_sentinel"]["nao"]), f(x["fpr"]["spc_envelope"]), f(x["recalibration_rate"], 2)] for x in dr["between_window"]]
        h = ["Benchmark", "$r$", "FPR NAO", "FPR NAO + sentinel", "FPR SPC", "Recalibrated"]
        tex("rq5_drift", r"False-positive rate when testing at $\eta_0(1+r)$ after calibration at $\eta_0 = 0.005$, without recalibration and with a one-run sentinel at $\alpha_s = 10^{-3}$ (drift figure, panel a).", "tab:rq5-drift-one-run", h, rows, "lrrrrr")
        L += ["## RQ5 Drift and cost", ""] + mdtable(h, rows) + [""]
        if co:
            keys = [k for k in co["campaign_executions"][0] if k != "M_tests"]
            rows = [[str(c["M_tests"])] + [str(c[k]) for k in keys] for c in co["campaign_executions"]]
            h = ["Tests M"] + keys
            tex("rq5_cost", "Circuit executions per test campaign.", "tab:rq5-cost", h, rows, "r" + "r" * len(keys))
            L += mdtable(h, rows) + [""]

    am = load("rq2_amplification", "summary.json")
    if am:
        rows = [[x["noise"], x["kind"].replace("_", " "), str(x["dose"]), f"{x['rejected']}/{x['n']}", f(x["rate"]), f(x["mean_predicted"])] for x in am["by_dose"]]
        h = ["Noise", "Pairs", "$k$", "Rejected", "Rate", "Predicted"]
        tex("rq2_dose", f"Rejection of noise-amplifying mutants by dose under NAO, with the power-model prediction (Brier score {am['brier']:.3f}; constant predictor {am['brier_constant']:.3f}).", "tab:rq2-dose", h, rows, "llrrrr")
        L += ["## RQ2 Noise-amplification dose", ""] + mdtable(h, rows) + ["", f"Brier score {am['brier']:.4f} (constant predictor {am['brier_constant']:.4f}).", ""]

    ze = load("rq3_extrapolation", "summary.json")
    if ze:
        rows = []
        for env, v in ze.items():
            for scope, x in v.items():
                for m in ("linear", "quadratic", "exponential"):
                    y = x[m]
                    rows.append([f"{env}, {scope.replace('_', ' ')}", m, f(y["accuracy"]), f(y["kappa"]), f(y["bug_recall"]), f(y["noise_recall"])])
        h = ["Data", "Extrapolation", r"Accuracy $\uparrow$", r"$\kappa$ $\uparrow$", r"Bug recall $\uparrow$", r"Noise recall $\uparrow$"]
        tex("rq3_extrapolation", "Attribution by extrapolation method on identical executions.", "tab:rq3-extrapolation", h, rows, "llrrrr")
        L += ["## RQ3 Choice of extrapolation", ""] + mdtable(h, rows) + [""]

    gt = load("rq4_global_threshold", "summary.json")
    if gt:
        rows = [[{"nao": "NAO", "noise_calibrated": "Noise-calibrated global threshold"}[o], f(x["recall"]), f(x["precision"]), f(x["f1"]), f(x["fpr"], 4),
                 f"{x['equivalent_rejected'][0]}/{x['equivalent_rejected'][1]}"] for o, x in gt["overall"].items()]
        h = ["Oracle", r"Recall $\uparrow$", r"Precision $\uparrow$", r"F1 $\uparrow$", r"FPR $\downarrow$", r"Equivalent rejected $\downarrow$"]
        tex("rq4_global", "NAO and a noise-calibrated global threshold (after Fortz et al.) on identical executions.", "tab:rq4-global", h, rows, "lrrrrr")
        L += ["## RQ4 Noise-calibrated global threshold", ""] + mdtable(h, rows) + [""]

    sp = load("rq1_sparsity", "results.json")
    if sp:
        rows = []
        for dd in sorted({r["d"] for r in sp}):
            row = [str(dd), f"{4096 / dd:g}"]
            for sh in ("flat", "skewed", "headtail"):
                r = next((x for x in sp if x["d"] == dd and x["shape"] == sh), None)
                row += [f(r["fpr"]["nao"]["0.01"], 4), f(r["fpr"]["nao1"]["0.01"], 4), f(r["fpr"]["nao"]["0.05"], 3)] if r else ["", "", ""]
            rows.append(row)
        h = ["Outcomes $d$", "Shots/outcome", r"Flat: NAO, $\alpha=0.01$", r"Flat: NAO-1, $\alpha=0.01$", r"Flat: NAO, $\alpha=0.05$",
             r"Skewed: NAO, $\alpha=0.01$", r"Skewed: NAO-1, $\alpha=0.01$", r"Skewed: NAO, $\alpha=0.05$",
             r"Head-tail: NAO, $\alpha=0.01$", r"Head-tail: NAO-1, $\alpha=0.01$", r"Head-tail: NAO, $\alpha=0.05$"]
        tex("rq1_sparsity", "Observed false-positive rate against the nominal level as the support grows.", "tab:rq1-sparsity", h, rows, "rrrrrrrrrrr")
        L += ["## RQ1 Calibration under sparse supports", ""] + mdtable(h, rows) + [""]

    ab = load("rq4_ablation", "summary.json")
    if ab:
        REF = {"A": ("Ideal (noiseless)", r"Fixed $\tau = 0.02$"), "B": ("1 self-baseline run", r"Fixed $\tau = 0.02$"),
               "C": ("1 self-baseline run", "SPC envelope"), "D": ("1 self-baseline run", "Null quantile (NAO-1)"),
               "E": ("8 pooled runs", "Null quantile"), "F": ("8 pooled runs", "Null quantile + drift test (NAO)")}
        G = [("synthetic", "Synthetic noise"), ("snapshots_random_compilation", "Snapshots, recompilation")]
        rows = []
        for k in "ABCDEF":
            row = [k, *REF[k]]
            for g, _ in G:
                x = ab["results"][g][k]
                row += [f(x["fpr"], 4), f(x["recall"]), f(x["f1"])]
            rows.append(row)
        h = ["", "Reference", "Threshold"] + [f"{n}: {m}" for _, n in G for m in (r"FPR $\downarrow$", r"Recall $\uparrow$", r"F1 $\uparrow$")]
        tex("rq4_ablation", "Ablation of NAO: one component added per row, on identical executions.", "tab:rq4-ablation", h, rows, "lllrrrrrr")
        L += ["## RQ4 Ablation", ""] + mdtable(h, rows) + [""]

    sp = load("rq5_sentinel", "results.json")
    if sp:
        P = ["none", "sentinel_m1_a0.001", "sentinel_m2_a0.01", "co_scheduled"]
        rows = [[PRETTY[x["benchmark"]], f(x["r"], 2)] + [f"{f(x['fpr'][p])} ({f(x['recalibration_rate'][p], 2)})" for p in P] for x in sp]
        h = ["Benchmark", "$r$", "None", "Sentinel (1, 1e-3)", "Sentinel (2, 1e-2)", "Co-scheduled"]
        tex("rq5_policies", r"FPR (recalibration rate) of NAO under recalibration policies after drift to $\eta_0(1+r)$.", "tab:rq5-drift", h, rows, "lrrrrr")
        L += ["## RQ5 Recalibration policies", ""] + mdtable(h, rows) + [""]

    es = load("ext_suite", "summary.json")
    if es:
        names = {"nao": "NAO", "nao1": "NAO-1", "spc_envelope": "SPC envelope", "fixed_0.01": r"Fixed $\tau=0.01$", "fixed_0.05": r"Fixed $\tau=0.05$",
                 "fixed_0.1": r"Fixed $\tau=0.1$", "qoin_style": "QOIN-style filter", "global_noise_calibrated": "Noise-calibrated global threshold"}
        rows = [[names[o], f(es["soundness"][o]["fpr"], 4), f"{es['soundness'][o]['programs_with_fpr_above_0.05']}/{es['programs']}",
                 f(es["detection"][o]["recall"]), f(es["detection"][o]["precision"]), f(es["detection"][o]["f1"])] for o in names]
        h = ["Oracle", "FPR", r"Unsound programs $\downarrow$", r"Recall $\uparrow$", r"Precision $\uparrow$", r"F1 $\uparrow$"]
        tex("ext_suite", f"Extended suite of {es['programs']} QASMBench programs ({es['n_faults']} non-equivalent faults).", "tab:ext-suite", h, rows, "lrrrrr")
        L += ["## Extended suite (QASMBench)", ""] + mdtable(h, rows) + [""]
        if "mutation" in es:
            rows = [[names[o]] + [f"{v['k']}/{v['n']} ({f(v['k'] / max(v['n'], 1))})" for v in es["mutation"][o].values()] for o in ("nao", "spc_envelope")]
            h = ["Oracle", r"Semantic $\uparrow$", "Noise-amplifying", r"Noise-neutral $\downarrow$"]
            L += mdtable(h, rows) + [""]

    rb = load("rq4_real_bugs", "summary.json"); rc = load("rq4_real_bugs", "catalogue.json")
    if rb and rc:
        rows = [[c["id"].replace("_", r"\_"), c["kind"], str(c["qubits"]), f(c["ideal_jsd"]), f(rb["by_bug"][c["id"]]["nao"], 2),
                 f(rb["by_bug"][c["id"]]["spc_envelope"], 2), f(rb["by_bug"][c["id"]]["fixed_0.05"], 2)] for c in rc]
        ov = rb["overall"]
        rows.append(["Observable bugs: recall / precision / F1", "", "", ""] +
                    [f"{f(ov[o]['recall'])} / {f(ov[o]['precision'])} / {f(ov[o]['f1'])}" for o in ("nao", "spc_envelope", "fixed_0.05")])
        rows.append(["FPR on runs of the fixed programs", "", "", ""] + [f(ov[o]["fpr"], 4) for o in ("nao", "spc_envelope", "fixed_0.05")])
        h = ["Bug", "Kind", "Qubits", "Ideal JSD", "NAO", "SPC", r"Fixed $\tau=0.05$"]
        tex("rq4_real_bugs", "Real bugs from Bugs4Q: detection rate of the buggy program.", "tab:rq4-realbugs", h, rows, "llrrrrr")
        L += ["## RQ4 Real bugs (Bugs4Q)", ""] + mdtable(h, [[c.replace(r"\_", "_") for c in r] for r in rows]) + [""]
        rows = [[names.get(o, o), f(x["recall"]), f(x["precision"]), f(x["f1"]), f(x["fpr"], 4)] for o, x in rb["overall"].items()]
        L += mdtable(["Oracle", "Recall", "Precision", "F1", "FPR"], rows) + [""]

    extra_tables(L)

    hw = load("hardware_analysis", "real", "analysis.json")
    L += ["## Hardware", "", "Real-device analysis: " + ("see `results/hardware_analysis/real/`." if hw else "pending (see docs/HARDWARE.md); dry-run results on a fake Heron backend are in `results/hardware_analysis/dryrun/`."), ""]
    open(os.path.join(RESULTS, "SUMMARY.md"), "w").write("\n".join(L))
    print("\n".join(L)[:3000])


if __name__ == "__main__":
    main()
