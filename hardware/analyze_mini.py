# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Offline analysis of campaign HM (run_mini.py).

  python hardware/analyze_mini.py            # real-hardware data
  python hardware/analyze_mini.py --dry-run  # fake-backend dry run

Step a (day 1), per benchmark and pooled:
  RQ1  false positives on the correct test runs of NAO (K = 8, drift test),
       NAO-1 (single reference), the SPC envelope (K = 9 calibration runs,
       single reference) and fixed thresholds against the exact ideal output;
       divergence of correct runs from the ideal output.
  RQ2  rejection of semantic, noise-amplifying and noise-neutral mutants.
  RQ4  recall on non-equivalent injected faults and semantic mutants.
  RQ3  linear ZNE attribution (scales 1 and 3) of the semantic and
       noise-amplifying mutants, against their exact labels.
Step b (later), judged against the calibration of step a:
  RQ5  sentinel (two runs pooled, alpha_s = 1e-2) decision; false positives of
       the stale calibration on the correct runs; detection of the faults with
       the stale reference and with a reference rebuilt from the sentinel runs.
Writes results/hardware_analysis/<real|dryrun>/hm/{analysis.json, REPORT.txt, hm_table.tex}.
"""
from __future__ import annotations
import argparse, collections, datetime as dt, glob, json, os, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from nisq_framework import compute_threshold, spc_envelope_threshold, jsd, pooled, attribute_from_counts, ideal_distribution, get_benchmark
from nisq_framework.stats import cohen_kappa, wilson

RAW = os.path.join(ROOT, "results", "hardware_raw", "hm")
ALPHA, ALPHA_S = 1e-3, 1e-2
FIXED = (0.005, 0.02, 0.05)
ORACLES = ["nao", "nao1", "spc"] + [f"fixed_{t:g}" for t in FIXED]
NAMES = {"nao": "NAO", "nao1": "NAO-1", "spc": "SPC envelope", **{f"fixed_{t:g}": f"Fixed tau = {t:g}" for t in FIXED}}


def latest(step, dry):
    rs = []
    for p in sorted(glob.glob(os.path.join(RAW, step, "*", "raw.json"))):
        r = json.load(open(p))
        if bool(r.get("dry_run")) == dry:
            rs.append(r)
    return rs[-1] if rs else None


def nq(c):
    return len(next(iter(c)))


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "ci": list(wilson(int(k), int(n))) if n else [float("nan")] * 2}


def calibrate(items, b, shots):
    cal = [it["counts"] for it in items if it["role"] == "calibration"]
    single = next(it["counts"] for it in items if it["role"] == "single_reference")
    rng = np.random.default_rng(0)
    rep = compute_threshold(cal, shots, ALPHA, "pooled", n_qubits=nq(cal[0]), rng=rng)
    tau1 = compute_threshold([], shots, ALPHA, "single", n_qubits=nq(cal[0]), reference_counts=single, rng=rng).tau
    ideal = ideal_distribution(get_benchmark(b)[0])
    O = {"nao": (pooled(cal), rep.tau), "nao1": (single, tau1), "spc": (single, spc_envelope_threshold(cal + [single]))}
    for t in FIXED:
        O[f"fixed_{t:g}"] = (ideal, t)
    return O, rep, cal, ideal


def step_a(r):
    shots = r["shots"]; by = collections.defaultdict(list)
    for it in r["items"]:
        by[it["benchmark"]].append(it)
    per, calib = {}, {}
    tot = {o: collections.Counter() for o in ORACLES}
    zne = []
    for b, items in by.items():
        O, rep, cal, ideal = calibrate(items, b, shots)
        calib[b] = {"tau_nao": rep.tau, "tau_null": rep.tau_null, "drift_detected": bool(rep.drift_detected),
                    "homogeneity_p": rep.homogeneity_p, "dispersion_ratio": rep.dispersion_ratio}
        rej = lambda o, c: jsd(O[o][0], c) > O[o][1]
        groups = {"correct": [it for it in items if it["role"] == "correct_test"],
                  "fault": [it for it in items if it["role"] in ("injected", "mutant") and it["truth"] == "fault"],
                  "noise_amplifying": [it for it in items if it.get("family") == "noise_amplifying" and it["role"] == "mutant"],
                  "noise_neutral": [it for it in items if it.get("family") == "noise_neutral" and it["role"] == "mutant"]}
        row = {"ideal_gap": [jsd(ideal, it["counts"]) for it in groups["correct"]]}
        for o in ORACLES:
            row[o] = {g: [int(rej(o, it["counts"])) for it in v] for g, v in groups.items()}
            for g, v in row[o].items():
                tot[o][g + "_k"] += sum(v); tot[o][g + "_n"] += len(v)
        row["fault_detail"] = [{"role": it["role"], "operator": it.get("operator"), "family": it.get("family"), "ideal_jsd": it.get("ideal_jsd"),
                                "nao": int(rej("nao", it["counts"])), "divergence": jsd(O["nao"][0], it["counts"])} for it in groups["fault"]]
        per[b] = row
        # attribution: scale 1 of P = single reference; scale 1 of a case = its mutant run
        s1 = {it["case"]: it for it in items if it.get("scale") == 1}
        s3 = {it["case"]: it for it in items if it.get("scale") == 3}
        for case in sorted(set(s1) & set(s3) - {"P"}):
            if "P" not in s3:
                break
            res = attribute_from_counts([s1["P"]["counts"], s3["P"]["counts"]], [s1[case]["counts"], s3[case]["counts"]], (1, 3), "linear",
                                        alpha=1e-2, n_outcomes=2 ** nq(s1["P"]["counts"]))
            zne.append({"benchmark": b, "case": case, "family": s1[case]["family"], "operator": s1[case]["operator"],
                        "truth": "bug" if s1[case]["truth"] == "fault" else "noise", "attribution": res["attribution"],
                        "r0": res["r0"], "null_q": res["null_q"], "rejected_by_nao": bool(rej("nao", s1[case]["counts"]))})
    summ = {o: {g: rate(tot[o][g + "_k"], tot[o][g + "_n"]) for g in ("correct", "fault", "noise_amplifying", "noise_neutral")} for o in ORACLES}
    zs = {}
    if zne:
        yt = [z["truth"] for z in zne]; yp = [z["attribution"] for z in zne]
        zs = {"n": len(zne), "accuracy": float(np.mean([x == y for x, y in zip(yt, yp)])), "kappa": cohen_kappa(yt, yp),
              "bug_recall": rate(sum(x == y == "bug" for x, y in zip(yt, yp)), yt.count("bug")),
              "noise_recall": rate(sum(x == y == "noise" for x, y in zip(yt, yp)), yt.count("noise"))}
    gaps = [g for b in per for g in per[b]["ideal_gap"]]
    return {"per_benchmark": per, "calibration": calib, "summary": summ, "attribution": {"cases": zne, "summary": zs},
            "ideal_gap_range": [min(gaps), max(gaps)] if gaps else None}


def step_b(ra, rb):
    t = lambda s: dt.datetime.strptime(s, "%Y%m%d_%H%M%S")
    shots = rb["shots"]; out = {"gap_hours": round((t(rb["submitted"]) - t(ra["submitted"])).total_seconds() / 3600, 1), "per_benchmark": {}}
    A = collections.defaultdict(list); Bb = collections.defaultdict(list)
    for it in ra["items"]:
        A[it["benchmark"]].append(it)
    for it in rb["items"]:
        Bb[it["benchmark"]].append(it)
    tot = collections.Counter()
    for b, items in Bb.items():
        cal0 = [it["counts"] for it in A[b] if it["role"] == "calibration"]
        rng = np.random.default_rng(0)
        ref0 = pooled(cal0); tau0 = compute_threshold(cal0, shots, ALPHA, "pooled", n_qubits=nq(cal0[0]), rng=rng).tau
        sent = [it["counts"] for it in items if it["role"] == "sentinel"]
        ts = compute_threshold(cal0, len(sent) * shots, ALPHA_S, "pooled", n_qubits=nq(cal0[0]), rng=rng).tau
        d_sent = jsd(ref0, pooled(sent)); fired = d_sent > ts
        ref1 = pooled(sent); tau1 = compute_threshold(sent, shots, ALPHA, "pooled", n_qubits=nq(sent[0]), rng=rng).tau
        corr = [it["counts"] for it in items if it["role"] == "correct_test"]
        flt = [it["counts"] for it in items if it["role"] == "injected"]
        row = {"sentinel_divergence": d_sent, "sentinel_threshold": ts, "sentinel_fired": bool(fired),
               "stale_fp": sum(jsd(ref0, c) > tau0 for c in corr), "fresh_fp": sum(jsd(ref1, c) > tau1 for c in corr), "n_correct": len(corr),
               "stale_detected": sum(jsd(ref0, c) > tau0 for c in flt), "fresh_detected": sum(jsd(ref1, c) > tau1 for c in flt), "n_faults": len(flt),
               "reference_shift": jsd(ref0, pooled(sent + corr))}
        row["policy_fp"] = row["fresh_fp"] if fired else row["stale_fp"]
        row["policy_detected"] = row["fresh_detected"] if fired else row["stale_detected"]
        out["per_benchmark"][b] = row
        for k in ("stale_fp", "fresh_fp", "policy_fp", "n_correct", "stale_detected", "fresh_detected", "policy_detected", "n_faults"):
            tot[k] += row[k]
        tot["fired"] += int(fired)
    out["summary"] = {"sentinel_fired": int(tot["fired"]), "benchmarks": len(Bb),
                      "fpr_stale": rate(tot["stale_fp"], tot["n_correct"]), "fpr_fresh": rate(tot["fresh_fp"], tot["n_correct"]),
                      "fpr_sentinel_policy": rate(tot["policy_fp"], tot["n_correct"]),
                      "recall_stale": rate(tot["stale_detected"], tot["n_faults"]), "recall_fresh": rate(tot["fresh_detected"], tot["n_faults"]),
                      "recall_sentinel_policy": rate(tot["policy_detected"], tot["n_faults"])}
    return out


def judge_set(items, shots, ideal):
    """NAO, NAO-1 (first calibration run as single reference) and fixed thresholds
    against the exact ideal output; returns per-oracle rejection lists by role."""
    cal = [it["counts"] for it in items if it["role"] == "calibration"]
    rng = np.random.default_rng(0)
    rep = compute_threshold(cal, shots, ALPHA, "pooled", n_qubits=nq(cal[0]), rng=rng)
    t1 = compute_threshold([], shots, ALPHA, "single", n_qubits=nq(cal[0]), reference_counts=cal[0], rng=rng).tau
    O = {"nao": (pooled(cal), rep.tau), "nao1": (cal[0], t1), **{f"fixed_{t:g}": (ideal, t) for t in FIXED}}
    out = {o: {"correct": [], "fault": []} for o in O}
    for it in items:
        if it["role"] == "calibration":
            continue
        g = "correct" if it["truth"] == "no_fault" else "fault"
        for o, (ref, tau) in O.items():
            out[o][g].append(int(jsd(ref, it["counts"]) > tau))
    return out, {"tau_nao": rep.tau, "drift_detected": bool(rep.drift_detected), "outcomes_observed": len(pooled(cal))}


C_ORACLES = ["nao", "nao1"] + [f"fixed_{t:g}" for t in FIXED]


def step_c(r):
    from nisq_framework.real_bugs import real_bug_pairs
    fixed = {"rb_" + p["id"]: p["fixed"] for p in real_bug_pairs()}
    by = collections.defaultdict(list)
    for it in r["items"]:
        by[it["benchmark"]].append(it)
    res = {"large": {}, "real_bugs": {}}
    tot = {g: {o: collections.Counter() for o in C_ORACLES} for g in res}
    for b, items in by.items():
        g = "real_bugs" if b.startswith("rb_") else "large"
        ideal = ideal_distribution(fixed[b] if g == "real_bugs" else get_benchmark(b)[0])
        v, cal = judge_set(items, r["shots"], ideal)
        res[g][b] = {**cal, **{o: {k: sum(x) for k, x in v[o].items()} | {"n_correct": len(v[o]["correct"]), "n_fault": len(v[o]["fault"])} for o in C_ORACLES}}
        for o in C_ORACLES:
            tot[g][o]["fp"] += sum(v[o]["correct"]); tot[g][o]["nc"] += len(v[o]["correct"])
            tot[g][o]["tp"] += sum(v[o]["fault"]); tot[g][o]["nf"] += len(v[o]["fault"])
    res["summary"] = {g: {o: {"correct": rate(t["fp"], t["nc"]), "fault": rate(t["tp"], t["nf"])} for o, t in tot[g].items()} for g in tot}
    return res


def step_d(ra, rd):
    t = lambda s: dt.datetime.strptime(s, "%Y%m%d_%H%M%S")
    shots = rd["shots"]; out = {"gap_hours": round((t(rd["submitted"]) - t(ra["submitted"])).total_seconds() / 3600, 1), "per_benchmark": {}}
    A = collections.defaultdict(list); Dd = collections.defaultdict(list)
    for it in ra["items"]:
        A[it["benchmark"]].append(it)
    for it in rd["items"]:
        Dd[it["benchmark"]].append(it)
    tot = collections.Counter()
    for b, items in Dd.items():
        rng = np.random.default_rng(0)
        cal0 = [it["counts"] for it in A[b] if it["role"] == "calibration"]
        ref0 = pooled(cal0); tau0 = compute_threshold(cal0, shots, ALPHA, "pooled", n_qubits=nq(cal0[0]), rng=rng).tau
        sent = [it["counts"] for it in items if it["role"] == "sentinel"]
        ts = compute_threshold(cal0, len(sent) * shots, ALPHA_S, "pooled", n_qubits=nq(cal0[0]), rng=rng).tau
        fired = jsd(ref0, pooled(sent)) > ts
        cal1 = [it["counts"] for it in items if it["role"] == "recalibration"]
        rep1 = compute_threshold(cal1, shots, ALPHA, "pooled", n_qubits=nq(cal1[0]), rng=rng)
        ref1, tau1 = pooled(cal1), rep1.tau
        corr = [it["counts"] for it in items if it["role"] == "correct_test"]
        flt = [it["counts"] for it in items if it["role"] == "injected"]
        row = {"sentinel_divergence": jsd(ref0, pooled(sent)), "sentinel_threshold": ts, "sentinel_fired": bool(fired),
               "shift": jsd(ref0, ref1), "tau_stale": tau0, "tau_fresh": tau1, "recalibration_drift_flag": bool(rep1.drift_detected),
               "stale_fp": sum(jsd(ref0, c) > tau0 for c in corr), "fresh_fp": sum(jsd(ref1, c) > tau1 for c in corr), "n_correct": len(corr),
               "stale_detected": sum(jsd(ref0, c) > tau0 for c in flt), "fresh_detected": sum(jsd(ref1, c) > tau1 for c in flt), "n_faults": len(flt)}
        row["policy_fp"] = row["fresh_fp"] if fired else row["stale_fp"]
        row["policy_detected"] = row["fresh_detected"] if fired else row["stale_detected"]
        out["per_benchmark"][b] = row
        for k in ("stale_fp", "fresh_fp", "policy_fp", "n_correct", "stale_detected", "fresh_detected", "policy_detected", "n_faults"):
            tot[k] += row[k]
        tot["fired"] += int(fired); tot["flag"] += int(rep1.drift_detected)
    out["summary"] = {"sentinel_fired": int(tot["fired"]), "benchmarks": len(Dd), "recalibration_drift_flags": int(tot["flag"]),
                      "fpr_stale": rate(tot["stale_fp"], tot["n_correct"]), "fpr_co_scheduled": rate(tot["fresh_fp"], tot["n_correct"]),
                      "fpr_sentinel_policy": rate(tot["policy_fp"], tot["n_correct"]),
                      "recall_stale": rate(tot["stale_detected"], tot["n_faults"]), "recall_co_scheduled": rate(tot["fresh_detected"], tot["n_faults"]),
                      "recall_sentinel_policy": rate(tot["policy_detected"], tot["n_faults"])}
    return out


def fmt(x):
    return f"{x['k']}/{x['n']}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    d = os.path.join(ROOT, "results", "hardware_analysis", "dryrun" if a.dry_run else "real", "hm"); os.makedirs(d, exist_ok=True)
    ra, rb, rp = latest("a", a.dry_run), latest("b", a.dry_run), latest("probe", a.dry_run)
    rc, rd = latest("c", a.dry_run), latest("d", a.dry_run)
    if ra is None:
        raise SystemExit("no step-a data yet")
    out = {"backend": ra["backend"], "job_a": ra["job_id"], "submitted_a": ra["submitted"], "dropped_a": ra.get("dropped", []),
           "qpu_seconds": {s: (r.get("quantum_seconds") if r else None) for s, r in (("probe", rp), ("a", ra), ("b", rb), ("c", rc), ("d", rd))},
           "a": step_a(ra)}
    if rb is not None:
        out["b"] = step_b(ra, rb); out["job_b"] = rb["job_id"]; out["submitted_b"] = rb["submitted"]
    if rc is not None:
        out["c"] = step_c(rc); out["job_c"] = rc["job_id"]; out["submitted_c"] = rc["submitted"]; out["dropped_c"] = rc.get("dropped", [])
    if rd is not None:
        out["d"] = step_d(ra, rd); out["job_d"] = rd["job_id"]; out["submitted_d"] = rd["submitted"]
    json.dump(out, open(os.path.join(d, "analysis.json"), "w"), indent=1, default=float)
    A = out["a"]; S = A["summary"]
    L = [f"CAMPAIGN HM ({'dry run on ' if a.dry_run else ''}{ra['backend']['name']}; step a {ra['submitted']}, job {ra['job_id']})",
         f"QPU seconds: {out['qpu_seconds']}", ""]
    L.append("RQ1/RQ2/RQ4 (step a): rejected / tested")
    L.append(f"  {'oracle':16s} {'correct runs':>14s} {'faults':>10s} {'noise-ampl.':>12s} {'noise-neutr.':>13s}")
    for o in ORACLES:
        L.append(f"  {NAMES[o]:16s} {fmt(S[o]['correct']):>14s} {fmt(S[o]['fault']):>10s} {fmt(S[o]['noise_amplifying']):>12s} {fmt(S[o]['noise_neutral']):>13s}")
    if A["ideal_gap_range"]:
        L.append(f"  divergence of correct runs from the ideal output: {A['ideal_gap_range'][0]:.4f} to {A['ideal_gap_range'][1]:.4f}")
    L.append("  calibration: " + ", ".join(f"{b} tau={c['tau_nao']:.4f} drift={c['drift_detected']}" for b, c in A["calibration"].items()))
    L.append("  NAO misses: " + (", ".join(f"{b}:{f['operator']} (ideal JSD {f['ideal_jsd']:.3f})" for b, r in A["per_benchmark"].items()
                                         for f in r["fault_detail"] if not f["nao"]) or "none"))
    z = A["attribution"]["summary"]
    if z:
        L += ["", f"RQ3 (step a): linear ZNE attribution on {z['n']} cases: accuracy {z['accuracy']:.3f}, kappa {z['kappa']:.3f}, "
              f"Bug recall {fmt(z['bug_recall'])}, Noise recall {fmt(z['noise_recall'])}"]
        for c in A["attribution"]["cases"]:
            L.append(f"  {c['benchmark']:7s} {c['case']} {c['family']:22s} {c['operator']:16s} truth={c['truth']:5s} pred={c['attribution']:5s} r0={c['r0']:.3f} null={c['null_q']:.3f}")
    if "b" in out:
        B = out["b"]; s = B["summary"]
        L += ["", f"RQ5 (step b, {B['gap_hours']} h after step a): sentinel fired on {s['sentinel_fired']} of {s['benchmarks']} benchmarks",
              f"  false positives: stale calibration {fmt(s['fpr_stale'])}, rebuilt from sentinel runs {fmt(s['fpr_fresh'])}, sentinel policy {fmt(s['fpr_sentinel_policy'])}",
              f"  faults detected: stale {fmt(s['recall_stale'])}, rebuilt {fmt(s['recall_fresh'])}, sentinel policy {fmt(s['recall_sentinel_policy'])}"]
        for b, r in B["per_benchmark"].items():
            L.append(f"  {b:7s} sentinel JSD={r['sentinel_divergence']:.5f} threshold={r['sentinel_threshold']:.5f} fired={r['sentinel_fired']} "
                     f"shift={r['reference_shift']:.5f} stale FP {r['stale_fp']}/{r['n_correct']}")
    if "c" in out:
        C = out["c"]["summary"]
        for g, title in (("large", "larger programs (Bell-8, GHZ-8, QAOA-8, HWE-6)"), ("real_bugs", "observable Bugs4Q bugs")):
            L += ["", f"Step c, {title}: rejected / tested"]
            for o in C_ORACLES:
                L.append(f"  {NAMES[o]:16s} correct runs {fmt(C[g][o]['correct']):>7s}   faults/bug runs {fmt(C[g][o]['fault']):>7s}")
            for b, x in out["c"][g].items():
                L.append(f"  {b:16s} tau={x['tau_nao']:.4f} outcomes={x['outcomes_observed']} drift={x['drift_detected']} "
                         f"NAO FP {x['nao']['correct']}/{x['nao']['n_correct']} detected {x['nao']['fault']}/{x['nao']['n_fault']}")
    if "d" in out:
        Dd = out["d"]; s = Dd["summary"]
        L += ["", f"RQ5 (step d, {Dd['gap_hours']} h after step a): sentinel fired on {s['sentinel_fired']} of {s['benchmarks']} benchmarks; "
              f"homogeneity test flagged {s['recalibration_drift_flags']} of the new calibrations",
              f"  false positives: stale calibration {fmt(s['fpr_stale'])}, co-scheduled (K = 8) {fmt(s['fpr_co_scheduled'])}, sentinel policy {fmt(s['fpr_sentinel_policy'])}",
              f"  faults detected: stale {fmt(s['recall_stale'])}, co-scheduled {fmt(s['recall_co_scheduled'])}, sentinel policy {fmt(s['recall_sentinel_policy'])}"]
        for b, r in Dd["per_benchmark"].items():
            L.append(f"  {b:7s} sentinel JSD={r['sentinel_divergence']:.5f} threshold={r['sentinel_threshold']:.5f} fired={r['sentinel_fired']} "
                     f"shift={r['shift']:.5f} stale FP {r['stale_fp']}/{r['n_correct']} fresh FP {r['fresh_fp']}/{r['n_correct']}")
    open(os.path.join(d, "REPORT.txt"), "w").write("\n".join(L) + "\n")
    # LaTeX table for the paper (RQ1, RQ2, RQ4 rows; RQ3/RQ5 numbers are in REPORT.txt)
    T = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"\textbf{Oracle} & \textbf{Correct runs rejected $\downarrow$} & \textbf{Faults detected $\uparrow$} & \textbf{Noise-ampl.\ rejected} & \textbf{Noise-neutral rejected $\downarrow$} \\", r"\midrule"]
    tn = {"nao": "NAO", "nao1": "NAO-1", "spc": "SPC envelope", **{f"fixed_{t:g}": f"Fixed $\\tau = {t:g}$" for t in FIXED}}
    for o in ORACLES:
        T.append(f"{tn[o]} & {fmt(S[o]['correct'])} & {fmt(S[o]['fault'])} & {fmt(S[o]['noise_amplifying'])} & {fmt(S[o]['noise_neutral'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(d, "hm_table.tex"), "w").write("\n".join(T) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
