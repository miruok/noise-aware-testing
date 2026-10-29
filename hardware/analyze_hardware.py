# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Offline analysis of hardware campaigns collected by run_hardware.py.

  python hardware/analyze_hardware.py            # real-hardware data only
  python hardware/analyze_hardware.py --dry-run  # dry-run (fake backend) data only
Writes results/hardware_analysis/<real|dryrun>/.
"""
from __future__ import annotations
import argparse, collections, glob, itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from nisq_framework import compute_threshold, jsd, pooled, attribute_from_counts
from nisq_framework.stats import cohen_kappa, confusion, wilson

RAW = os.path.join(ROOT, "results", "hardware_raw")
ALPHA = 1e-3


def load(campaign, dry):
    out = []
    for p in sorted(glob.glob(os.path.join(RAW, campaign, "*", "raw.json"))):
        r = json.load(open(p))
        if bool(r.get("dry_run")) == dry:
            r["_path"] = os.path.dirname(p); out.append(r)
    return out


def nq(counts):
    return len(next(iter(counts)))


def analyze_h1(runs):
    res = []
    for r in runs:
        by = collections.defaultdict(list)
        for it in r["items"]:
            by[it["benchmark"]].append(it)
        for b, items in by.items():
            cal = [it["counts"] for it in items if it["role"] == "calibration"]
            shots = sum(cal[0].values())
            rep = compute_threshold(cal, shots, ALPHA, "pooled", n_qubits=nq(cal[0]), rng=np.random.default_rng(0))
            rep_s = compute_threshold([], shots, ALPHA, "single", n_qubits=nq(cal[0]), reference_counts=cal[0], rng=np.random.default_rng(1))
            ref = pooled(cal)
            def killed(c, mode):
                return (jsd(ref, c) > rep.tau) if mode == "nao" else (jsd(cal[0], c) > rep_s.tau)
            rows = {}
            for mode in ("nao", "nao1"):
                ct = [killed(it["counts"], mode) for it in items if it["role"] == "correct_test"]
                faults = [killed(it["counts"], mode) for it in items if it["truth"] == "fault"]
                cats = collections.defaultdict(list)
                for it in items:
                    if it["role"] in ("mutant", "injected"):
                        cats[it.get("family", "injected_fault")].append(killed(it["counts"], mode))
                rows[mode] = {"fpr_correct": float(np.mean(ct)), "n_correct": len(ct),
                              "recall_faults": float(np.mean(faults)), "n_faults": len(faults),
                              "recall_ci": list(wilson(int(sum(faults)), len(faults))),
                              "rejection_by_family": {k: {"rate": float(np.mean(v)), "n": len(v)} for k, v in cats.items()}}
            res.append({"run": os.path.basename(r["_path"]), "backend": r["backend"]["name"], "benchmark": b,
                        "tau_nao": rep.tau, "tau_nao1": rep_s.tau, "drift_detected": rep.drift_detected,
                        "homogeneity_p": rep.homogeneity_p, "dispersion_ratio": rep.dispersion_ratio, **rows})
    return res


def analyze_h2(runs):
    sessions = []
    for r in runs:
        by = collections.defaultdict(list)
        for it in r["items"]:
            by[it["benchmark"]].append(it["counts"])
        sessions.append({"stamp": r["submitted"], "by": dict(by)})
    sessions.sort(key=lambda s: s["stamp"])
    res = {"sessions": [], "cross_session": []}
    for s in sessions:
        for b, cal in s["by"].items():
            rep = compute_threshold(cal, sum(cal[0].values()), ALPHA, "pooled", n_qubits=nq(cal[0]), rng=np.random.default_rng(0))
            res["sessions"].append({"stamp": s["stamp"], "benchmark": b, "homogeneity_p": rep.homogeneity_p,
                                    "dispersion_ratio": rep.dispersion_ratio, "drift_detected": rep.drift_detected})
    import datetime as dt
    t = lambda x: dt.datetime.strptime(x, "%Y%m%d_%H%M%S")
    for i, j in itertools.product(range(len(sessions)), repeat=2):
        if i == j: continue
        for b in sessions[i]["by"]:
            cal = sessions[i]["by"][b]; tests = sessions[j]["by"].get(b, [])
            if not tests: continue
            rep = compute_threshold(cal, sum(cal[0].values()), ALPHA, "pooled", n_qubits=nq(cal[0]), rng=np.random.default_rng(0))
            ref = pooled(cal)
            fp = [jsd(ref, c) > rep.tau for c in tests]
            res["cross_session"].append({"benchmark": b, "calibrated": sessions[i]["stamp"], "tested": sessions[j]["stamp"],
                                         "gap_hours": round((t(sessions[j]["stamp"]) - t(sessions[i]["stamp"])).total_seconds() / 3600, 2),
                                         "fpr": float(np.mean(fp)), "n": len(fp)})
    return res


def h2_summary(runs):
    """Forward session pairs only (calibration earlier than test). Stale-calibration
    false-positive rates by elapsed time, and the two-run sentinel (alpha_s = 1e-2,
    threshold for 2N shots) evaluated on the first two runs of the later session,
    with the remaining runs as the test runs it protects."""
    S = []
    for r in runs:
        by = collections.defaultdict(list)
        for it in r["items"]:
            by[it["benchmark"]].append(it["counts"])
        S.append((dt_parse(r["submitted"]), dict(by)))
    S.sort(key=lambda x: x[0])
    rows = []
    for i, (ti, bi) in enumerate(S):
        for tj, bj in S[i + 1:]:
            for b, cal in bi.items():
                rng = np.random.default_rng(0); shots = sum(cal[0].values())
                tau = compute_threshold(cal, shots, ALPHA, "pooled", n_qubits=nq(cal[0]), rng=rng).tau
                ts = compute_threshold(cal, 2 * shots, 1e-2, "pooled", n_qubits=nq(cal[0]), rng=rng).tau
                ref = pooled(cal); runs_j = bj[b]
                rows.append({"benchmark": b, "gap_hours": (tj - ti).total_seconds() / 3600, "same_day": ti.date() == tj.date(),
                             "shift_over_tau": jsd(ref, pooled(runs_j)) / tau, "fp": int(sum(jsd(ref, c) > tau for c in runs_j)), "n": len(runs_j),
                             "sentinel_fired": bool(jsd(ref, pooled(runs_j[:2])) > ts),
                             "fp_protected": int(sum(jsd(ref, c) > tau for c in runs_j[2:])), "n_protected": len(runs_j) - 2})
    out = {"pairs": rows, "by_benchmark": {}, "sentinel": {}}
    bins = (("<6h", 0, 6), ("6-12h", 6, 12), ("12-24h", 12, 24), (">=24h", 24, 1e9))
    for b in sorted({r["benchmark"] for r in rows}):
        R_ = [r for r in rows if r["benchmark"] == b]
        out["by_benchmark"][b] = {lab: {"pairs": len(x), "fpr": sum(r["fp"] for r in x) / sum(r["n"] for r in x), "max_shift_over_tau": max(r["shift_over_tau"] for r in x)}
                                  for lab, lo, hi in bins for x in [[r for r in R_ if lo <= r["gap_hours"] < hi]] if x}
        out["by_benchmark"][b]["all"] = {"pairs": len(R_), "fpr": sum(r["fp"] for r in R_) / sum(r["n"] for r in R_)}
    need = [r for r in rows if r["fp_protected"] > 0]; ok = [r for r in rows if r["fp_protected"] == 0]
    out["sentinel"] = {"pairs": len(rows), "stale_invalid": len(need), "caught": sum(r["sentinel_fired"] for r in need),
                       "stale_valid": len(ok), "false_alarms": sum(r["sentinel_fired"] for r in ok),
                       "fp_without_sentinel": [sum(r["fp_protected"] for r in rows), sum(r["n_protected"] for r in rows)],
                       "fp_missed_by_sentinel": sum(r["fp_protected"] for r in rows if not r["sentinel_fired"])}
    return out


def dt_parse(x):
    import datetime as _dt
    return _dt.datetime.strptime(x, "%Y%m%d_%H%M%S")


def analyze_h3(runs):
    res = []
    for r in runs:
        by = collections.defaultdict(dict)
        for it in r["items"]:
            by[(it["benchmark"], it["case"])][it["scale"]] = it
        for (b, case), d in by.items():
            if case == "P": continue
            P = by[(b, "P")]
            out = attribute_from_counts([P[1]["counts"], P[3]["counts"]], [d[1]["counts"], d[3]["counts"]], (1, 3), "linear", alpha=1e-2)
            res.append({"benchmark": b, "case": case, "family": d[1]["family"], "operator": d[1]["operator"],
                        "truth": d[1]["truth"], "attribution": out["attribution"], "r0": out["r0"], "null_q": out["null_q"],
                        "r_by_scale": out["r_by_scale"]})
    summ = {}
    if res:
        yt = [x["truth"] for x in res]; yp = [x["attribution"] for x in res]
        summ = {"n": len(res), "accuracy": float(np.mean([a == b for a, b in zip(yt, yp)])), "kappa": cohen_kappa(yt, yp),
                "confusion": confusion([x == "bug" for x in yt], [x == "bug" for x in yp])}
    return {"cases": res, "summary": summ}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    d = os.path.join(ROOT, "results", "hardware_analysis", "dryrun" if a.dry_run else "real"); os.makedirs(d, exist_ok=True)
    out = {"h1": analyze_h1(load("h1", a.dry_run)), "h2": analyze_h2(load("h2", a.dry_run)), "h3": analyze_h3(load("h3", a.dry_run))}
    h2r = load("h2", a.dry_run)
    out["h2_summary"] = h2_summary(h2r) if len(h2r) >= 2 else {}
    json.dump(out, open(os.path.join(d, "analysis.json"), "w"), indent=1, default=float)
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"HARDWARE ANALYSIS ({'dry run on fake backend' if a.dry_run else 'real hardware'})\n\nH1 mutation campaign\n")
        for x in out["h1"]:
            f.write(f"  {x['backend']} {x['benchmark']:9s} tau_NAO={x['tau_nao']:.4f} drift={x['drift_detected']} dispersion={x['dispersion_ratio']:.2f}\n")
            for m in ("nao", "nao1"):
                y = x[m]
                f.write(f"     {m:8s} FPR={y['fpr_correct']:.2f} (n={y['n_correct']}) recall={y['recall_faults']:.2f} (n={y['n_faults']}) by family: "
                        + ", ".join(f"{k}={v['rate']:.2f}" for k, v in y["rejection_by_family"].items()) + "\n")
        f.write("\nH2 drift sessions\n")
        for x in out["h2"]["sessions"]:
            f.write(f"  {x['stamp']} {x['benchmark']:9s} homogeneity_p={x['homogeneity_p']:.3g} dispersion={x['dispersion_ratio']:.2f} drift={x['drift_detected']}\n")
        for x in out["h2"]["cross_session"]:
            f.write(f"  calibrated {x['calibrated']} -> tested {x['tested']} ({x['gap_hours']} h) {x['benchmark']:9s} FPR={x['fpr']:.2f}\n")
        hs = out["h2_summary"]
        if hs:
            f.write("\nH2 summary (forward pairs; stale calibration judged on the 8 runs of a later session)\n")
            for b, v in hs["by_benchmark"].items():
                f.write(f"  {b:8s} " + "  ".join(f"{k}: FPR={x['fpr']:.3f} ({x['pairs']} pairs)" for k, x in v.items()) + "\n")
            s_ = hs["sentinel"]
            f.write(f"  sentinel (2 runs, alpha_s=1e-2) on {s_['pairs']} pairs: fired on {s_['caught']} of {s_['stale_invalid']} stale calibrations that "
                    f"produced false positives, and on {s_['false_alarms']} of {s_['stale_valid']} that did not; false positives on the protected runs: "
                    f"{s_['fp_without_sentinel'][0]}/{s_['fp_without_sentinel'][1]} without sentinel, {s_['fp_missed_by_sentinel']} missed by it\n")
        f.write("\nH3 attribution\n")
        for x in out["h3"]["cases"]:
            f.write(f"  {x['benchmark']:9s} {x['case']:4s} {x['family']:24s} {x['operator']:18s} truth={x['truth']:5s} pred={x['attribution']:5s} r0={x['r0']:.3f} null={x['null_q']:.3f}\n")
        if out["h3"]["summary"]:
            s = out["h3"]["summary"]; f.write(f"  accuracy={s['accuracy']:.3f} kappa={s['kappa']:.3f}\n")
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
