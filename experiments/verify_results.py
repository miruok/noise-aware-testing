# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Integrity check: recompute headline numbers from per-record files and
compare with each summary.json. Exit code 1 on any mismatch."""
import json, os, sys
from common import RESULTS
P = lambda *a: os.path.join(RESULTS, *a)
bad = []


def check(name, a, b, tol=1e-9):
    ok = abs(a - b) <= tol
    print(f"{'OK ' if ok else 'BAD'} {name}: records={a} summary={b}")
    if not ok: bad.append(name)


if os.path.exists(P("rq1_soundness", "per_window.json")):
    w = json.load(open(P("rq1_soundness", "per_window.json"))); s = json.load(open(P("rq1_soundness", "summary.json")))
    for o in s["by_oracle"]:
        check(f"RQ1 {o} FP", sum(sum(x["fp"][o]) for c in w for x in c["windows"]), s["by_oracle"][o]["fp"])
        check(f"RQ1 {o} tests", sum(len(x["fp"][o]) for c in w for x in c["windows"]), s["by_oracle"][o]["tests"])
if os.path.exists(P("rq4_faults", "records.json")):
    r = json.load(open(P("rq4_faults", "records.json"))); s = json.load(open(P("rq4_faults", "summary.json")))
    ev = [x for x in r if x["kind"] == "correct" or x["fault"]]
    for o in s["overall"]:
        check(f"RQ4-faults {o} TP", sum(x[o] for x in ev if x["fault"]), s["overall"][o]["tp"])
        check(f"RQ4-faults {o} FP", sum(x[o] for x in ev if not x["fault"]), s["overall"][o]["fp"])
if os.path.exists(P("rq2_mutation", "records.json")):
    r = json.load(open(P("rq2_mutation", "records.json"))); s = json.load(open(P("rq2_mutation", "summary.json")))
    for o, v in s["rejection_by_class"].items():
        for c, x in v.items():
            check(f"RQ2-mutation {o} {c}", sum(y[f"rejected_{o}"] for y in r if y["taxonomy_class"] == c), x["k"])
    check("RQ2-mutation semantic mutants non-equivalent", sum(y["output_equivalent"] for y in r if y["taxonomy_class"] == "semantic"), 0)
    check("RQ2-mutation equivalent mutants equivalent", sum(not y["output_equivalent"] for y in r if y["taxonomy_class"] != "semantic"), 0)
if os.path.exists(P("rq3_attribution", "records.json")):
    r = json.load(open(P("rq3_attribution", "records.json"))); s = json.load(open(P("rq3_attribution", "summary.json")))
    for m in ("zne_linear", "zne_quadratic", "reexecution_triage"):
        check(f"RQ3 {m} accuracy", sum(x["truth"] == x[m] for x in r) / len(r), s["all_cases"][m]["accuracy"], 1e-12)
if os.path.exists(P("supplementary", "results.json")) and os.path.exists(P("rq3_attribution", "records.json")):
    r = json.load(open(P("rq3_attribution", "records.json"))); s = json.load(open(P("supplementary", "results.json")))
    for m, v in s["rq3_triggered_by_noise"].items():
        check(f"Supplementary RQ3 {m} triggered cases", sum(c["cases"] for c in v.values()), sum(x["triggered"] for x in r))
        check(f"Supplementary RQ3 {m} triggered correct", sum(c["TP"] + c["TN"] for c in v.values()),
              sum(x["triggered"] and x[m] == x["truth"] for x in r))
    w = json.load(open(P("rq1_soundness", "per_window.json")))
    for key, pb in s["rq1_per_program"]["per_benchmark"].items():
        check(f"Supplementary RQ1 {key} FP over programs", sum(v["fp"] for v in pb.values()), sum(sum(x["fp"][key]) for c in w for x in c["windows"]))
if os.path.exists(P("rq4_global_threshold", "records.json")):
    r = json.load(open(P("rq4_global_threshold", "records.json"))); s = json.load(open(P("rq4_global_threshold", "summary.json")))
    ev = [x for x in r if x["kind"] == "correct" or x["fault"]]
    for o in s["overall"]:
        check(f"RQ4-global {o} TP", sum(x[o] for x in ev if x["fault"]), s["overall"][o]["tp"])
        check(f"RQ4-global {o} FP", sum(x[o] for x in ev if not x["fault"]), s["overall"][o]["fp"])
if os.path.exists(P("rq4_ablation", "summary.json")):
    s = json.load(open(P("rq4_ablation", "summary.json")))["results"]
    for part, g in (("synthetic", "synthetic"), ("snapshots", "snapshots_random_compilation")):
        if not os.path.exists(P("rq4_ablation", f"records_{part}.json")) or g not in s:
            continue
        r = json.load(open(P("rq4_ablation", f"records_{part}.json")))
        for k in "ABCDEF":
            check(f"RQ4-ablation {part} {k} FP", sum(x[k] for x in r if x["kind"] == "correct"), s[g][k]["fp"])
            check(f"RQ4-ablation {part} {k} recall", sum(x[k] for x in r if x["fault"]) / sum(x["fault"] for x in r), s[g][k]["recall"], 1e-12)
print("\nALL CHECKS PASSED" if not bad else f"\nMISMATCHES: {bad}")
sys.exit(1 if bad else 0)
