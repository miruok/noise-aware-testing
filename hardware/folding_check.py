# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Folding check on hardware: does global unitary folding realize the nominal
noise scale factor s = 3 after transpilation for ibm_kingston?

For every attribution pair archived from the hardware campaigns (HM first job,
H3), the ISA circuit executed at s = 3 is compared with the ISA circuit of the
same program at s = 1. Reported per pair: the ratio of two-qubit gates, of
physical single-qubit gates (sx, x; rz is virtual) and of the first-order
error budget sum(n_g * e_g) with the backend's median gate errors recorded in
the job, which is the scale factor that a first-order (linear in error)
noise model would see. Readout error is not folded and is excluded. A second check compares, for the
semantics-preserving mutants of H1, the physical operations of each compiled
mutant with those of the compiled calibration circuit of its benchmark.
Output: results/hardware_analysis/folding_check/.
"""
from __future__ import annotations
import json, os
import numpy as np
from qiskit import qpy

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "results", "hardware_raw")
OUT = os.path.join(HERE, "..", "results", "hardware_analysis", "folding_check")
JOBS = [("HM", os.path.join(RAW, "hm", "a", "20261021_114118_ibm_kingston")),
        ("H3", os.path.join(RAW, "h3", "20261022_143846_ibm_kingston"))]
VIRTUAL = {"rz", "barrier", "delay", "id", "measure", "reset"}


def gate_counts(c):
    n2 = n1 = 0
    for inst in c.data:
        if inst.operation.name in VIRTUAL:
            continue
        if inst.operation.num_qubits >= 2:
            n2 += 1
        else:
            n1 += 1
    return n2, n1


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for camp, d in JOBS:
        raw = json.load(open(os.path.join(d, "raw.json")))
        with open(os.path.join(d, "isa_circuits.qpy"), "rb") as fh:
            isa = qpy.load(fh)
        er = raw["backend"]["error_rates"]
        e2 = er["cz"]["median"]; e1 = er["sx"]["median"]
        s1, s3 = {}, {}
        for i, it in enumerate(raw["items"]):
            if "case" not in it or "scale" not in it:
                continue
            key = (it["benchmark"], it["case"])
            (s1 if it["scale"] == 1 else s3 if it["scale"] == 3 else {})[key] = (it, isa[i])
        for key in sorted(s3):
            if key not in s1:
                continue
            (it1, c1), (_, c3) = s1[key], s3[key]
            a2, a1 = gate_counts(c1); b2, b1 = gate_counts(c3)
            budget1, budget3 = a2 * e2 + a1 * e1, b2 * e2 + b1 * e1
            rows.append({"campaign": camp, "benchmark": key[0], "case": key[1], "family": it1.get("family", "reference"),
                         "operator": it1.get("operator", "reference"),
                         "two_qubit": [a2, b2], "one_qubit_physical": [a1, b1],
                         "ratio_two_qubit": b2 / a2 if a2 else None, "ratio_one_qubit": b1 / a1 if a1 else None,
                         "ratio_error_budget": budget3 / budget1 if budget1 else None})
    r2 = np.array([r["ratio_two_qubit"] for r in rows if r["ratio_two_qubit"]])
    rb = np.array([r["ratio_error_budget"] for r in rows if r["ratio_error_budget"]])
    summ = {"pairs": len(rows), "pairs_with_two_qubit_gates": int(len(r2)), "two_qubit_ratio": {"min": float(r2.min()), "median": float(np.median(r2)), "max": float(r2.max()),
                                                     "exactly_3": int(np.isclose(r2, 3).sum())},
            "error_budget_ratio": {"min": float(rb.min()), "median": float(np.median(rb)), "max": float(rb.max())}}
    json.dump({"rows": rows, "summary": summ}, open(os.path.join(OUT, "folding_check.json"), "w"), indent=1)
    with open(os.path.join(OUT, "REPORT.txt"), "w") as f:
        f.write(f"FOLDING CHECK on ibm_kingston ISA circuits: s = 3 versus s = 1 ({len(rows)} pairs)\n")
        f.write(f"two-qubit gate ratio: min {summ['two_qubit_ratio']['min']:.2f}, median {summ['two_qubit_ratio']['median']:.2f}, "
                f"max {summ['two_qubit_ratio']['max']:.2f}; exactly 3 in {summ['two_qubit_ratio']['exactly_3']} of {summ['pairs_with_two_qubit_gates']} pairs with two-qubit gates\n")
        f.write(f"first-order error-budget ratio: min {summ['error_budget_ratio']['min']:.2f}, median {summ['error_budget_ratio']['median']:.2f}, "
                f"max {summ['error_budget_ratio']['max']:.2f}\n\n")
        for r in rows:
            f.write(f"  {r['campaign']} {r['benchmark']:7s} {r['case']:3s} {r['operator']:22s} 2q {r['two_qubit'][0]:3d} -> {r['two_qubit'][1]:3d}  "
                    f"1q {r['one_qubit_physical'][0]:3d} -> {r['one_qubit_physical'][1]:3d}  budget ratio {r['ratio_error_budget']:.2f}\n")
    h1 = os.path.join(RAW, "h1", "20261022_143620_ibm_kingston")
    raw = json.load(open(os.path.join(h1, "raw.json")))
    with open(os.path.join(h1, "isa_circuits.qpy"), "rb") as fh:
        isa = qpy.load(fh)
    base = {it["benchmark"]: gate_counts(isa[i]) for i, it in enumerate(raw["items"]) if it["role"] == "calibration"}
    cls = []
    for i, it in enumerate(raw["items"]):
        if it.get("family") in ("noise_amplifying", "noise_neutral"):
            n2, n1 = gate_counts(isa[i]); b2, b1 = base[it["benchmark"]]
            cls.append({"benchmark": it["benchmark"], "family": it["family"], "operator": it["operator"],
                        "delta_2q": n2 - b2, "delta_1q": n1 - b1})
    json.dump(cls, open(os.path.join(OUT, "h1_class_check.json"), "w"), indent=1)
    with open(os.path.join(OUT, "REPORT.txt"), "a") as f:
        f.write("\nH1 semantics-preserving mutants: physical operations added after compilation\n")
        for fam in ("noise_amplifying", "noise_neutral"):
            F = [c for c in cls if c["family"] == fam]
            f.write(f"  {fam}: {sum(c['delta_2q'] > 0 or c['delta_1q'] > 0 for c in F)} of {len(F)} add physical operations\n")
    print(open(os.path.join(OUT, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
