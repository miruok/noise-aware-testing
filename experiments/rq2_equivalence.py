# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ2 (equivalent mutants and faults): how often do mutation operators and
fault injectors produce programs that are semantically equivalent to the
original, and does the input state matter?

(a) Equivalence census: for every operator and benchmark, 100 raw draws
    without filtering; fraction equivalent on |0...0> and on |0...0> plus 3
    random computational-basis inputs.
(b) Fault catalogue: every injector on every benchmark; equivalent on
    |0...0>; whether the unitary changes although the measured output does
    not; equivalence over the 4-input set.
(c) Placement ablation: the same inserted-gate operators placed after the
    final measurements (a placement that can never change the measured
    output) vs before them.
Output: results/rq2_equivalence/.
"""
from __future__ import annotations
import argparse, collections, os, time
import numpy as np
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import CXGate, TGate, XGate
from common import BENCH, out_dir, save, write_meta, get_benchmark
from nisq_framework import MutationEngine, OPERATORS, EQUIVALENT_FAMILIES, TAXONOMY_CLASS, ideal_jsd, inject_all
from nisq_framework.semantics import unitary_equivalent


def inputs_for(qc, rng):
    return ("",) + tuple(format(int(x), f"0{qc.num_qubits}b") for x in rng.integers(1, 2 ** qc.num_qubits, 3))


def census(n_draws, seed):
    rows = []
    for b in BENCH:
        qc, _ = get_benchmark(b)
        rng = np.random.default_rng(seed)
        multi = inputs_for(qc, rng)
        eng = MutationEngine(qc, seed=seed)
        for fam, ops in OPERATORS.items():
            for op in ops:
                n = eq0 = eqm = 0
                for _ in range(n_draws):
                    r = eng._apply(op)
                    if r is None:
                        break
                    m = r[0]; n += 1
                    eq0 += ideal_jsd(qc, m, ("",)) < 1e-10
                    eqm += ideal_jsd(qc, m, multi) < 1e-10
                rows.append({"benchmark": b, "family": fam, "taxonomy_class": TAXONOMY_CLASS[fam], "operator": op,
                             "applicable_draws": n, "equivalent_zero_input": eq0, "equivalent_multi_input": eqm,
                             "rate_zero_input": eq0 / n if n else None, "rate_multi_input": eqm / n if n else None})
    return rows


def catalogue(seed):
    rows = []
    for b in BENCH:
        qc, _ = get_benchmark(b)
        multi = inputs_for(qc, np.random.default_rng(seed))
        for r in inject_all(qc):
            row = {"benchmark": b, "injector": r["injector"], "applicable": r["applicable"]}
            if r["applicable"]:
                row.update(ideal_jsd=r["ideal_jsd"], equivalent_zero_input=r["output_equivalent"],
                           unitary_changed_output_unchanged=r["basis_invisible"],
                           equivalent_multi_input=bool(ideal_jsd(qc, r["circuit"], multi) < 1e-10))
            rows.append(row)
    return rows


def placement(n, seed):
    """Insert CX / T / X either after all measurements or at a random point before them."""
    rows = []
    gates = {"insert_cnot": (CXGate, 2), "insert_t": (TGate, 1), "insert_x": (XGate, 1)}
    for b in BENCH:
        qc, _ = get_benchmark(b)
        rng = np.random.default_rng(seed)
        mpos = next(i for i, ins in enumerate(qc.data) if ins.operation.name == "measure")
        for op, (G, k) in gates.items():
            for where in ("after_measurement", "before_measurement"):
                eq = 0
                for _ in range(n):
                    m = qc.copy(); qs = [int(x) for x in rng.choice(qc.num_qubits, k, replace=False)]
                    pos = len(m.data) if where == "after_measurement" else int(rng.integers(0, mpos + 1))
                    m.data.insert(pos, CircuitInstruction(G(), [m.qubits[q] for q in qs], []))
                    eq += ideal_jsd(qc, _drop_post_measurement(m)) < 1e-10
                rows.append({"benchmark": b, "operator": op, "placement": where, "draws": n, "equivalent": eq, "rate": eq / n})
    return rows


def _drop_post_measurement(qc):
    """Gates after a qubit's measurement cannot change its recorded bit."""
    from qiskit import QuantumCircuit
    measured = set(); out = QuantumCircuit(*qc.qregs, *qc.cregs)
    for ins in qc.data:
        qs = {qc.find_bit(q).index for q in ins.qubits}
        if ins.operation.name == "measure":
            measured |= qs
        elif ins.operation.name != "barrier" and qs & measured:
            continue
        out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=5); ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq2_equivalence", a.out)
    cen = census(a.draws, a.seed); save(cen, os.path.join(d, "equivalence_census.json"))
    cat = catalogue(a.seed); save(cat, os.path.join(d, "fault_catalogue.json"))
    pla = placement(50, a.seed); save(pla, os.path.join(d, "placement_ablation.json"))
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in cen:
        x = agg[(r["family"], r["operator"])]; x[0] += r["applicable_draws"]; x[1] += r["equivalent_zero_input"]; x[2] += r["equivalent_multi_input"]
    summ = {"census_by_operator": {f"{f}|{op}": {"draws": v[0], "rate_zero_input": v[1] / v[0], "rate_multi_input": v[2] / v[0]}
                                   for (f, op), v in agg.items() if v[0]}}
    app = [r for r in cat if r["applicable"]]
    summ["faults"] = {"applicable": len(app), "equivalent_zero_input": sum(r["equivalent_zero_input"] for r in app),
                      "unitary_changed_output_unchanged": sum(r["unitary_changed_output_unchanged"] for r in app),
                      "equivalent_multi_input": sum(r["equivalent_multi_input"] for r in app)}
    pa = collections.defaultdict(lambda: [0, 0])
    for r in pla:
        pa[r["placement"]][0] += r["equivalent"]; pa[r["placement"]][1] += r["draws"]
    summ["placement"] = {k: {"equivalent": v[0], "draws": v[1], "rate": v[0] / v[1]} for k, v in pa.items()}
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ2 EQUIVALENCE CENSUS\n\n(a) Equivalence census (fraction of raw draws equivalent)\n")
        for k, v in sorted(summ["census_by_operator"].items()):
            f.write(f"  {k:45s} draws={v['draws']:5d}  |0>={v['rate_zero_input']:.3f}  4 inputs={v['rate_multi_input']:.3f}\n")
        s = summ["faults"]
        f.write(f"\n(b) Faults: {s['equivalent_zero_input']}/{s['applicable']} applicable injections equivalent on |0...0> "
                f"({s['unitary_changed_output_unchanged']} change the unitary but not the output); "
                f"{s['equivalent_multi_input']} remain equivalent over 4 inputs\n")
        for r in app:
            if r["equivalent_zero_input"]:
                f.write(f"    {r['benchmark']:8s} {r['injector']:20s} multi-input equivalent={r['equivalent_multi_input']}\n")
        f.write("\n(c) Placement ablation (insert CX / T / X)\n")
        for k, v in summ["placement"].items():
            f.write(f"  {k:20s} equivalent {v['equivalent']}/{v['draws']} = {v['rate']:.3f}\n")
    write_meta(d, vars(a), t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
