# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ2 (taxonomy, compiled-circuit check): do the semantics-preserving mutants
have the physical property that defines their class after compilation?

Definition 3 assigns a semantics-preserving mutant to the noise-amplifying
class if it inserts operations that survive compilation, and to the
noise-neutral class if it reorders or re-expresses gates without doing so.
The statevector labels verify the semantic half of the definition; this
script verifies the physical half. It regenerates exactly the mutants of the
mutation analysis (MutationEngine seed 42, 15 per family; core benchmarks as
in rq2_mutation.py and the 32 QASMBench programs as in ext_suite.py), compiles
each mutant and its original as the experiments do (fixed policy,
optimization level 1, mutant placed on the original's initial layout), and
compares the number of noisy operations of the compiled circuits on three
targets:
  synthetic  the Aer backend of the synthetic noise model, on which every
             gate except barriers and measurements carries depolarizing error;
  cx_native  FakeKolkataV2 (CNOT-native IBM device snapshot);
  cz_native  FakeKingston (CZ-native IBM Heron device snapshot, the device of
             the hardware campaigns).
On the IBM targets, rz, barrier, delay, id and measure are not counted, since
rz is implemented virtually; a physical gate is any other operation, and
two-qubit gates are counted separately.
Finally, the rejection records of the mutation analysis (results/rq2_mutation,
results/ext_suite) are joined with these labels: rejection rates are reported
separately for mutants that do and do not satisfy their class on the
synthetic target (on which the mutation analysis was executed), and the
noise-amplifying versus noise-neutral contrast is repeated with the
noise-neutral class restricted to mutants verified to add no operation
(--analyze-only re-runs this step on an existing records.json).
Output: results/taxonomy_compilation/.
"""
from __future__ import annotations
import argparse, os, time
from collections import defaultdict
import numpy as np
from common import BENCH, RESULTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, MutationEngine, make_noisy_backend
from nisq_framework.benchmarks import QASMBENCH, load_qasmbench

FAMS = ("noise_amplifying", "noise_neutral")
NOT_PHYSICAL = {"rz", "barrier", "delay", "id", "measure", "reset"}


def counts(t, target):
    n2 = n1 = 0
    for inst in t.data:
        name = inst.operation.name
        if name in ("barrier", "measure", "delay", "reset"):
            continue
        if target != "synthetic" and name in NOT_PHYSICAL:
            continue
        if inst.operation.num_qubits >= 2:
            n2 += 1
        else:
            n1 += 1
    return n2, n1


def contrasts(d):
    import json
    from rq2_mutation import cell_tests
    T = json.load(open(os.path.join(d, "records.json")))
    sources = {"core": os.path.join(RESULTS, "rq2_mutation", "records.json"),
               "qasmbench": os.path.join(RESULTS, "ext_suite", "mutation_records.json")}
    out = {}
    for suite, path in sources.items():
        R = json.load(open(path)); pos = defaultdict(int); lab = {}
        for t in T:
            if t["suite"] == suite:
                k = (t["benchmark"], t["family"]); lab[k + (pos[k],)] = t; pos[k] += 1
        pos = defaultdict(int); R2 = []
        for r in R:
            if r["family"] in FAMS:
                k = (r["benchmark"], r["noise"], r["family"]); i = pos[k]; pos[k] += 1
                t = lab[(r["benchmark"], r["family"], i)]
                assert t["operator"] == r["operator"], (suite, k, i)
                r = dict(r, adds=t["synthetic"]["delta_2q"] > 0 or t["synthetic"]["delta_1q"] > 0)
            R2.append(r)
        R2 = [r for r in R2 if r["original_passed_nao"]]
        res = {}
        for fam in FAMS:
            for adds in (True, False):
                S = [r for r in R2 if r["family"] == fam and r["adds"] == adds]
                res[f"{fam}_adds_{adds}"] = {"n": len(S), "rejected": sum(r["rejected_nao"] for r in S)}
        verified = [r for r in R2 if not (r["family"] == "noise_neutral" and r["adds"])]
        res["amplifying_vs_verified_neutral"] = {nl or "pooled": cell_tests([r for r in verified if nl is None or r["noise"] == nl],
                                                                          "taxonomy_class", "noise_amplifying", "noise_neutral")
                                                 for nl in ("low", "medium", "high", None)}
        out[suite] = res
    save(out, os.path.join(d, "class_verified_contrasts.json"))
    with open(os.path.join(d, "REPORT.txt"), "a") as f:
        f.write("\nRejection under NAO by whether the mutant satisfies its class on the synthetic target\n")
        for suite, res in out.items():
            f.write(f"[{suite}]\n")
            for k, v in res.items():
                if k.startswith("noise_"):
                    f.write(f"  {k:28s} n={v['n']:5d} rejected={v['rejected']}\n")
            for nl, v in res["amplifying_vs_verified_neutral"].items():
                f.write(f"  amplifying vs verified neutral, {nl:6s}: CMH p={v['cmh_p']:.3g}  sign-flip p={v['signflip_p']:.3g}  "
                        f"cells {v['cells_higher']}/{v['cells_equal']}/{v['cells_lower']}\n")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--analyze-only")
    a = ap.parse_args()
    if a.analyze_only:
        contrasts(a.analyze_only); print(open(os.path.join(a.analyze_only, "REPORT.txt")).read()); return
    t0 = time.time(); d = out_dir("taxonomy_compilation", a.out)
    from qiskit_ibm_runtime.fake_provider import FakeKolkataV2, FakeKingston
    targets = {"synthetic": make_noisy_backend(0.005), "cx_native": FakeKolkataV2(), "cz_native": FakeKingston()}
    ex = Executor(seed=11)
    progs = [("core", b, get_benchmark(b)[0]) for b in BENCH] + [("qasmbench", b, load_qasmbench(b)) for b in QASMBENCH]
    recs = []
    for suite, b, qc in progs:
        eng = MutationEngine(qc, seed=42)
        muts = {f: eng.generate(15, f) for f in ("noise_amplifying", "noise_neutral", "semantic_additive", "semantic_subtractive")}
        base = {tg: counts(ex.compile(qc, be), tg) for tg, be in targets.items()}
        for fam in FAMS:
            for m in muts[fam]:
                r = {"suite": suite, "benchmark": b, "family": fam, "operator": m.operator, "meta": {k: str(v) for k, v in m.meta.items()}}
                for tg, be in targets.items():
                    n2, n1 = counts(ex.compile(m.circuit, be, layout_from=qc), tg)
                    r[tg] = {"delta_2q": n2 - base[tg][0], "delta_1q": n1 - base[tg][1]}
                recs.append(r)
        print(f"{suite:9s} {b:22s} done", flush=True)
    save(recs, os.path.join(d, "records.json"))

    def added(r, tg):
        return r[tg]["delta_2q"] > 0 or r[tg]["delta_1q"] > 0

    summ = {}
    for suite in ("core", "qasmbench", "all"):
        S = [r for r in recs if suite == "all" or r["suite"] == suite]
        out = {}
        for fam in FAMS:
            F = [r for r in S if r["family"] == fam]
            per_t = {}
            for tg in targets:
                per_t[tg] = {"n": len(F), "adds_physical_ops": sum(added(r, tg) for r in F),
                             "adds_2q": sum(r[tg]["delta_2q"] > 0 for r in F),
                             "changes_2q_count": sum(r[tg]["delta_2q"] != 0 for r in F),
                             "by_operator": {}}
                ops = defaultdict(list)
                for r in F:
                    key = r["operator"]
                    if r["operator"] == "inverse_pair_1q":
                        key += ":" + r["meta"].get("pair", "?").split("-")[0]
                    ops[key].append(r)
                for k, L in sorted(ops.items()):
                    per_t[tg]["by_operator"][k] = {"n": len(L), "adds_physical_ops": sum(added(r, tg) for r in L),
                                                   "adds_2q": sum(r[tg]["delta_2q"] > 0 for r in L),
                                                   "mean_delta_2q": float(np.mean([r[tg]["delta_2q"] for r in L])),
                                                   "mean_delta_1q": float(np.mean([r[tg]["delta_1q"] for r in L]))}
            out[fam] = per_t
        summ[suite] = out
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("TAXONOMY: physical property of semantics-preserving mutants after compilation\n"
                "(adds physical ops: more noisy operations than the compiled original)\n")
        for suite, out in summ.items():
            f.write(f"\n[{suite}]\n")
            for fam, per_t in out.items():
                for tg, v in per_t.items():
                    f.write(f"  {fam:16s} {tg:9s} n={v['n']:4d} adds physical ops {v['adds_physical_ops']:4d}  adds 2q {v['adds_2q']:4d}  changes 2q count {v['changes_2q_count']:4d}\n")
                    for k, o in v["by_operator"].items():
                        f.write(f"      {k:22s} n={o['n']:4d} adds {o['adds_physical_ops']:4d}  adds 2q {o['adds_2q']:4d}  "
                                f"mean d2q {o['mean_delta_2q']:+.2f}  mean d1q {o['mean_delta_1q']:+.2f}\n")
    contrasts(d)
    write_meta(d, vars(a) | {"targets": list(targets), "optimization_level": 1, "seed_transpiler": 11}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
