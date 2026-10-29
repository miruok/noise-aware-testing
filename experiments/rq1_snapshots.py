# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ1 (soundness on device noise models): Qiskit fake backends built from IBM
device calibration snapshots, under two compilation policies:
  fixed    one seeded transpilation per circuit (recommended practice)
  random   a fresh transpilation on every execution (layout and routing may
           change between runs, emulating recompilation variability)
(a) Soundness: W windows x T correct runs; NAO, NAO-1, SPC envelope, fixed
    thresholds; NAO drift flags.
(b) Detection (fixed policy): semantic mutants and non-equivalent injected
    faults; equivalent mutants as negative controls.
Output: results/rq1_snapshots/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from qiskit_aer import AerSimulator
from common import SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import (Executor, MutationEngine, FAMILIES, compute_threshold, spc_envelope_threshold, jsd, pooled, inject_all)

SMALL = ["bell", "ghz_3", "qft_3", "qaoa_3", "vqe_3"]
LARGE = ["bell_8", "ghz_8", "qaoa_8"]
FIXED = [0.005, 0.01, 0.02, 0.05, 0.10]
ALPHA = 1e-3
ORACLES = ["nao", "nao1", "spc_envelope"] + [f"fixed_{t:g}" for t in FIXED]


def backends():
    from qiskit_ibm_runtime.fake_provider import FakeKolkataV2, FakeLagosV2, FakeManilaV2
    return [(FakeManilaV2(), SMALL), (FakeLagosV2(), SMALL), (FakeKolkataV2(), SMALL + LARGE)]


def oracles(qc, be, ex, rng, B, ideal):
    runs = ex.counts_many(qc, be, SHOTS, 13); nq = qc.num_qubits
    rep = compute_threshold(runs[:8], SHOTS, ALPHA, "pooled", n_qubits=nq, B=B, rng=rng)
    O = {"nao1": (runs[12], compute_threshold([], SHOTS, ALPHA, "single", n_qubits=nq, B=B, reference_counts=runs[12], rng=rng).tau),
         "nao": (pooled(runs[:8]), rep.tau), "spc_envelope": (runs[12], spc_envelope_threshold(runs[:12]))}
    for t in FIXED:
        O[f"fixed_{t:g}"] = (ideal, t)
    return O, rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=15)
    ap.add_argument("--tests", type=int, default=5); ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=31); ap.add_argument("--mutants", type=int, default=8)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq1_snapshots", a.out); rng = np.random.default_rng(a.seed)
    sound, detect = [], []
    for be, benches in backends():
        for b in benches:
            qc, _ = get_benchmark(b)
            for policy in ("fixed", "random"):
                ex = Executor(seed=a.seed, layout_policy=policy)
                ideal = ex.counts(qc, AerSimulator(), SHOTS)
                fp, drift, taus = {}, [], {}
                for _ in range(a.windows):
                    O, rep = oracles(qc, be, ex, rng, a.B, ideal); drift.append(rep.drift_detected)
                    for o, (ref, tau) in O.items():
                        taus.setdefault(o, []).append(tau)
                    for _ in range(a.tests):
                        c = ex.counts(qc, be, SHOTS)
                        for o, (ref, tau) in O.items():
                            fp[o] = fp.get(o, 0) + (jsd(ref, c) > tau)
                n = a.windows * a.tests
                sound.append({"backend": be.name, "benchmark": b, "policy": policy, "tests": n,
                              "fpr": {o: v / n for o, v in fp.items()}, "median_tau": {o: float(np.median(v)) for o, v in taus.items()},
                              "drift_flag_rate": float(np.mean(drift))})
                print(be.name, b, policy, {o: round(v / n, 3) for o, v in fp.items() if not o.startswith("fixed")}, "drift", np.mean(drift), flush=True)
            ex = Executor(seed=a.seed + 1, layout_policy="fixed")
            eng = MutationEngine(qc, seed=42)
            cases = [(m.family, m.operator, m.circuit, m.semantic_fault) for f in FAMILIES for m in eng.generate(a.mutants, f)]
            cases += [("injected_fault", r["injector"], r["circuit"], not r["output_equivalent"]) for r in inject_all(qc) if r["applicable"]]
            O, _ = oracles(qc, be, ex, rng, a.B, ex.counts(qc, AerSimulator(), SHOTS))
            for fam, op, T, fault in cases:
                c = ex.counts(T, be, SHOTS, layout_from=qc)
                detect.append({"backend": be.name, "benchmark": b, "case_type": fam, "operator": op, "fault": bool(fault),
                               **{o: bool(jsd(ref, c) > tau) for o, (ref, tau) in O.items()}})
    save({"soundness": sound, "detection": detect}, os.path.join(d, "results.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("RQ1 DEVICE-SNAPSHOT BACKENDS\n\n(a) FPR by backend and compilation policy (pooled over benchmarks)\n")
        for bn in sorted({s["backend"] for s in sound}):
            for pol in ("fixed", "random"):
                S = [s for s in sound if s["backend"] == bn and s["policy"] == pol]; n = sum(s["tests"] for s in S)
                f.write(f"  {bn:13s} {pol:6s} " + " ".join(f"{o}={sum(s['fpr'][o]*s['tests'] for s in S)/n:.3f}" for o in ORACLES)
                        + f"  drift_flag={np.mean([s['drift_flag_rate'] for s in S]):.2f}\n")
        f.write("\n  per benchmark\n")
        for s in sound:
            f.write(f"    {s['backend']:13s} {s['benchmark']:7s} {s['policy']:6s} tau_NAO={s['median_tau']['nao']:.4f} tau_SPC={s['median_tau']['spc_envelope']:.4f} FPR_NAO={s['fpr']['nao']:.3f} FPR_NAO1={s['fpr']['nao1']:.3f} FPR_SPC={s['fpr']['spc_envelope']:.3f} drift={s['drift_flag_rate']:.2f}\n")
        f.write("\n(b) detection (fixed policy): recall on faults; rejection rate of equivalent programs\n")
        for o in ORACLES:
            F = [x[o] for x in detect if x["fault"]]; N = [x[o] for x in detect if not x["fault"]]
            f.write(f"  {o:13s} recall={np.mean(F):.3f} (n={len(F)})  equivalent rejected={np.mean(N):.3f} (n={len(N)})\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha": ALPHA}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
