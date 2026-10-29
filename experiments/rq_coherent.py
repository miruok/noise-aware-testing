# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
RQ1 and RQ4 under coherent noise: false-positive rate and fault detection when
the gate errors are systematic over-rotations rather than depolarizing.

The synthetic noise of the main experiments is depolarizing (incoherent), and
the device snapshots add relaxation and readout error. Real devices also have
coherent errors, which rotate the state systematically and can shift output
probabilities in either direction. This driver adds them:
  every single-qubit gate is followed by RX(eps) on its qubit, and every
  two-qubit gate by RZZ(5 eps) on its qubits (coherent unitary errors),
  composed with depolarizing error of level eta = 0.001 (single-qubit eta,
  two-qubit 10 eta) and symmetric readout error 2 eta,
for eps in {0.02, 0.05} rad. Per core benchmark and eps, 20 calibration
windows judge five correct runs and one run of every applicable injected
fault each, with the oracles of rq4_faults.py on identical executions.
Output: results/rq_coherent/.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, coherent_unitary_error
from qiskit.circuit.library import RXGate, RZZGate
from common import BENCH, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, compute_threshold, spc_envelope_threshold, jsd, pooled, inject_all
from nisq_framework.stats import confusion, cluster_bootstrap
from rq4_faults import FIXED, ALPHA

ETA = 0.001
EPS = {"coherent_0.02": 0.02, "coherent_0.05": 0.05}
ONE_Q = ["u1", "u2", "u3", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "id", "sx"]
TWO_Q = ["cx", "cz"]
ORACLES = ["nao", "nao1", "spc_envelope"] + [f"fixed_{t:g}" for t in FIXED]


def coherent_backend(eps):
    nm = NoiseModel()
    e1 = depolarizing_error(ETA, 1).compose(coherent_unitary_error(RXGate(eps).to_matrix()))
    e2 = depolarizing_error(10 * ETA, 2).compose(coherent_unitary_error(RZZGate(5 * eps).to_matrix()))
    nm.add_all_qubit_quantum_error(e1, ONE_Q)
    nm.add_all_qubit_quantum_error(e2, TWO_Q)
    p = 2 * ETA
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return AerSimulator(noise_model=nm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--B", type=int, default=3000); ap.add_argument("--seed", type=int, default=515)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("rq_coherent", a.out)
    ex = Executor(seed=a.seed); rng = np.random.default_rng(a.seed); ideal_b = AerSimulator()
    recs, drift = [], {}
    for lvl, eps in EPS.items():
        be = coherent_backend(eps)
        for b in BENCH:
            qc, _ = get_benchmark(b)
            inj = [r for r in inject_all(qc) if r["applicable"]]
            flags = 0
            for w in range(a.windows):
                runs = ex.counts_many(qc, be, SHOTS, 13); ideal = ex.counts(qc, ideal_b, SHOTS)
                rp = compute_threshold(runs[:8], SHOTS, ALPHA, "pooled", n_qubits=qc.num_qubits, B=a.B, rng=rng)
                rs = compute_threshold([], SHOTS, ALPHA, "single", n_qubits=qc.num_qubits, B=a.B, reference_counts=runs[12], rng=rng)
                flags += int(rp.drift_detected)
                O = {"nao": (pooled(runs[:8]), rp.tau), "nao1": (runs[12], rs.tau), "spc_envelope": (runs[12], spc_envelope_threshold(runs[:12]))}
                for t in FIXED:
                    O[f"fixed_{t:g}"] = (ideal, t)
                judge = lambda c: {o: int(jsd(ref, c) > tau) for o, (ref, tau) in O.items()}
                for _ in range(5):
                    recs.append({"level": lvl, "benchmark": b, "window": w, "kind": "correct", "fault": False, "equivalent": False,
                                 **judge(ex.counts(qc, be, SHOTS))})
                for r in inj:
                    recs.append({"level": lvl, "benchmark": b, "window": w, "kind": r["injector"], "fault": not r["output_equivalent"],
                                 "equivalent": bool(r["output_equivalent"]), **judge(ex.counts(r["circuit"], be, SHOTS, layout_from=qc))})
            drift[f"{lvl}|{b}"] = flags / a.windows
            print(lvl, b, flush=True)
    save(recs, os.path.join(d, "records.json"))
    summ = {"eps": EPS, "eta": ETA, "overall": {}, "by_level": {}, "drift_flag_rate": drift}
    for scope in ["overall"] + list(EPS):
        S = recs if scope == "overall" else [r for r in recs if r["level"] == scope]
        ev = [r for r in S if r["kind"] == "correct" or r["fault"]]
        out = {}
        for o in ORACLES:
            cm = confusion([r["fault"] for r in ev], [bool(r[o]) for r in ev])
            clusters = {}
            for r in S:
                if r["kind"] == "correct":
                    clusters.setdefault((r["level"], r["benchmark"], r["window"]), []).append(r[o])
            cb = cluster_bootstrap(list(clusters.values()), B=2000, seed=3)
            out[o] = {k: cm[k] for k in ("tp", "fn", "fp", "tn", "recall", "precision", "f1", "fpr")}
            out[o]["fpr_ci"] = cb["ci"]
            out[o]["unsound_programs"] = int(sum(np.mean([r[o] for r in S if r["benchmark"] == b and r["kind"] == "correct"]) > 0.05 for b in BENCH))
        (summ["overall"] if scope == "overall" else summ["by_level"].setdefault(scope, {})).update(out)
    save(summ, os.path.join(d, "summary.json"))
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write(f"RQ1/RQ4 UNDER COHERENT NOISE (eta = {ETA}; RX(eps) after 1q gates, RZZ(5 eps) after 2q gates)\n")
        for scope, v in [("overall", summ["overall"])] + list(summ["by_level"].items()):
            f.write(f"\n[{scope}]\n")
            for o, x in v.items():
                f.write(f"  {o:13s} FPR={x['fpr']:.4f} [{x['fpr_ci'][0]:.4f}, {x['fpr_ci'][1]:.4f}] unsound programs={x['unsound_programs']}/9 "
                        f"recall={x['recall']:.3f} precision={x['precision']:.3f} F1={x['f1']:.3f}\n")
        f.write("\nDrift flagged by the homogeneity test (fraction of windows)\n")
        for k, v in drift.items():
            f.write(f"  {k}: {v:.2f}\n")
    write_meta(d, vars(a) | {"shots": SHOTS, "alpha": ALPHA, "eta": ETA, "eps": EPS, "executions": ex.n_executions}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
