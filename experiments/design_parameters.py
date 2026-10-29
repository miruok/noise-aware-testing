# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Design study (parameter selection, Section 4.7): variance-based (Sobol) sensitivity of oracle
parameters, computed by functional ANOVA on complete factorial grids.

(a) SPC envelope: K in {4, 8, 12, 16, 20} x k in {2, 3, 4, 5, 6} x tau_min in
    {0.005, 0.01, 0.02, 0.03, 0.05} (125 configurations) on 9 benchmarks x 5
    noise levels, 5 windows per cell. Each window draws 20 calibration runs,
    1 reference, 5 correct runs and every non-equivalent injected fault once;
    all configurations are evaluated on the same executions (paired).
(b) NAO: alpha in {1e-2, 1e-3, 1e-4} x K in {0 (NAO-1), 4, 8, 16} x shots in
    {1024, 4096, 16384}, 5 benchmarks at medium noise, 8 windows.
Responses: FPR on correct runs, recall on non-equivalent faults.
Output: results/design_parameters/.
"""
from __future__ import annotations
import argparse, itertools, os, time
import numpy as np
from common import BENCH, NOISE, SHOTS, out_dir, save, write_meta, get_benchmark
from nisq_framework import Executor, compute_threshold, spc_envelope_threshold, jsd, pooled, inject_all, make_noisy_backend
from nisq_framework.stats import sobol_factorial

GRID_SPC = {"K": [4, 8, 12, 16, 20], "k": [2, 3, 4, 5, 6], "tau_min": [0.005, 0.01, 0.02, 0.03, 0.05]}
BENCH_NAO = ["bell", "ghz_3", "qaoa_3", "bell_8", "qaoa_8"]


def part_spc(windows, seed):
    ex = Executor(seed=seed); cells = []
    for b in BENCH:
        qc, _ = get_benchmark(b)
        faults = [r for r in inject_all(qc) if r["applicable"] and not r["output_equivalent"]]
        for nl, lv in NOISE.items():
            be = make_noisy_backend(lv)
            acc = {cfg: [0, 0, 0, 0] for cfg in itertools.product(*GRID_SPC.values())}
            for _ in range(windows):
                cal = ex.counts_many(qc, be, SHOTS, 20); ref = ex.counts(qc, be, SHOTS)
                d_ok = [jsd(ref, c) for c in ex.counts_many(qc, be, SHOTS, 5)]
                d_f = [jsd(ref, ex.counts(f["circuit"], be, SHOTS)) for f in faults]
                for (K, k, tm) in acc:
                    tau = spc_envelope_threshold(cal[:K], k, tm); x = acc[(K, k, tm)]
                    x[0] += sum(v > tau for v in d_ok); x[1] += len(d_ok); x[2] += sum(v > tau for v in d_f); x[3] += len(d_f)
            for (K, k, tm), x in acc.items():
                cells.append({"benchmark": b, "noise": nl, "K": K, "k": k, "tau_min": tm,
                              "fpr": x[0] / x[1], "recall": x[2] / x[3] if x[3] else None, "fp": x[0], "n_ok": x[1], "det": x[2], "n_fault": x[3]})
        print("SPC grid", b, "done", flush=True)
    conf = {}
    for c in cells:
        x = conf.setdefault((c["K"], c["k"], c["tau_min"]), [0, 0, 0, 0])
        x[0] += c["fp"]; x[1] += c["n_ok"]; x[2] += c["det"]; x[3] += c["n_fault"]
    rows = [{"K": K, "k": k, "tau_min": tm, "fpr": v[0] / v[1], "recall": v[2] / v[3]} for (K, k, tm), v in conf.items()]
    cl = [c for c in cells if c["recall"] is not None]
    return cells, rows, {"config_level": {r: sobol_factorial(rows, list(GRID_SPC), r) for r in ("fpr", "recall")},
                         "cell_level": {r: sobol_factorial(cl, list(GRID_SPC) + ["benchmark", "noise"], r) for r in ("fpr", "recall")}}


def part_nao(windows, seed, B):
    ex = Executor(seed=seed); rng = np.random.default_rng(seed); be = make_noisy_backend(0.005)
    circ = {b: get_benchmark(b)[0] for b in BENCH_NAO}
    faults = {b: [r for r in inject_all(qc) if r["applicable"] and not r["output_equivalent"]] for b, qc in circ.items()}
    rows = []
    for alpha, K, shots in itertools.product([1e-2, 1e-3, 1e-4], [0, 4, 8, 16], [1024, 4096, 16384]):
        fp = nt = det = nf = 0
        for b, qc in circ.items():
            for _ in range(windows):
                if K == 0:
                    ref = ex.counts(qc, be, shots)
                    tau = compute_threshold([], shots, alpha, "single", n_qubits=qc.num_qubits, B=B, reference_counts=ref, rng=rng).tau
                else:
                    cal = ex.counts_many(qc, be, shots, K); ref = pooled(cal)
                    tau = compute_threshold(cal, shots, alpha, "pooled", n_qubits=qc.num_qubits, B=B, rng=rng).tau
                for _ in range(4):
                    fp += jsd(ref, ex.counts(qc, be, shots)) > tau; nt += 1
                for f_ in faults[b]:
                    det += jsd(ref, ex.counts(f_["circuit"], be, shots)) > tau; nf += 1
        rows.append({"alpha": alpha, "K": K, "shots": shots, "fpr": fp / nt, "recall": det / nf, "tests": nt, "fault_evals": nf})
        print(rows[-1], flush=True)
    return rows, {r: sobol_factorial(rows, ["alpha", "K", "shots"], r) for r in ("fpr", "recall")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--windows-spc", type=int, default=5)
    ap.add_argument("--windows-nao", type=int, default=8); ap.add_argument("--seed-spc", type=int, default=29)
    ap.add_argument("--seed-nao", type=int, default=23); ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--out"); a = ap.parse_args()
    t0 = time.time(); d = out_dir("design_parameters", a.out)
    cells, rows, S = part_spc(a.windows_spc, a.seed_spc)
    save({"cells": cells, "configs": rows, "sobol": S}, os.path.join(d, "spc_envelope_grid.json"))
    nrows, NS = part_nao(a.windows_nao, a.seed_nao, a.B)
    save({"grid": nrows, "sobol": NS}, os.path.join(d, "nao_grid.json"))
    sound = [r for r in rows if r["fpr"] == 0]
    with open(os.path.join(d, "REPORT.txt"), "w") as f:
        f.write("PARAMETER SELECTION STUDY\n\n(a) SPC envelope grid (125 configurations)\n")
        f.write(f"  sound configurations (FPR = 0): {len(sound)}/{len(rows)}\n")
        for tm in GRID_SPC["tau_min"]:
            R = [r for r in rows if r["tau_min"] == tm]; Sd = [r for r in R if r["fpr"] == 0]
            f.write(f"  tau_min={tm:<6} unsound fraction={1 - len(Sd)/len(R):.2f}  mean recall (sound)={np.mean([r['recall'] for r in Sd]) if Sd else float('nan'):.3f}\n")
        for lvl in ("config_level", "cell_level"):
            for r, s in S[lvl].items():
                f.write(f"  Sobol {lvl} {r}: " + "  ".join(f"{k}: S1={v['S1']:.3f} ST={v['ST']:.3f}" for k, v in s.items() if k != "_variance") + "\n")
        f.write("\n(b) NAO grid (alpha x K x shots)\n")
        for x in nrows:
            f.write(f"  alpha={x['alpha']:<7g} K={x['K']:<3d} shots={x['shots']:<6d} FPR={x['fpr']:.4f} recall={x['recall']:.3f}\n")
        for r, s in NS.items():
            f.write(f"  Sobol {r}: " + "  ".join(f"{k}: S1={v['S1']:.3f} ST={v['ST']:.3f}" for k, v in s.items() if k != "_variance") + "\n")
    write_meta(d, vars(a) | {"shots_spc": SHOTS}, t0)
    print(open(os.path.join(d, "REPORT.txt")).read())


if __name__ == "__main__":
    main()
