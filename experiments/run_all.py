# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Run every experiment, then tables, figures and the integrity check
(about 13 hours on 2 CPU cores). Published results in results/<experiment>/
are never overwritten; new runs land in results/reruns/."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["rq2_equivalence.py", "rq1_soundness.py", "rq1_snapshots.py", ("rq1_sparsity.py", "--shape", "flat"), ("rq1_sparsity.py", "--shape", "skewed"),
         ("rq1_sparsity.py", "--shape", "headtail"), ("rq1_sparsity.py", "--shape", "summary"), "rq1_bootstrap_stability.py", "rq4_faults.py", "rq2_mutation.py", "taxonomy_compilation.py",
         "rq4_scaling.py", "design_parameters.py", "rq3_attribution.py", "rq5_drift_cost.py",
         "rq2_amplification.py", "rq3_extrapolation.py", "rq4_global_threshold.py", "rq5_sentinel.py",
         "rq4_real_bugs.py", ("rq4_global_lopo.py", "--suite", "core"), ("rq4_global_lopo.py", "--suite", "qasmbench"), "rq_coherent.py", ("rq4_ablation.py", "--part", "synthetic"), ("rq4_ablation.py", "--part", "snapshots"),
         ("rq4_ablation.py", "--part", "summary"), ("ext_suite.py", "--part", "thresholds")] + [("ext_suite.py", "--part", "detection", "--chunk", str(i), "--chunks", "4") for i in range(4)] + [
         ("ext_suite.py", "--part", "mutation"), ("ext_suite.py", "--part", "summary"),
         "supplementary_analyses.py", "make_tables.py", "make_figures.py", "verify_results.py"]
for s in STEPS:
    args = list(s) if isinstance(s, tuple) else [s]
    t = time.time(); print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)
    print(f"=== {s} done in {time.time() - t:.0f} s", flush=True)
