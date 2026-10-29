# Noise-Aware Testing of NISQ Quantum Programs: replication package

**Authors:** Mir Mohammad Yousuf ([ORCID 0009-0001-7453-3358](https://orcid.org/0009-0001-7453-3358)) and
Shabir Ahmad Sofi ([ORCID 0000-0002-8740-2875](https://orcid.org/0000-0002-8740-2875)),
Department of Information Technology, National Institute of Technology Srinagar, India.

**Contact:** Mir Mohammad Yousuf, yousuf_2022phaite006@nitsri.ac.in

Replication package of the paper *Noise-Aware Testing of NISQ Quantum Programs: Self-Baseline
Oracle with a Controlled False-Positive Rate* (M. M. Yousuf and S. A. Sofi).

This package contains the implementation, experiment drivers, raw results, tables and
figures for every research question of the paper, plus the scripts used for the IBM
Quantum hardware campaigns.

## What the noise-aware oracle does

The noise-aware oracle (NAO) is a differential test oracle. It compares a program under test P' with a trusted
reference program P executed on the same backend (regression and mutation testing).
A correct run and the reference are multinomial samples of one output distribution, so
the null distribution of their Jensen-Shannon divergence is known from finite-shot
statistics:

    8 ln2 * JSD(p_ref, p_test) / (1/n_ref + 1/n_test)  ~  chi2(d - 1)

The acceptance threshold is the (1 - alpha) quantile of this null, estimated by a
parametric bootstrap (with a Chao1 correction for outcomes not yet observed), so the
false-positive rate is a nominal level set in advance. NAO pools K calibration runs into a
high-precision reference and runs a chi-square homogeneity test across them to detect
drift (for example noise changes or recompilation); only then is the threshold widened.
NAO-1 is the single-reference variant (one extra execution). Recall follows from the
noncentral chi-square (`nisq_framework/power.py`).

The package also provides: a mutation engine with exact statevector labels and a
three-class taxonomy (semantic, noise-amplifying, noise-neutral), eight
fault injectors with equivalence labels, a hardware-executable bug-noise attribution
by zero-noise extrapolation, and the baselines used in the paper (SPC envelope, fixed
threshold against an ideal reference, QOIN-style noise filter, simulator re-execution
triage).

## Layout

```
nisq_framework/                    library
  execution.py           seeded, layout-stable execution; integer counts
  divergence.py          base-2 Jensen-Shannon divergence
  threshold.py           null-quantile threshold, homogeneity (drift) test, SPC envelope
  power.py               analytic detection power, minimum detectable JSD, shot budget
  semantics.py           exact ideal semantics of measured circuits (statevector)
  mutation.py            mutation operators and taxonomy
  faults.py              fault injectors with equivalence labels
  oracles.py             NAO, NAO-1 and baseline oracles
  attribution.py         zero-noise-extrapolation attribution; re-execution triage
  benchmarks.py, noise.py, stats.py
experiments/             one driver per research question + tables, figures, checks
hardware/                IBM Quantum campaigns and their analysis
tests/                   pytest suite (23 tests)
results/                 raw records, summaries, SUMMARY.md, tables/ (LaTeX), figures/ (PDF, PNG)
docs/HARDWARE.md         how to run the hardware campaigns
```

## Installation

```bash
python -m venv venv && source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                    # 23 passed, about 10 s
```

Results were produced with Python 3.13, Qiskit 2.6.0, Qiskit Aer 0.18.0 and
qiskit-ibm-runtime 0.50.0; each `results/*/meta.json` records the exact environment,
configuration, seeds and number of executions.

## Reproducing the results

```bash
cd experiments
python run_all.py          # all experiments, tables, figures and checks, about 13 hours on 2 CPU cores
python make_tables.py      # results/SUMMARY.md and results/tables/*.tex
python make_figures.py     # results/figures/*.pdf, *.png
python verify_results.py   # recomputes headline numbers from raw records
python check_paper_tables.py <paper>/sections   # every number in the paper's result tables against results/tables/
```

Published results in `results/<experiment>/` are never overwritten; new runs are
written to `results/reruns/<experiment>/<timestamp>/`. All drivers use fixed master
seeds, so reruns on the same software versions reproduce the published numbers.

| Research question (paper section) | Driver | Results |
|---|---|---|
| RQ1 Does the oracle control its false-positive rate at the nominal level? (5.1) | `rq1_soundness.py`, `rq1_snapshots.py`, `rq1_sparsity.py --shape flat`, `--shape skewed`, `--shape headtail`, `--shape summary`, `rq1_bootstrap_stability.py` | `results/rq1_soundness/`, `results/rq1_snapshots/`, `results/rq1_sparsity/`, `results/rq1_bootstrap_stability/` |
| RQ2 Equivalent mutants, taxonomy and mutation score (5.2) | `rq2_equivalence.py`, `rq2_mutation.py`, `rq2_amplification.py`, `taxonomy_compilation.py` (physical property of each class after compilation) | `results/rq2_equivalence/`, `results/rq2_mutation/`, `results/rq2_amplification/`, `results/taxonomy_compilation/` |
| RQ3 Can failures be attributed to faults or noise without a noiseless simulator? (5.3) | `rq3_attribution.py`, `rq3_extrapolation.py` | `results/rq3_attribution/`, `results/rq3_extrapolation/` |
| RQ4 Fault detection, power model and scaling (5.4) | `rq4_faults.py`, `rq4_scaling.py`, `rq4_global_threshold.py` | `results/rq4_faults/`, `results/rq4_scaling/`, `results/rq4_global_threshold/` |
| RQ1, RQ2, RQ4 on 32 QASMBench programs (extended suite) | `ext_suite.py --part thresholds`, `--part detection --chunk i --chunks 4` (i = 0..3), `--part mutation`, `--part summary` | `results/ext_suite/` |
| RQ4 noise-calibrated global threshold on held-out programs (leave one program out) | `rq4_global_lopo.py --suite core`, `--suite qasmbench` | `results/rq4_global_lopo/` |
| RQ1 and RQ4 under coherent over-rotation errors | `rq_coherent.py` | `results/rq_coherent/` |
| RQ4 ablation of the oracle's components | `rq4_ablation.py --part synthetic`, `--part snapshots`, `--part summary` | `results/rq4_ablation/` |
| RQ4 real bugs from Bugs4Q | `rq4_real_bugs.py` (circuits in `nisq_framework/real_bugs.py`) | `results/rq4_real_bugs/` |
| RQ5 Drift and cost (5.5) | `rq5_drift_cost.py`, `rq5_sentinel.py` | `results/rq5_drift_cost/`, `results/rq5_sentinel/` |
| Parameter selection study (4.7) | `design_parameters.py` | `results/design_parameters/` |
| Supplementary analyses of existing records: RQ1 per program and per window, RQ3 by noise level on rejected cases, RQ4 power model by stratum | `supplementary_analyses.py` | `results/supplementary/` |
| Hardware | `hardware/run_hardware.py`, `hardware/analyze_hardware.py` | `results/hardware_raw/`, `results/hardware_analysis/` |
| Hardware, folding check of the attribution circuits (scale factor realized after transpilation) | `hardware/folding_check.py` | `results/hardware_analysis/folding_check/` |
| Hardware, mini campaign HM (fits the free 10-minute allowance) | `hardware/run_mini.py`, `hardware/analyze_mini.py` | `results/hardware_raw/hm/`, `results/hardware_analysis/real/hm/` |

## Benchmarks and noise

Nine programs (Bell, GHZ-3, QFT-3, QAOA-3, VQE-3, Bell-8, GHZ-8, QAOA-8, HWE-6) plus
GHZ and QAOA families up to 16 qubits; depolarising noise with level eta (single-qubit
eta, two-qubit min(10 eta, 0.5), readout min(2 eta, 0.5)) at eta in {0.0005, 0.001,
0.005, 0.02, 0.05}; three device-snapshot backends (FakeManilaV2, FakeLagosV2,
FakeKolkataV2); IBM Quantum hardware.

## Oracle and class names in result files

The JSON and `REPORT.txt` files use short keys; the paper uses the names on the right.

| Key | Name in the paper | Meaning |
|---|---|---|
| `nao`, `nao_a0.001`, `nao_a0.01` | NAO | noise-aware oracle, pooled reference of K = 8 calibration runs, null-quantile threshold at the given alpha (default 1e-3), with drift test |
| `nao1`, `nao1_a0.001`, `nao1_a0.01` | NAO-1 | single-reference variant (K = 1) |
| `nao_marginal` | marginal NAO | NAO on all two-qubit marginals (Bonferroni), scaling study |
| `spc_envelope` | SPC envelope | baseline: threshold max(mu + k sigma, tau_min) over K = 12 calibration runs |
| `fixed_<tau>` | fixed threshold | baseline: fixed tau against a noiseless (ideal) reference |
| `qoin_style` | QOIN-style filter | baseline: per-outcome noise correction, then tau = 0.05 against the ideal output |
| `global_noise_calibrated`, `noise_calibrated` | noise-calibrated global threshold | baseline after Fortz et al.: one threshold per noise model for all programs |
| `zne_linear`, `zne_quadratic`, `exponential` | ZNE attribution | zero-noise-extrapolation attribution (linear, quadratic or exponential) |
| `reexecution_triage` | re-execution triage | simulator-based attribution baseline |
| `semantic`, `noise_amplifying`, `noise_neutral` | taxonomy classes | semantic, noise-amplifying and noise-neutral mutants |
| `semantic_additive`, `semantic_subtractive` | operator families | additive and subtractive families of the semantic class |
| `A` to `F` (`rq4_ablation`) | ablation configurations A to F | A: ideal reference, fixed tau = 0.02; B: one self-baseline run, fixed tau = 0.02; C: one self-baseline run, SPC envelope; D: one self-baseline run, null quantile (NAO-1); E: 8 pooled runs, null quantile, no drift test; F: 8 pooled runs, null quantile, drift test (NAO) |

## Third-party programs

`benchmarks/qasmbench/` contains 32 unmodified OpenQASM programs from QASMBench
(https://github.com/pnnl/QASMBench), redistributed under its license
(`benchmarks/qasmbench/LICENSE_QASMBench`). `nisq_framework/real_bugs.py`
transcribes ten bugs from Bugs4Q (https://github.com/Z-928/Bugs4Q, Apache 2.0);
each entry records its Bugs4Q path and original source.

## License and citation

MIT License (`LICENSE`). Citation metadata: `CITATION.cff`.
