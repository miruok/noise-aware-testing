# Running the hardware campaigns

Mir Mohammad Yousuf and Shabir Ahmad Sofi, National Institute of Technology Srinagar.

## Campaign HM (mini campaign): sized to fit the free IBM Quantum allowance

HM tests every research question on one IBM device with about half of the Open Plan
allowance (10 minutes of QPU time) and keeps the rest as a reserve. It is independent of
campaigns H1 to H3 below; run it first.

| Step | When | Circuits | Estimated QPU time | Research questions |
|---|---|---|---|---|
| `probe` | day 1 | 4 | about 7 s | measures QPU seconds per circuit; sizes the next steps |
| `a` | day 1, right after the probe | 128 | about 3.5 min | RQ1 soundness, RQ2 mutants and taxonomy, RQ3 ZNE attribution, RQ4 detection |
| `b` | at least 24 h after `a` (better: after the next device calibration, or in the next allowance period) | 34 | about 1 min | RQ5 drift and the sentinel, judged against the calibration of `a` |
| `c` | any time | 164 | about 3 min | larger programs (Bell-8, GHZ-8, QAOA-8, HWE-6) and the 8 observable Bugs4Q bugs on hardware |
| `d` | about 48 h after `a` | 74 | about 1.5 min | RQ5 third session: sentinel, full recalibration with K = 8 runs, stale vs recalibrated reference |

Benchmarks: Bell, GHZ-3, QFT-3, QAOA-3, VQE-3. Step `a` executes, per benchmark, 8
calibration runs, 1 single-reference run, 4 correct test runs, up to 3 non-equivalent
injected faults, 2 semantic mutants, 2 noise-amplifying mutants (`cx_pair`), 1
noise-neutral mutant, and the reference and the four non-neutral mutants at noise scale 3.
Step `b` executes 2 sentinel runs, 3 correct runs and 2 faults per benchmark.

Steps, from the package root (setup as below):
```bash
python hardware/run_mini.py plan                      # budget; no account needed
python hardware/run_mini.py a --dry-run               # whole pipeline on a fake backend
python hardware/run_mini.py probe --backend ibm_fez   # use one backend for all steps
python hardware/run_mini.py plan                      # re-estimates with the measured rate
python hardware/run_mini.py a --backend ibm_fez
python hardware/analyze_mini.py                       # results so far
# at least 24 hours later:
python hardware/run_mini.py b --backend ibm_fez
python hardware/analyze_mini.py
python hardware/run_mini.py c --backend ibm_fez       # any time
# about 48 hours after step a:
python hardware/run_mini.py d --backend ibm_fez
python hardware/analyze_mini.py                       # final results
```
* Every step refuses to submit if its estimate exceeds the remaining allowance (minus a
  60 s reserve). If it does not fit, it drops the scale-3 circuits first, then the mutants,
  one benchmark at a time, and records what it dropped.
* The QPU seconds of each job are read from IBM Quantum and stored in `raw.json`. If they
  cannot be read, check the job on the IBM Quantum dashboard.
* All circuits of a step are interleaved in a seeded random order inside one job, and
  step `b` reuses the qubit layout of step `a`.
* Results: `results/hardware_analysis/real/hm/REPORT.txt`, `analysis.json` and
  `hm_table.tex` (the results as a LaTeX table).

## Campaigns H1 to H3

QAOA-3) and H3 (attribution on GHZ-3 and QAOA-3) need about 6 to 7 minutes of QPU time
in addition to HM, so they are run on an IBM Quantum instance with enough remaining QPU time.
They use the same noise-aware layout choice as HM, keep the layout of the first campaign for all later
campaigns and sessions, interleave circuits in random order, record QPU seconds, and
refuse to submit when the account's remaining allowance (minus a 60 s reserve) is too small.

To keep the bookkeeping of H1 to H3 separate from HM, save the instance used for them under
its own name, once:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<API key>", instance="<CRN>", name="h-campaigns", overwrite=True)
```
Then, with `--account second` on every command and the same backend throughout:
```bash
python hardware/run_hardware.py plan
python hardware/run_hardware.py h1 --backend ibm_kingston --account h-campaigns   # about 3 min
python hardware/run_hardware.py h3 --backend ibm_kingston --account h-campaigns   # about 1 min
python hardware/run_hardware.py h2 --backend ibm_kingston --account h-campaigns   # about 20 s; one session
python hardware/analyze_hardware.py
```
Run `h2` 8 times, every 2 to 3 hours across two days, so that at least one session falls on a
different calibration day. Steps of HM keep using the default account (no `--account`).

### What H1 to H3 execute
* `h1` mutation campaign: per benchmark, 8 calibration runs, 10 correct runs, 20
  semantic mutants, 10 equivalent mutants (negative controls) and every
  non-equivalent injected fault.
* `h2` drift session: 8 calibration runs of each circuit; repeated sessions give the
  stability of the output distribution over hours and days.
* `h3` attribution: noise-amplifying mutants and semantic mutants, each folded
  at noise scales 1 and 3, analyzed by zero-noise extrapolation.

All circuits of a campaign are transpiled once with a fixed seed onto one initial
layout per benchmark and submitted as a single job.
