# Probe on ibm_fez (16 June 2026)

Bell and GHZ-3, 4096 shots, one session. Per circuit: eight calibration runs, one
reference run and three test runs judged against that reference (configuration
`noise_aware`), and three test runs for each of three fixed thresholds judged against
the ideal output (configurations `fixed`); 42 executions in total.
`rq1_real_per_cell.json` records the divergence of every test run from its reference.
The paper compares the divergences from the reference run with the closed-form NAO-1
threshold (`nisq_framework.threshold.chi2_threshold`); the pass/fail flags in the file
belong to the envelope threshold used when the probe was executed and are not used.
