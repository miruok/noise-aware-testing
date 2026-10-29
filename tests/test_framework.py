# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
import numpy as np, pytest
from qiskit import transpile
from qiskit_aer import AerSimulator
from scipy.spatial.distance import jensenshannon
from nisq_framework import (Executor, jsd, MutationEngine, FAMILIES, NOISE_AMPLIFYING, EQUIVALENT_FAMILIES,
                  inject_all, fold, ideal_distribution, compute_threshold, chi2_threshold,
                  detection_power, make_noisy_backend, get_benchmark, BENCHMARKS)
from nisq_framework.semantics import check_terminal_measurements, first_measure_index
from nisq_framework.stats import sobol_factorial
from nisq_framework.divergence import jsd_arrays


def test_jsd_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p, q = rng.random(6), rng.random(6)
        p, q = p / p.sum(), q / q.sum()
        ref = jensenshannon(p, q, base=2) ** 2
        assert abs(jsd(dict(enumerate(p)), dict(enumerate(q)), smooth=False) - ref) < 1e-12


def test_executor_reproducible():
    qc, _ = get_benchmark("ghz_3"); b = make_noisy_backend(0.02)
    assert Executor(seed=5).counts(qc, b, 1000) == Executor(seed=5).counts(qc, b, 1000)


@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_mutants_labelled_and_placed(name):
    qc, _ = get_benchmark(name)
    eng = MutationEngine(qc, seed=3)
    for fam in FAMILIES:
        for m in eng.generate(6, fam):
            assert check_terminal_measurements(m.circuit)
            assert m.position <= first_measure_index(qc)
            if fam in EQUIVALENT_FAMILIES:
                assert m.output_equivalent and m.ideal_jsd < 1e-10
            else:
                assert not m.output_equivalent and m.ideal_jsd > 1e-10


def test_noise_amplifying_mutants_survive_transpilation():
    qc, _ = get_benchmark("ghz_3"); b = make_noisy_backend(0.005)
    base = sum(transpile(qc, b, optimization_level=1, seed_transpiler=11).count_ops().values())
    for m in MutationEngine(qc, seed=7).generate(12, NOISE_AMPLIFYING):
        t = transpile(m.circuit, b, optimization_level=1, seed_transpiler=11)
        assert sum(v for k, v in t.count_ops().items() if k != "barrier") > base, m.operator


def test_known_equivalent_injection():
    qc, _ = get_benchmark("qaoa_8")
    rec = {r["injector"]: r for r in inject_all(qc) if r["applicable"]}
    assert rec["swap_qubits"]["output_equivalent"]
    assert not rec["remove_entanglement"]["output_equivalent"]


def test_fold_preserves_semantics():
    for name in ["bell", "qaoa_3", "vqe_3"]:
        qc, _ = get_benchmark(name); p0 = ideal_distribution(qc)
        for s in (3, 5):
            assert jsd(p0, ideal_distribution(fold(qc, s)), smooth=False) < 1e-10


def test_fold_keeps_noise_amplifying_pairs():
    qc, _ = get_benchmark("ghz_3"); b = make_noisy_backend(0.005)
    m = MutationEngine(qc, seed=7).generate(1, NOISE_AMPLIFYING)[0]
    n = lambda t: sum(v for k, v in t.count_ops().items() if k not in ("barrier", "measure"))
    t0 = transpile(fold(qc, 3), b, optimization_level=1, seed_transpiler=11)
    t1 = transpile(fold(m.circuit, 3), b, optimization_level=1, seed_transpiler=11)
    assert n(t1) > n(t0)


def test_threshold_coverage_synthetic():
    rng = np.random.default_rng(0); N = 4096; alpha = 0.01
    p = rng.dirichlet(np.ones(16)); fp = 0; T = 300
    for _ in range(T):
        ref = {str(k): int(v) for k, v in enumerate(rng.multinomial(N, p)) if v}
        rep = compute_threshold([], N, alpha, "single", reference_counts=ref, B=2000, rng=rng)
        test = {str(k): int(v) for k, v in enumerate(rng.multinomial(N, p)) if v}
        fp += jsd(ref, test) > rep.tau
    assert fp / T < 3 * alpha + 0.01


def test_chi2_and_power_consistent():
    d, N = 16, 4096
    tau = chi2_threshold(d, N, N, 1e-3)
    rng = np.random.default_rng(2); p = np.ones(d) / d
    null = jsd_arrays(rng.multinomial(N, p, 20000), rng.multinomial(N, p, 20000))
    assert abs(np.quantile(null, 1 - 1e-3) - tau) / tau < 0.15
    assert detection_power(0.0, d, N, N, 1e-3) <= 1e-3 + 1e-9
    assert detection_power(0.05, d, N, N, 1e-3) > 0.99


def test_sobol_additive():
    rows = [{"a": a, "b": b, "y": a + 2 * b} for a in range(3) for b in range(3)]
    s = sobol_factorial(rows, ["a", "b"], "y")
    assert abs(s["a"]["S1"] - 0.2) < 1e-9 and abs(s["b"]["S1"] - 0.8) < 1e-9


def test_semantics_agrees_with_aer_sampling():
    sim = AerSimulator(seed_simulator=3)
    for name in ["ghz_3", "qaoa_3", "vqe_3"]:
        qc, _ = get_benchmark(name)
        for fam in FAMILIES:
            for m in MutationEngine(qc, seed=11).generate(2, fam):
                c = sim.run(transpile(m.circuit, sim, seed_transpiler=1), shots=100_000).result().get_counts()
                assert jsd(ideal_distribution(m.circuit), {k: v / 1e5 for k, v in c.items()}) < 2e-4


def test_exponential_extrapolation_recovers_ideal_under_global_depolarizing():
    import numpy as np
    from nisq_framework.attribution import _exp_extrapolate
    p0 = np.array([0.7, 0.3, 0.0, 0.0]); u = 0.25; f = 0.8
    X = np.stack([u + (p0 - u) * f ** s for s in (1, 3)])[None]
    assert np.allclose(_exp_extrapolate(X, (1, 3), 4)[0], p0)


def test_cell_level_tests():
    from nisq_framework.stats import cmh_test, signflip_test
    assert abs(signflip_test([0.2] * 9) - 2 / 512) < 1e-12
    assert cmh_test([((10, 0), (0, 10))] * 3) < 1e-6


def test_real_bug_catalogue():
    from nisq_framework.real_bugs import real_bug_pairs
    from nisq_framework import ideal_jsd
    pairs = real_bug_pairs()
    assert len(pairs) == 10
    observable = sum(ideal_jsd(p["fixed"], p["buggy"]) > 1e-10 for p in pairs)
    assert observable == 8


def test_qasmbench_suite_loads():
    from nisq_framework.benchmarks import QASMBENCH, load_qasmbench
    assert len(QASMBENCH) == 32
    assert all(load_qasmbench(n).num_qubits <= 10 for n in QASMBENCH)
