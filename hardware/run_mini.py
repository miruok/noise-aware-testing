# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Campaign HM (mini campaign): a hardware validation sized for the IBM Quantum
Open Plan (10 minutes of QPU time). It complements campaigns H1 to H3
(run_hardware.py) and tests every research question on one device with
about half of the allowance, keeping the rest as a reserve.

Three steps, run in this order:
  probe  4 circuits (Bell and GHZ-3, two runs each). Measures the QPU seconds
         per circuit on the chosen backend; every later step is sized with it.
  a      one job on day 1. Per benchmark (Bell, GHZ-3, QFT-3, QAOA-3, VQE-3):
           8 calibration runs                       NAO reference (RQ1, RQ4)
           1 single-reference run                   NAO-1 and SPC reference
           4 correct test runs                      soundness (RQ1)
           3 non-equivalent injected faults         detection (RQ4)
           1 additive + 1 subtractive semantic mutant   mutation score (RQ2)
           2 noise-amplifying mutants (cx_pair)     taxonomy (RQ2)
           1 noise-neutral mutant                   taxonomy (RQ2)
           P and the 4 non-neutral mutants at fold scale 3   attribution (RQ3)
         = 26 circuits per benchmark, 130 in total.
  b      one job at least 24 hours after a (ideally after the device's next
         calibration, or in the next allowance period). Per benchmark:
           2 sentinel runs, 3 correct test runs, the first 2 faults of a
         = 7 circuits per benchmark, 35 in total, judged against the
         calibration of step a (RQ5: drift and the sentinel).

  c      one job (any day). Larger programs and real bugs on hardware:
           Bell-8, GHZ-8, QAOA-8, HWE-6: 8 calibration runs, 4 correct runs,
             3 non-equivalent injected faults, 1 additive + 1 subtractive
             semantic mutant (17 circuits each, 68 in total)
           8 observable Bugs4Q bugs: 8 calibration runs and 2 test runs of the
             fixed program, 2 runs of the buggy program (12 each, 96 in total)
  d      one job about 48 hours after a (third session). Per HM benchmark:
           2 sentinel runs, 8 new calibration runs, 3 correct test runs and the
           first 2 faults of a (15 each, 75 in total); tests the sentinel
           policy with a full recalibration of K = 8 runs.

Steps a, b and d use the same layout per benchmark (chosen in a); step c
chooses its own layouts. Every step reuses the initial layout of step a (fixed compilation policy),
interleaves all circuits in a seeded random order inside one job, and refuses
to submit when its estimate would exceed the remaining allowance; if it does
not fit, the lowest-priority items (first the scale-3 circuits, then the
mutants) are dropped until it fits.

Usage (from the package root):
  python hardware/run_mini.py plan                       # budget, no account needed
  python hardware/run_mini.py a --dry-run                # full pipeline on a fake backend
  python hardware/run_mini.py b --dry-run
  python hardware/run_mini.py probe --backend ibm_fez    # real hardware, about 5-10 s
  python hardware/run_mini.py a --backend ibm_fez        # about 3-4 min
  python hardware/run_mini.py b --backend ibm_fez        # about 1 min, >= 24 h later
  python hardware/run_mini.py c --backend ibm_fez        # about 3 min, any time
  python hardware/run_mini.py d --backend ibm_fez        # about 1.5 min, ~48 h after a
  python hardware/analyze_mini.py                        # tables for the paper
Raw data: results/hardware_raw/hm/<step>/<stamp>_<backend>/raw.json
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from qiskit import qpy, transpile
from nisq_framework import get_benchmark, MutationEngine, NOISE_AMPLIFYING, NOISE_NEUTRAL, SEM_ADD, SEM_SUB, inject_all, fold, ideal_jsd
from nisq_framework.real_bugs import real_bug_pairs
from run_hardware import get_backend, backend_snapshot, submit, SEC_PER_SHOT, SEC_PER_CIRCUIT

BENCH = ["bell", "ghz_3", "qft_3", "qaoa_3", "vqe_3"]
LARGE = ["bell_8", "ghz_8", "qaoa_8", "hwe_6"]
STEPS = ("probe", "a", "b", "c", "d")
REFS = {}                  # reference circuit per program key (benchmarks and real bugs)


def reference(key):
    return REFS[key] if key in REFS else get_benchmark(key)[0]
FAULT_ORDER = ["wrong_angle", "extra_x", "remove_entanglement", "wrong_basis", "drop_last_gate", "swap_qubits", "wrong_target", "phase_flip"]
ALLOWANCE = 600.0          # Open Plan: 10 minutes of QPU time
RESERVE = 60.0             # never plan to use the last minute
RAW = os.path.join(ROOT, "results", "hardware_raw", "hm")
SEED = 2026


def faults_for(qc):
    inj = {r["injector"]: r for r in inject_all(qc) if r["applicable"] and not r["output_equivalent"]}
    return [inj[k] for k in FAULT_ORDER if k in inj][:3]


def mutants_for(qc, seed):
    eng = MutationEngine(qc, seed=seed)
    sem = eng.generate(1, SEM_ADD) + eng.generate(1, SEM_SUB)
    # inverse pairs of single-qubit Z, S and T gates are virtual on IBM devices; cx_pair adds physical error
    amp = [m for m in eng.generate(20, NOISE_AMPLIFYING) if m.operator == "cx_pair"][:2]
    neu = eng.generate(1, NOISE_NEUTRAL)
    return sem, amp, neu


def item(b, role, circuit, truth, priority, **kw):
    return {"benchmark": b, "role": role, "circuit": circuit, "truth": truth, "priority": priority, **kw}


def build_a(bench):
    items = []
    for i, b in enumerate(bench):
        qc, _ = get_benchmark(b)
        items += [item(b, "calibration", qc, "no_fault", 1, idx=k) for k in range(8)]
        items.append(item(b, "single_reference", qc, "no_fault", 1, scale=1, case="P"))
        items += [item(b, "correct_test", qc, "no_fault", 1, idx=k) for k in range(4)]
        for j, r in enumerate(faults_for(qc)):
            items.append(item(b, "injected", r["circuit"], "fault", 1, operator=r["injector"], ideal_jsd=r["ideal_jsd"], fault_idx=j))
        sem, amp, neu = mutants_for(qc, SEED + i)
        cases = []
        for j, m in enumerate(sem + amp):
            case = f"T{j}"
            items.append(item(b, "mutant", m.circuit, "fault" if m.semantic_fault else "no_fault", 2, family=m.family,
                              operator=m.operator, ideal_jsd=m.ideal_jsd, scale=1, case=case))
            cases.append((case, m))
        for m in neu:
            items.append(item(b, "mutant", m.circuit, "fault" if m.semantic_fault else "no_fault", 2, family=m.family,
                              operator=m.operator, ideal_jsd=m.ideal_jsd))
        items.append(item(b, "zne", fold(qc, 3), "no_fault", 3, scale=3, case="P"))
        for case, m in cases:
            items.append(item(b, "zne", fold(m.circuit, 3), "fault" if m.semantic_fault else "no_fault", 3, family=m.family,
                              operator=m.operator, ideal_jsd=m.ideal_jsd, scale=3, case=case))
    return items


def build_b(bench):
    items = []
    for b in bench:
        qc, _ = get_benchmark(b)
        items += [item(b, "sentinel", qc, "no_fault", 1, idx=k) for k in range(2)]
        items += [item(b, "correct_test", qc, "no_fault", 1, idx=k) for k in range(3)]
        for j, r in enumerate(faults_for(qc)[:2]):
            items.append(item(b, "injected", r["circuit"], "fault", 1, operator=r["injector"], ideal_jsd=r["ideal_jsd"], fault_idx=j))
    return items


def build_c():
    items = []
    for i, b in enumerate(LARGE):
        qc, _ = get_benchmark(b)
        items += [item(b, "calibration", qc, "no_fault", 1, idx=k) for k in range(8)]
        items += [item(b, "correct_test", qc, "no_fault", 1, idx=k) for k in range(4)]
        for j, r in enumerate(faults_for(qc)):
            items.append(item(b, "injected", r["circuit"], "fault", 1, operator=r["injector"], ideal_jsd=r["ideal_jsd"], fault_idx=j))
        eng = MutationEngine(qc, seed=SEED + 100 + i)
        for m in eng.generate(1, SEM_ADD) + eng.generate(1, SEM_SUB):
            items.append(item(b, "mutant", m.circuit, "fault" if m.semantic_fault else "no_fault", 2, family=m.family,
                              operator=m.operator, ideal_jsd=m.ideal_jsd))
    for p in real_bug_pairs():
        d = ideal_jsd(p["fixed"], p["buggy"])
        if d < 1e-10:
            continue                       # not observable on |0...0>: excluded, as in the simulation study
        key = "rb_" + p["id"]; REFS[key] = p["fixed"]
        items += [item(key, "calibration", p["fixed"], "no_fault", 1, idx=k, bug=p["id"]) for k in range(8)]
        items += [item(key, "correct_test", p["fixed"], "no_fault", 1, idx=k, bug=p["id"]) for k in range(2)]
        items += [item(key, "real_bug", p["buggy"], "fault", 1, idx=k, bug=p["id"], kind=p["kind"], ideal_jsd=d) for k in range(2)]
    return items


def build_d(bench):
    items = []
    for b in bench:
        qc, _ = get_benchmark(b)
        items += [item(b, "sentinel", qc, "no_fault", 1, idx=k) for k in range(2)]
        items += [item(b, "recalibration", qc, "no_fault", 1, idx=k) for k in range(8)]
        items += [item(b, "correct_test", qc, "no_fault", 1, idx=k) for k in range(3)]
        for j, r in enumerate(faults_for(qc)[:2]):
            items.append(item(b, "injected", r["circuit"], "fault", 1, operator=r["injector"], ideal_jsd=r["ideal_jsd"], fault_idx=j))
    return items


def build_probe():
    return [item(b, "probe", get_benchmark(b)[0], "no_fault", 1, idx=k) for b in ("bell", "ghz_3") for k in range(2)]


def records(step, dry):
    out = []
    for p in sorted(glob.glob(os.path.join(RAW, step, "*", "raw.json"))):
        r = json.load(open(p))
        if bool(r.get("dry_run")) == dry:
            out.append(r)
    return out


def used_seconds(dry):
    return sum(r.get("quantum_seconds") or r["estimated_seconds"] for s in STEPS for r in records(s, dry))


def sec_per_circuit(dry, shots):
    """QPU seconds per circuit, measured over every completed job (the probe
    alone overestimates it, because per-job overhead is spread over 4 circuits)."""
    done = [r for s in STEPS for r in records(s, dry) if r.get("quantum_seconds")]
    if done:
        q = sum(r["quantum_seconds"] * shots / r["shots"] for r in done); n = sum(len(r["items"]) for r in done)
        return q / n, f"measured over {len(done)} completed job(s), {n} circuits"
    return shots * SEC_PER_SHOT + SEC_PER_CIRCUIT, "conservative default (run the probe first)"


def fit(items, rate, remaining):
    dropped = []
    for pr in (3, 2):
        while len(items) * rate > remaining and any(it["priority"] == pr for it in items):
            # drop a whole benchmark's items of this priority at a time, last benchmark first
            b = [it["benchmark"] for it in items if it["priority"] == pr][-1]
            dropped += [f"{b}:{it['role']}" for it in items if it["priority"] == pr and it["benchmark"] == b]
            items = [it for it in items if not (it["priority"] == pr and it["benchmark"] == b)]
    return items, dropped


def layouts_from_a(backend_name, dry):
    ra = [r for r in records("a", dry) if r["backend"]["name"] == backend_name]
    if not ra:
        raise SystemExit(f"steps b and d need step a on {backend_name} first")
    return {k: v for k, v in ra[-1]["initial_layouts"].items()}, ra[-1]["submitted"]


def success_estimate(isa, target):
    """Product of (1 - reported error) over all gates and measurements of a
    transpiled circuit: the backend's own estimate of its success probability."""
    p = 1.0
    for inst in isa.data:
        name = inst.operation.name
        if name in ("barrier", "delay") or name not in target.operation_names:
            continue
        qargs = tuple(isa.find_bit(q).index for q in inst.qubits)
        props = target[name].get(qargs)
        if props is not None and props.error is not None:
            p *= max(0.0, 1.0 - props.error)
    return p


def choose_layout(ref, backend, seed_transpiler=11, tries=24):
    """Noise-aware layout: among the layouts the transpiler proposes for several
    seeds and optimization levels, keep the one with the highest estimated
    success probability on the backend's current calibration."""
    best = None
    for lvl in (1, 3):
        for s in range(tries // 2):
            t = transpile(ref, backend, optimization_level=lvl, seed_transpiler=seed_transpiler + s)
            lay = [int(x) for x in t.layout.initial_index_layout()[: ref.num_qubits]]
            t1 = transpile(ref, backend, optimization_level=1, seed_transpiler=seed_transpiler, initial_layout=lay)
            sc = success_estimate(t1, backend.target)
            if best is None or sc > best[0]:
                best = (sc, lay)
    return best[1], best[0]


def transpile_items(items, backend, layouts, seed_transpiler=11):
    scores = {}
    for it in items:
        b = it["benchmark"]
        if b not in layouts:
            layouts[b], scores[b] = choose_layout(reference(b), backend, seed_transpiler)
            print(f"  layout {b}: qubits {layouts[b]}, estimated success probability {scores[b]:.3f}")
        it["isa"] = transpile(it["circuit"], backend, optimization_level=1, seed_transpiler=seed_transpiler, initial_layout=layouts[b])
    return layouts


def job_seconds(jid, dry):
    if dry:
        return None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        acc = os.environ.get("NAT_IBM_ACCOUNT")
        job = (QiskitRuntimeService(name=acc) if acc else QiskitRuntimeService()).job(jid)
        u = job.usage() if callable(getattr(job, "usage", None)) else None
        if u is None:
            u = job.metrics().get("usage", {}).get("quantum_seconds")
        return float(u) if u is not None else None
    except Exception as e:
        print(f"could not read QPU usage of job {jid}: {e!r}; record it from the IBM Quantum dashboard")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["plan", *STEPS])
    ap.add_argument("--backend", default=None); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shots", type=int, default=4096); ap.add_argument("--yes", action="store_true")
    ap.add_argument("--allowance", type=float, default=ALLOWANCE, help="QPU seconds available in total")
    ap.add_argument("--account", default=None, help="name of a saved IBM Quantum account (save_account(name=...))")
    a = ap.parse_args()
    if a.account:
        os.environ["NAT_IBM_ACCOUNT"] = a.account
    if a.step == "plan":
        rate, src = sec_per_circuit(a.dry_run, a.shots)
        tot = 0
        for s, f in (("probe", build_probe), ("a", lambda: build_a(BENCH)), ("b", lambda: build_b(BENCH)), ("c", build_c), ("d", lambda: build_d(BENCH))):
            n = len(f()); est = n * (a.shots * SEC_PER_SHOT + SEC_PER_CIRCUIT if s == "probe" else rate); tot += est
            print(f"{s:5s}: {n:3d} circuits x {a.shots} shots, about {est:.0f} s")
        print(f"total about {tot:.0f} s of {a.allowance:.0f} s ({src}); already used: {used_seconds(a.dry_run):.0f} s")
        return
    items = {"probe": build_probe, "a": lambda: build_a(BENCH), "b": lambda: build_b(BENCH), "c": build_c, "d": lambda: build_d(BENCH)}[a.step]()
    rate, src = sec_per_circuit(a.dry_run, a.shots)
    if a.step == "probe":
        rate = a.shots * SEC_PER_SHOT + SEC_PER_CIRCUIT
    remaining = a.allowance - RESERVE - used_seconds(a.dry_run)
    items, dropped = fit(items, rate, remaining)
    est = len(items) * rate
    print(f"step {a.step}: {len(items)} circuits, estimated {est:.0f} s ({src}); remaining before reserve {remaining:.0f} s")
    if dropped:
        print(f"  to fit the allowance, dropped {len(dropped)} low-priority circuits: {sorted(set(d.split(':')[0] + ' ' + d.split(':')[1] for d in dropped))}")
    if est > remaining or not items:
        raise SystemExit("not enough allowance left for this step")
    backend = get_backend(a.backend, a.dry_run)
    since = None
    if a.step in ("b", "d"):
        layouts, since = layouts_from_a(backend.name, a.dry_run)
    else:
        layouts = {}
    if not a.dry_run and not a.yes:
        if input(f"submit {len(items)} circuits to {backend.name}? [y/N] ").strip().lower() != "y":
            return
    layouts = transpile_items(items, backend, layouts)
    order = np.random.default_rng(SEED + len(items)).permutation(len(items))
    batch = [items[i] for i in order]
    t0 = time.time(); counts, jid = submit(batch, backend, a.shots)
    for it, c in zip(batch, counts):
        it["counts"] = c
    for pos, i in enumerate(order):
        items[i]["position_in_job"] = int(pos)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(RAW, a.step, f"{stamp}_{'dryrun_' if a.dry_run else ''}{backend.name}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "circuits.qpy"), "wb") as f:
        qpy.dump([it["circuit"] for it in items], f)
    with open(os.path.join(d, "isa_circuits.qpy"), "wb") as f:
        qpy.dump([it["isa"] for it in items], f)
    qs = job_seconds(jid, a.dry_run)
    meta = [{k: v for k, v in it.items() if k not in ("circuit", "isa")} | {"isa_depth": it["isa"].depth(),
             "isa_two_qubit_gates": sum(v for k, v in it["isa"].count_ops().items() if k in ("cz", "ecr", "cx"))} for it in items]
    json.dump({"campaign": "hm", "step": a.step, "backend": backend_snapshot(backend), "dry_run": a.dry_run, "job_id": jid,
               "shots": a.shots, "submitted": stamp, "calibration_from": since, "wall_seconds": round(time.time() - t0, 1),
               "estimated_seconds": round(est, 1), "quantum_seconds": qs, "dropped": dropped,
               "initial_layouts": layouts, "items": meta}, open(os.path.join(d, "raw.json"), "w"), indent=1)
    print(f"saved {len(items)} results -> {d}  (job {jid}; QPU seconds {qs if qs is not None else 'n/a'}; "
          f"allowance used so far about {used_seconds(a.dry_run):.0f} s)")


if __name__ == "__main__":
    main()
