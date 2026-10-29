# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""
Hardware campaigns on IBM Quantum devices.

All circuits of a campaign are built locally, labelled with exact ideal
semantics, transpiled ONCE with a fixed seed onto ONE initial layout, and
submitted as a single SamplerV2 job (many PUBs), which minimises queue and
QPU overhead. Raw counts, circuits (QPY), job ids and a backend snapshot
are saved; analysis is offline (analyze_hardware.py).

Campaigns
  h1  mutation campaign: K=8 calibration runs, 10 correct tests,
      semantics-changing mutants, semantics-preserving negative controls,
      non-equivalent injected faults.
  h2  drift session: K=8 calibration runs of each circuit. Run this command
      repeatedly (e.g. every 2-3 hours over 2 days); each call is one
      timestamped session.
  h3  attribution: noise-amplifying mutants and semantic mutants,
      each folded at scales 1 and 3 (and the reference at both scales).

Usage
  python hardware/run_hardware.py plan                      # budget only
  python hardware/run_hardware.py h1 --dry-run              # local fake backend
  python hardware/run_hardware.py h1 --backend ibm_fez      # real hardware
  python hardware/run_hardware.py h2 --backend ibm_fez
  python hardware/run_hardware.py h3 --backend ibm_fez

Real runs need: pip install qiskit-ibm-runtime, and a saved account:
  from qiskit_ibm_runtime import QiskitRuntimeService
  QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                    token="<API key>", instance="<CRN>")
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from qiskit import qpy, transpile
from nisq_framework import get_benchmark, MutationEngine, NOISE_AMPLIFYING, NOISE_NEUTRAL, SEM_ADD, SEM_SUB, inject_all, fold

DEFAULT_BENCH = ["ghz_3", "qaoa_3", "bell_8"]
SEC_PER_SHOT = 0.00030      # conservative estimate incl. reset/readout; refine after first job
SEC_PER_CIRCUIT = 0.5       # per-PUB overhead estimate


def build_h1(bench, seed=42):
    items = []
    for b in bench:
        qc, _ = get_benchmark(b)
        items += [{"benchmark": b, "role": "calibration", "idx": i, "circuit": qc, "truth": "no_fault"} for i in range(8)]
        items += [{"benchmark": b, "role": "correct_test", "idx": i, "circuit": qc, "truth": "no_fault"} for i in range(10)]
        eng = MutationEngine(qc, seed=seed)
        for cat, n in ((SEM_ADD, 10), (SEM_SUB, 10), (NOISE_AMPLIFYING, 6), (NOISE_NEUTRAL, 4)):
            for m in eng.generate(n, cat):
                items.append({"benchmark": b, "role": "mutant", "family": cat, "operator": m.operator,
                              "ideal_jsd": m.ideal_jsd, "circuit": m.circuit,
                              "truth": "fault" if m.semantic_fault else "no_fault"})
        for r in inject_all(qc):
            if r["applicable"] and not r["output_equivalent"]:
                items.append({"benchmark": b, "role": "injected", "operator": r["injector"],
                              "ideal_jsd": r["ideal_jsd"], "circuit": r["circuit"], "truth": "fault"})
    return items


def build_h2(bench):
    items = []
    for b in bench:
        qc, _ = get_benchmark(b)
        items += [{"benchmark": b, "role": "calibration", "idx": i, "circuit": qc, "truth": "no_fault"} for i in range(8)]
    return items


def build_h3(bench, seed=42):
    items = []
    for b in bench:
        qc, _ = get_benchmark(b)
        for s in (1, 3):
            items.append({"benchmark": b, "role": "reference", "scale": s, "case": "P", "circuit": fold(qc, s), "truth": "no_fault"})
        eng = MutationEngine(qc, seed=seed + 1)
        # hardware-relevant noise-adding operators only (1q Z/S/T pairs are virtual on IBM devices)
        noise_cases = [m for m in eng.generate(12, NOISE_AMPLIFYING) if m.operator in ("cx_pair", "rotation_pair") or m.meta.get("pair") in ("h-h", "x-x", "y-y")][:5]
        sem = eng.generate(3, SEM_ADD) + eng.generate(2, SEM_SUB)
        for j, m in enumerate(noise_cases + sem):
            for s in (1, 3):
                items.append({"benchmark": b, "role": "attribution_case", "case": f"T{j}", "scale": s,
                              "family": m.family, "operator": m.operator, "ideal_jsd": m.ideal_jsd,
                              "circuit": fold(m.circuit, s), "truth": "bug" if m.semantic_fault else "noise"})
    return items


def budget(items, shots):
    n = len(items)
    return {"circuits": n, "shots_per_circuit": shots, "total_shots": n * shots,
            "estimated_qpu_seconds": round(n * shots * SEC_PER_SHOT + n * SEC_PER_CIRCUIT, 1)}


def get_backend(name, dry_run):
    if dry_run:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        return FakeTorino()
    from qiskit_ibm_runtime import QiskitRuntimeService
    acc = os.environ.get("NAT_IBM_ACCOUNT")
    svc = QiskitRuntimeService(name=acc) if acc else QiskitRuntimeService()
    return svc.backend(name) if name else svc.least_busy(operational=True, simulator=False)


RAW_H = os.path.join(ROOT, "results", "hardware_raw")
ALLOWANCE, RESERVE = 600.0, 60.0


def h_records(dry):
    out = []
    for c in ("h1", "h2", "h3"):
        for p in sorted(glob.glob(os.path.join(RAW_H, c, "*", "raw.json"))):
            r = json.load(open(p))
            if bool(r.get("dry_run")) == dry:
                out.append(r)
    return out


def used_seconds(dry):
    return sum(r.get("quantum_seconds") or r.get("estimated_seconds") or 0 for r in h_records(dry))


def saved_layouts(backend_name, dry):
    """Layouts of the first H campaign on this backend, reused by later ones so
    that every session executes on the same qubits."""
    for r in h_records(dry):
        if r["backend"]["name"] == backend_name and r.get("initial_layouts"):
            return {k: list(v) for k, v in r["initial_layouts"].items()}
    return {}


def transpile_batch(items, backend, layouts=None, seed_transpiler=11):
    """Noise-aware layout per benchmark (highest success probability estimated
    from the backend's reported errors, as in run_mini.py), reused across
    campaigns and sessions; every circuit of a benchmark uses that layout."""
    from run_mini import choose_layout
    layouts = dict(layouts or {})
    for it in items:
        b = it["benchmark"]
        if b not in layouts:
            layouts[b], sc = choose_layout(get_benchmark(b)[0], backend, seed_transpiler)
            print(f"  layout {b}: qubits {layouts[b]}, estimated success probability {sc:.3f}")
        it["isa"] = transpile(it["circuit"], backend, optimization_level=1,
                              seed_transpiler=seed_transpiler, initial_layout=layouts[b])
    return layouts


def submit(items, backend, shots):
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    job = sampler.run([it["isa"] for it in items], shots=shots)
    res = job.result()
    out = []
    for it, pub in zip(items, res):
        data = pub.data; arr = None
        for name in getattr(data, "__dict__", {}) or {}:
            cand = getattr(data, name)
            if hasattr(cand, "get_counts"):
                arr = cand; break
        out.append({k.replace(" ", ""): int(v) for k, v in arr.get_counts().items()})
    jid = job.job_id() if callable(getattr(job, "job_id", None)) else str(getattr(job, "job_id", "local"))
    return out, jid


def backend_snapshot(backend):
    snap = {"name": getattr(backend, "name", str(backend)), "num_qubits": getattr(backend, "num_qubits", None)}
    try:
        t = backend.target
        errs = {}
        for op in ("cz", "ecr", "cx", "sx", "x", "measure"):
            if op in t.operation_names:
                vals = [p.error for p in t[op].values() if p is not None and p.error is not None]
                if vals:
                    errs[op] = {"median": sorted(vals)[len(vals) // 2], "n": len(vals)}
        snap["error_rates"] = errs
    except Exception as e:
        snap["error"] = repr(e)
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", choices=["plan", "h1", "h2", "h3"])
    ap.add_argument("--backend", default=None); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shots", type=int, default=4096); ap.add_argument("--bench", nargs="*", default=DEFAULT_BENCH)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--allowance", type=float, default=ALLOWANCE, help="QPU seconds of the account used for H1 to H3")
    ap.add_argument("--account", default=None, help="name of a saved IBM Quantum account (save_account(name=...))")
    a = ap.parse_args()
    if a.account:
        os.environ["NAT_IBM_ACCOUNT"] = a.account
    builders = {"h1": lambda: build_h1(a.bench), "h2": lambda: build_h2(a.bench[:2]), "h3": lambda: build_h3(a.bench[:2])}
    if a.campaign == "plan":
        from run_mini import sec_per_circuit
        rate, src = sec_per_circuit(a.dry_run, a.shots)
        n = {k: len(f()) for k, f in builders.items()}
        for k in n:
            print(f"{k}: {n[k]} circuits, about {n[k] * rate:.0f} s" + ("  per session" if k == "h2" else ""))
        tot = (n["h1"] + n["h3"] + 8 * n["h2"]) * rate
        print(f"h1 + h3 + 8 sessions of h2: about {tot:.0f} s ({src}); used on this account: {used_seconds(a.dry_run):.0f} s of {a.allowance:.0f} s")
        return
    items = builders[a.campaign]()
    from run_mini import sec_per_circuit, job_seconds
    import numpy as np
    rate, src = sec_per_circuit(a.dry_run, a.shots)
    est = len(items) * rate; remaining = a.allowance - RESERVE - used_seconds(a.dry_run)
    print(f"{a.campaign}: {len(items)} circuits, estimated {est:.0f} s ({src}); remaining on this account before reserve {remaining:.0f} s")
    if est > remaining:
        raise SystemExit("not enough allowance left on this account for this campaign")
    backend = get_backend(a.backend, a.dry_run)
    if not a.dry_run and not a.yes:
        if input(f"submit {len(items)} circuits to {backend.name}? [y/N] ").strip().lower() != "y":
            return
    layouts = transpile_batch(items, backend, saved_layouts(backend.name, a.dry_run))
    order = np.random.default_rng(7 + len(items)).permutation(len(items))
    batch = [items[i] for i in order]
    t0 = time.time(); counts_b, jid = submit(batch, backend, a.shots)
    counts = [None] * len(items)
    for pos, i in enumerate(order):
        counts[i] = counts_b[pos]; items[i]["position_in_job"] = int(pos)
    qs = job_seconds(jid, a.dry_run)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "dryrun_" + backend.name if a.dry_run else backend.name
    d = os.path.join(ROOT, "results", "hardware_raw", a.campaign, f"{stamp}_{tag}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "circuits.qpy"), "wb") as f:
        qpy.dump([it["circuit"] for it in items], f)
    with open(os.path.join(d, "isa_circuits.qpy"), "wb") as f:
        qpy.dump([it["isa"] for it in items], f)
    meta = [{k: v for k, v in it.items() if k not in ("circuit", "isa")} | {"counts": c,
             "isa_depth": it["isa"].depth(), "isa_two_qubit_gates": sum(v for k, v in it["isa"].count_ops().items() if k in ("cz", "ecr", "cx"))}
            for it, c in zip(items, counts)]
    json.dump({"campaign": a.campaign, "backend": backend_snapshot(backend), "dry_run": a.dry_run,
               "job_id": jid, "shots": a.shots, "submitted": stamp, "wall_seconds": round(time.time() - t0, 1),
               "estimated_seconds": round(est, 1), "quantum_seconds": qs,
               "initial_layouts": {k: list(map(int, v)) for k, v in layouts.items()}, "items": meta},
              open(os.path.join(d, "raw.json"), "w"), indent=1)
    print(f"saved {len(items)} results -> {d}  (job {jid}; QPU seconds {qs if qs is not None else 'n/a'}; "
          f"used on this account so far about {used_seconds(a.dry_run):.0f} s)")


if __name__ == "__main__":
    main()
