# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Shared configuration and I/O for the experiment drivers."""
from __future__ import annotations
import json, os, platform, sys, time
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from nisq_framework import get_benchmark, NOISE_LEVELS, PRETTY  # noqa: E402,F401

RESULTS = os.path.join(ROOT, "results")
SHOTS = 4096
NOISE = dict(NOISE_LEVELS)
BENCH = ["bell", "ghz_3", "qft_3", "qaoa_3", "vqe_3", "bell_8", "ghz_8", "qaoa_8", "hwe_6"]


def environment() -> Dict[str, Any]:
    import numpy, scipy, qiskit, qiskit_aer
    env = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
           "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__, "platform": platform.platform()}
    try:
        import qiskit_ibm_runtime
        env["qiskit_ibm_runtime"] = qiskit_ibm_runtime.__version__
    except Exception:
        pass
    return env


def out_dir(name: str, explicit: str | None = None) -> str:
    """Results live in results/<name>/. If that folder already holds a finished
    run (meta.json), a new run is written to results/reruns/<name>/<timestamp>/
    so published results are never overwritten. --out selects a folder."""
    d = explicit or os.path.join(RESULTS, name)
    if explicit is None and os.path.exists(os.path.join(d, "meta.json")):
        d = os.path.join(RESULTS, "reruns", name, time.strftime("%Y%m%d_%H%M%S"))
        print(f"[published results exist; writing this run to {d}]")
    os.makedirs(d, exist_ok=True)
    return d


def _default(o):
    import numpy as np
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, set): return sorted(o)
    return str(o)


def save(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def write_meta(d: str, config: Dict[str, Any], t0: float):
    save({"config": config, "environment": environment(),
          "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
          "wall_seconds": round(time.time() - t0, 1)}, os.path.join(d, "meta.json"))
