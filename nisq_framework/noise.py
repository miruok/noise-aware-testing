# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Synthetic NISQ noise model: depolarising gate errors and symmetric readout
error parameterised by one error level eta (single-qubit eta, two-qubit
min(10 eta, 0.5), readout min(2 eta, 0.5))."""
from __future__ import annotations

NOISE_LEVELS = {"very_low": 0.0005, "low": 0.001, "medium": 0.005, "high": 0.02, "very_high": 0.05}


def build_noise_model(level: float):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(level, 1),
                                   ["u1", "u2", "u3", "h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(min(level * 10, 0.5), 2), ["cx", "cz"])
    p = max(0.0, min(level * 2, 0.5))
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def make_noisy_backend(level: float):
    from qiskit_aer import AerSimulator
    return AerSimulator(noise_model=build_noise_model(level))
