# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Noise-aware testing framework for NISQ quantum programs.

Authors: Mir Mohammad Yousuf and Shabir Ahmad Sofi, Department of Information
Technology, National Institute of Technology Srinagar, India.
"""
__author__ = "Mir Mohammad Yousuf, Shabir Ahmad Sofi"
__email__ = "yousuf_2022phaite006@nitsri.ac.in"
__version__ = "1.0.0"
from .execution import Executor
from .divergence import jsd, pooled
from .noise import NOISE_LEVELS, build_noise_model, make_noisy_backend
from .benchmarks import get as get_benchmark, BENCHMARKS, PRETTY
from .threshold import compute_threshold, chi2_threshold, spc_envelope_threshold
from .power import detection_power, min_detectable_delta, shots_needed
from .semantics import ideal_distribution, ideal_jsd, unitary_equivalent, with_input
from .mutation import (MutationEngine, FAMILIES, EQUIVALENT_FAMILIES, TAXONOMY_CLASS,
                       NOISE_AMPLIFYING, NOISE_NEUTRAL, SEM_ADD, SEM_SUB, OPERATORS)
from .faults import INJECTORS, inject_all
from .oracles import NoiseAwareOracle, SPCEnvelopeOracle, FixedThresholdOracle, QOINStyleOracle
from .attribution import fold, attribute_zne, attribute_from_counts, reexecution_triage

__version__ = "1.0.0"
