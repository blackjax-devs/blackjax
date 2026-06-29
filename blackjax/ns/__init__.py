# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Nested sampling algorithms.

Nested sampling is a Monte Carlo method for Bayesian computation, used for
evidence (marginal likelihood) estimation and posterior sampling.

Available modules:

- `base`: Core components for Nested Sampling.
- `adaptive`: Adaptive nested sampling combining SMC-style adaptive tempering
  with per-step inner-kernel parameter tuning and evidence tracking.
- `nss`: Nested slice sampling, with hit-and-run (``build_kernel``) or
  slice-within-Gibbs (``build_swig_kernel``) inner kernels.
- `integrator`: NSIntegrator for tracking evidence integration.
- `utils`: Utility functions for processing nested sampling results.
- `from_mcmc`: Utilities to build nested sampling algorithms from MCMC kernels.
"""
from . import adaptive, base, from_mcmc, integrator, nss, utils

__all__ = [
    "base",
    "adaptive",
    "integrator",
    "nss",
    "utils",
    "from_mcmc",
]
