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
"""Nested Sampling Algorithms in BlackJAX.

This subpackage provides implementations of Nested Sampling algorithms.

Nested Sampling is a Monte Carlo method for Bayesian computation, primarily
used for evidence (marginal likelihood) calculation and posterior sampling.
It is particularly well-suited for problems with multi-modal posteriors or
complex likelihood landscapes.

Available modules:
------------------
- `base`: Provides core components for Nested Sampling.
- `adaptive`: Implements Adaptive Nested Sampling, combining SMC tempering
- `nss`: Implements Nested Slice Sampling, using Hit-and-Run Slice Sampling as
         the inner kernel with adaptive tuning of its proposal mechanism.
- `integrator`: Provides NSIntegrator for tracking evidence integration.
- `utils`: Contains utility functions for processing and analyzing Nested
           Sampling results.
- `from_mcmc`: Utilities to build Nested Sampling algorithms from MCMC kernels.

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
