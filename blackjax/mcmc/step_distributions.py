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
"""Univariate step distribution for (elliptical) Hit-&-Run algorithms."""

from typing import Callable, NamedTuple

import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax.lax import lgamma


class StepDistribution(NamedTuple):
    logpdf: Callable
    cdf: Callable
    ppf: Callable


def normchi(df) -> StepDistribution:
    loc = jnp.exp(0.5 * jnp.log(2.0) + lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0))
    scale = jnp.sqrt(df - loc**2)

    def logpdf(x):
        log_mass = jnp.log(1.0 - jstats.norm.cdf(0.0, loc=loc, scale=scale))
        logp = jstats.norm.logpdf(x, loc=loc, scale=scale) - log_mass
        return jnp.where(x >= 0.0, logp, -jnp.inf)

    def cdf(x):
        cdf_0 = jstats.norm.cdf(0.0, loc=loc, scale=scale)
        mass = 1.0 - cdf_0
        cdf_val = (jstats.norm.cdf(x, loc=loc, scale=scale) - cdf_0) / mass
        return jnp.where(x >= 0.0, jnp.clip(cdf_val, 0.0, 1.0), 0.0)

    def ppf(p):
        cdf_0 = jstats.norm.cdf(0.0, loc=loc, scale=scale)
        p_scaled = cdf_0 + p * (1.0 - cdf_0)
        return jstats.norm.ppf(p_scaled, loc=loc, scale=scale)

    return StepDistribution(logpdf, cdf, ppf)
