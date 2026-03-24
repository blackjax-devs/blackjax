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
"""Univariate step distributions for (elliptical) Hit-&-Run algorithms."""

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


class sym_dist:
    def __init__(self, dist):
        self._dist = dist

    def pdf(self, x):
        return 0.5 * jnp.maximum(self._dist.pdf(x), self._dist.pdf(-x))

    def logpdf(self, x):
        return 0.5 * jnp.maximum(self._dist.logpdf(x), self._dist.logpdf(-x))

    def cdf(self, x):
        return 0.5 * (self._dist.cdf(x) - self._dist.cdf(-x) + 1)

    def ppf(self, y):
        ppf_pos = self._dist.ppf(2 * y - 1)
        ppf_neg = -self._dist.ppf(2 * (1 - y) - 1)
        return jnp.where(y > 0.5, ppf_pos, ppf_neg)


class posnorm:
    def __init__(
        self,
        loc,
        scale,
    ):
        self._dist = tfd.TruncatedNormal(
            loc=loc,
            scale=scale,
            low=0,
            high=jnp.inf,
        )

    def logpdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.log_prob(
                x,
            ),
            -jnp.inf,
        )

    def pdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.prob(
                x,
            ),
            0,
        )

    def cdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.cdf(
                x,
            ),
            0.0,
        )

    def ppf(
        self,
        x,
    ):
        return self._dist.quantile(
            x,
        )


class normchi:
    def __init__(
        self,
        df,
    ):
        _chi = tfd.Chi(df=df)
        # loc, scale = jnp.sqrt(df-1), _chi.stddev()
        loc, scale = _chi.mean(), _chi.stddev()
        self._dist = posnorm(
            loc=loc,
            scale=scale,
        )

    def logpdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.logpdf(
                x,
            ),
            -jnp.inf,
        )

    def pdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.pdf(
                x,
            ),
            0,
        )

    def cdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.cdf(
                x,
            ),
            0.0,
        )

    def ppf(
        self,
        x,
    ):
        return self._dist.ppf(
            x,
        )


class chi:
    def __init__(
        self,
        df,
    ):
        self._dist = tfd.Chi(
            df=df,
        )

    def logpdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.log_prob(
                x,
            ),
            -jnp.inf,
        )

    def pdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.prob(
                x,
            ),
            0,
        )

    def cdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.cdf(
                x,
            ),
            0.0,
        )

    def ppf(
        self,
        x,
    ):
        return self._dist.quantile(
            x,
        )

    def sample(
        self,
        key,
    ):
        return self._dist.sample(seed=key)


class lognorm:
    def __init__(
        self,
    ):
        self._dist = tfd.LogNormal(
            loc=0.0,
            scale=1.0,
        )

    def logpdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.log_prob(
                x,
            ),
            -jnp.inf,
        )

    def pdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.prob(
                x,
            ),
            0,
        )

    def cdf(
        self,
        x,
    ):
        return jnp.where(
            x >= 0.0,
            self._dist.cdf(
                x,
            ),
            0.0,
        )

    def ppf(
        self,
        x,
    ):
        return self._dist.quantile(
            x,
        )

    def sample(
        self,
        key,
    ):
        return self._dist.sample(seed=key)
