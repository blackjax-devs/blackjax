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
from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.types import ArrayLikeTree, ArrayTree


def logdensity_estimator(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Builds a simple estimator for the log-density.

    This estimator first appeared in :cite:p:`robbins1951stochastic`. The `logprior_fn` function has a
    single argument:  the current position (value of parameters). The
    `loglikelihood_fn` takes two arguments: the current position and a batch of
    data; if there are several variables (as, for instance, in a supervised
    learning contexts), they are passed in a tuple.

    This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.

    Parameters
    ----------
    logprior_fn
        The log-probability density function corresponding to the prior
        distribution.
    loglikelihood_fn
        The log-probability density function corresponding to the likelihood.
    data_size
        The number of items in the full dataset.

    """

    def logdensity_estimator_fn(
        position: ArrayLikeTree, minibatch: ArrayLikeTree
    ) -> ArrayTree:
        """Return an approximation of the log-posterior density.

        Parameters
        ----------
        position
            The current value of the random variables.
        batch
            The current batch of data

        Returns
        -------
        An approximation of the value of the log-posterior density function for
        the current value of the random variables.

        """
        logprior = logprior_fn(position)
        batch_loglikelihood = jax.vmap(loglikelihood_fn, in_axes=(None, 0))
        return logprior + data_size * jnp.mean(
            batch_loglikelihood(position, minibatch), axis=0
        )

    return logdensity_estimator_fn


def grad_estimator(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Build a simple estimator for the gradient of the log-density."""

    logdensity_estimator_fn = logdensity_estimator(
        logprior_fn, loglikelihood_fn, data_size
    )
    return jax.grad(logdensity_estimator_fn)


def control_variates(
    logdensity_grad_estimator: Callable,
    centering_position: ArrayLikeTree,
    data: ArrayLikeTree,
) -> Callable:
    """Builds a control variate gradient estimator :cite:p:`baker2019control`.

    This algorithm was ported from :cite:p:`coullon2022sgmcmcjax`.

    Parameters
    ----------
    logdensity_grad_estimator
        A function that approximates the target's gradient function.
    data
        The full dataset.
    centering_position
        Centering position for the control variates (typically the MAP).

    """
    cv_grad_value = logdensity_grad_estimator(centering_position, data)

    def cv_grad_estimator_fn(
        position: ArrayLikeTree, minibatch: ArrayLikeTree
    ) -> ArrayTree:
        """Return an approximation of the log-posterior density.

        Parameters
        ----------
        position
            The current value of the random variables.
        batch
            The current batch of data. The first dimension is assumed to be the
            batch dimension.

        Returns
        -------
        An approximation of the value of the log-posterior density function for
        the current value of the random variables.

        """
        grad_estimate = logdensity_grad_estimator(position, minibatch)
        center_grad_estimate = logdensity_grad_estimator(centering_position, minibatch)

        return jax.tree.map(
            lambda grad_est, cv_grad_est, cv_grad: cv_grad + grad_est - cv_grad_est,
            grad_estimate,
            center_grad_estimate,
            cv_grad_value,
        )

    return cv_grad_estimator_fn
