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
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import PyTree


class GradientEstimator(NamedTuple):
    init: Callable
    estimate: Callable


def estimator(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> GradientEstimator:
    """Builds a simple gradient estimator.

    This estimator first appeared in [1]_. The `logprior_fn` function has a
    single argument:  the current position (value of parameters). The
    `loglikelihood_fn` takes two arguments: the current position and a batch of
    data; if there are several variables (as, for instance, in a supervised
    learning contexts), they are passed in a tuple.

    This algorithm was ported from [2]_.

    Parameters
    ----------
    logprior_fn
        The log-probability density function corresponding to the prior
        distribution.
    loglikelihood_fn
        The log-probability density function corresponding to the likelihood.
    data_size
        The number of items in the full dataset.

    References
    ----------
    .. [1]: Robbins H. and Monro S. A stochastic approximation method. Annals
            of Mathematical Statistics, 22(30):400-407, 1951.
    .. [2]: Coullon, J., & Nemeth, C. (2022). SGMCMCJax: a lightweight JAX
            library for stochastic gradient Markov chain Monte Carlo algorithms.
            Journal of Open Source Software, 7(72), 4113.

    """

    def logposterior_estimator_fn(position: PyTree, minibatch: PyTree) -> PyTree:
        """Returns an approximation of the log-posterior density.

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

    return jax.grad(logposterior_estimator_fn)


def control_variates(
    grad_estimator: Callable,
    centering_position: PyTree,
    data: PyTree,
) -> Callable:
    """Builds a control variate gradient estimator [1]_.

    This algorithm was ported from [2]_.

    Parameters
    ----------
    grad_estimator
        A function that approximates the target's gradient function.
    data
        The full dataset.
    centering_position
        Centering position for the control variates (typically the MAP).

    References
    ----------
    .. [1]: Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2019).
            Control variates for stochastic gradient MCMC. Statistics
            and Computing, 29(3), 599-615.
    .. [2]: Coullon, J., & Nemeth, C. (2022). SGMCMCJax: a lightweight JAX
            library for stochastic gradient Markov chain Monte Carlo algorithms.
            Journal of Open Source Software, 7(72), 4113.

    """
    cv_grad_value = grad_estimator(centering_position, data)

    def cv_grad_estimator_fn(position: PyTree, minibatch: PyTree) -> PyTree:
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
        grad_estimate = grad_estimator(position, minibatch)
        center_grad_estimate = grad_estimator(centering_position, minibatch)

        return jax.tree_map(
            lambda grad_est, cv_grad_est, cv_grad: cv_grad + grad_est - cv_grad_est,
            grad_estimate,
            center_grad_estimate,
            cv_grad_value,
        )

    return cv_grad_estimator_fn
