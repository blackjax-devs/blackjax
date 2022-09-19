from typing import Callable

import jax
import jax.numpy as jnp

from blackjax.types import PyTree


def grad_estimator(
    logprior_fn: Callable, loglikelihood_fn: Callable, data_size: int
) -> Callable:
    """Builds a simple gradient estimator.

    This estimator first appeared in [1]_. The `logprior_fn` function has a
    single argument:  the current position (value of parameters). The
    `loglikelihood_fn` takes two arguments: the current position and a batch of
    data; if there are several variables (as, for instance, in a supervised
    learning contexts), they are passed in a tuple.

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

    """

    def logposterior_estimator_fn(position: PyTree, data_batch: PyTree) -> float:
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
            batch_loglikelihood(position, data_batch), axis=0
        )

    logprob_grad = jax.grad(logposterior_estimator_fn)

    return logprob_grad


def cv_grad_estimator(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    data: PyTree,
    centering_position: PyTree,
) -> Callable:
    """Builds a control variate gradient estimator [1]_.

    Parameters
    ----------
    logprior_fn
        The log-probability density function corresponding to the prior
        distribution.
    loglikelihood_fn
        The log-probability density function corresponding to the likelihood.
    data
        The full dataset.
    centering_position
        Centering position for the control variates (typically the MAP).

    References
    ----------
    .. [1]: Baker, J., Fearnhead, P., Fox, E. B., & Nemeth, C. (2019).
            Control variates for stochastic gradient MCMC. Statistics
            and Computing, 29(3), 599-615.

    """
    data_size = jax.tree_leaves(data)[0].shape[0]
    logposterior_grad_estimator_fn = grad_estimator(
        logprior_fn, loglikelihood_fn, data_size
    )

    # Control Variates use the gradient on the full dataset
    logposterior_grad_center = logposterior_grad_estimator_fn(centering_position, data)

    def logposterior_estimator_fn(position: PyTree, data_batch: PyTree) -> float:
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
        logposterior_grad_estimate = logposterior_grad_estimator_fn(
            position, data_batch
        )
        logposterior_grad_center_estimate = logposterior_grad_estimator_fn(
            centering_position, data_batch
        )

        def control_variate(grad_estimate, center_grad_estimate, center_grad):
            return grad_estimate + center_grad - center_grad_estimate

        return jax.tree_util.tree_map(
            control_variate,
            logposterior_grad_estimate,
            logposterior_grad_center,
            logposterior_grad_center_estimate,
        )

    return logposterior_estimator_fn
