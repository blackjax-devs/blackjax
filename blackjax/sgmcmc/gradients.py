from typing import Callable, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp

from blackjax.types import PyTree


class GradientEstimator(NamedTuple):
    init: Callable
    estimate: Callable


def grad_estimator(
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

    def init_fn(_) -> None:
        return None

    def logposterior_estimator_fn(
        position: PyTree, minibatch: PyTree
    ) -> Tuple[PyTree, None]:
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

    def grad_estimator_fn(_, position, data_batch):
        return jax.grad(logposterior_estimator_fn)(position, data_batch), None

    return GradientEstimator(init_fn, grad_estimator_fn)


class CVGradientState(NamedTuple):
    """The state of the CV gradient estimator contains the gradient of the
    Control Variate computed on the whole dataset at initialization.

    """

    control_variate_grad: PyTree


def cv_grad_estimator(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    data: PyTree,
    centering_position: PyTree,
) -> GradientEstimator:
    """Builds a control variate gradient estimator [1]_.

    This algorithm was ported from [2]_.

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
    .. [2]: Coullon, J., & Nemeth, C. (2022). SGMCMCJax: a lightweight JAX
            library for stochastic gradient Markov chain Monte Carlo algorithms.
            Journal of Open Source Software, 7(72), 4113.

    """
    data_size = jax.tree_leaves(data)[0].shape[0]
    logposterior_grad_estimator_fn = grad_estimator(
        logprior_fn, loglikelihood_fn, data_size
    ).estimate

    def init_fn(full_dataset: PyTree) -> CVGradientState:
        """Compute the control variate on the whole dataset."""
        return CVGradientState(
            logposterior_grad_estimator_fn(None, centering_position, full_dataset)[0]
        )

    def grad_estimator_fn(
        grad_estimator_state: CVGradientState, position: PyTree, minibatch: PyTree
    ) -> Tuple[PyTree, CVGradientState]:
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
            None, position, minibatch
        )[0]
        logposterior_grad_center_estimate = logposterior_grad_estimator_fn(
            None, centering_position, minibatch
        )[0]

        def control_variate(grad_estimate, center_grad_estimate, center_grad):
            return grad_estimate + center_grad - center_grad_estimate

        return (
            control_variate(
                logposterior_grad_estimate,
                grad_estimator_state.control_variate_grad,
                logposterior_grad_center_estimate,
            ),
            grad_estimator_state,
        )

    return GradientEstimator(init_fn, grad_estimator_fn)


GradientState = Union[None, CVGradientState]
