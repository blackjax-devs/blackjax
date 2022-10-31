"""Public API for the Contour Stochastic gradient Langevin Dynamics kernel.

References
----------
.. [0]: Deng, W., Lin, G., Liang, F. (2020).
        A Contour Stochastic Gradient Langevin Dynamics Algorithm
        for Simulations of Multi-modal Distributions.
        In Neural Information Processing Systems (NeurIPS 2020).

.. [1]: Deng, W., Liang, S., Hao, B., Lin, G., Liang, F. (2022)
        Interacting Contour Stochastic Gradient Langevin Dynamics
        In International Conference on Learning Representations (ICLR)
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.sgmcmc.diffusions import overdamped_langevin
from blackjax.types import Array, PRNGKey, PyTree

__all__ = ["ContourSGLDState", "init", "kernel"]


class ContourSGLDState(NamedTuple):
    r"""State of the Contour SgLD algorithm.

    Parameters
    ----------
    position
        Current position in the sample space.
    energy_pdf
        Vector with `m` non-negative values that sum to 1. The `i`-th value
        of the vector is equal to :math:`\int_{S_1} \pi(\mathrm{d}x)` where
        :math:`S_i` is the `i`-th energy partition.
    energy_idx
        Index `i` such that the current position belongs to :math:`S_i`.

    """
    position: PyTree
    energy_pdf: Array
    energy_idx: int


def init(position: PyTree, num_partitions=512):
    energy_pdf = (
        jnp.arange(num_partitions, 0, -1) / jnp.arange(num_partitions, 0, -1).sum()
    )
    return ContourSGLDState(position, energy_pdf, num_partitions - 1)


def kernel(num_partitions=512, energy_gap=10, min_energy=0) -> Callable:
    r"""

    Parameters
    ----------
    num_partitions
        The number of partitions we divide the energy landscape into.
    energy_gap
        The difference in energy :math:`\Delta u` between the successive
        partitions. Can be determined by running e.g. an optimizer to determine
        the range of energies. `num_partition` * `energy_gap` should match this
        range.
    min_energy
        A rough estimate of the minimum energy in a dataset, which should be
        strictly smaller than the exact minimum energy! e.g. if the minimum
        energy of a dataset is 3456, we can set min_energy to be any value
        smaller than 3456. Set it to 0 is acceptable, but not efficient enough.
        the closer the gap between min_energy and 3456 is, the better.
    """

    integrator = overdamped_langevin()

    def one_step(
        rng_key: PRNGKey,
        state: ContourSGLDState,
        logdensity_estimator_fn: Callable,
        minibatch: PyTree,
        step_size_diff: float,  # step size for Langevin diffusion
        step_size_stoch: float = 1e-3,  # step size for stochastic approximation
        zeta: float = 1,
        temperature: float = 1.0,
    ) -> ContourSGLDState:
        r"""Multil-modal sampling via Contour SGLD.

        We are interested in the simulations of :math:`\exp(-U(x) / T)`,
        where :math:`U` is an energy function and :math:`T` is the temperature.

        To do so we partition the energy space into :math:`m`:

        .. math::
            S_0 = {x: U(x) <= u_1}
            S_1 = {x: u_1 < U(x) <= u_2}
            S_2 = {x: u_2 < U(x) <= u_3}
            ...
            S_{m-2} = {x: u_{m-2} < U(x) <= u_{m-1}}
            S_{m-1} = {x: U(x) > u_{m-1}}

        where :math:`-\inf < u_1 < u_2 < · · · < u_{m−1} < \inf`. We assume
        :math:`u_{i+1} − u_i = \Delta u` for :math:`i = 1, \dots , m−2`.

        Parameters
        ----------
        rng_key
            State of the pseudo-random number generator.
        state
            Current state of the CSGLD sampler
        logdensity_estimator_fn
            Function that returns an estimation of the value of the density
            function at the current position.
        minibatch
            Minibatch of data.
        step_size_diff
            Step size for the dynamics integration. Also called learning rate.
        step_size_stoch
            Step size for the update of the energy estimation.
        zeta
            Hyperparameter that controls the geometric property of the flattened
            density. If `zeta=0` the function reduces to the SGLD step function.
        temperature
            Temperature parameter :math:`T`.

        References
        ----------
        .. [0]: Deng, W., Lin, G., Liang, F. (2020).
                A Contour Stochastic Gradient Langevin Dynamics Algorithm
                for Simulations of Multi-modal Distributions.
                In Neural Information Processing Systems (NeurIPS 2020).

        [1]: Deng, W., Liang, S., Hao, B., Lin, G., Liang, F. (2022)
                Interacting Contour Stochastic Gradient Langevin Dynamics
                In International Conference on Learning Representations (ICLR)
        """

        position, energy_pdf, idx = state

        # Update the position using the overdamped Langevin diffusion
        gradient_multiplier = (
            1.0
            + zeta
            * temperature
            * (jnp.log(energy_pdf[idx]) - jnp.log(energy_pdf[idx - 1]))
            / energy_gap
        )

        logprob_grad = jax.grad(logdensity_estimator_fn)(position, minibatch)
        position = integrator(
            rng_key,
            position,
            jax.tree_util.tree_map(lambda g: gradient_multiplier * g, logprob_grad),
            step_size_diff,
            temperature,
        )

        # Update the stochastic approximation to the energy histogram
        neg_logprob = -logdensity_estimator_fn(position, minibatch)
        idx = jax.lax.min(
            jax.lax.max(
                jax.lax.floor((neg_logprob - min_energy) / energy_gap + 1).astype(
                    "int32"
                ),
                1,
            ),
            num_partitions - 1,
        )

        energy_pdf_update = -energy_pdf.copy()
        energy_pdf_update = energy_pdf_update.at[idx].set(energy_pdf_update[idx] + 1)
        energy_pdf = jax.tree_util.tree_map(
            lambda e: e + step_size_stoch * energy_pdf[idx] * energy_pdf_update,
            energy_pdf,
        )

        return ContourSGLDState(position, energy_pdf, idx)

    return one_step
