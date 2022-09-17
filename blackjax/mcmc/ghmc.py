"""Public API for the Generalized (Non-reversible w/ persistent momentum) HMC Kernel"""
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
from blackjax.types import PRNGKey, PyTree

__all__ = ["GHMCState", "init", "kernel"]


class GHMCState(NamedTuple):
    """State of the Generalized HMC algorithm.

    The Generalized HMC algorithm is persistent on its momentum, hence
    taking as input a position and momentum pair, updating and returning
    it for the next iteration. The algorithm also uses a persistent slice
    to perform a non-reversible Metropolis Hastings update, thus we also
    store the current slice variable and return its updated version after
    each iteration. To make computations more efficient, we also store
    the current potential energy as well as the current gradient of the
    potential energy.
    """

    position: PyTree
    momentum: PyTree
    potential_energy: float
    potential_energy_grad: PyTree
    slice: float


def init(
    rng_key: PRNGKey,
    position: PyTree,
    logprob_fn: Callable,
    logprob_grad_fn: Optional[Callable] = None,
):
    def potential_fn(x):
        return -logprob_fn(x)

    if logprob_grad_fn:
        potential_energy_grad = jax.tree_map(
            lambda g: -1.0 * g, logprob_grad_fn(position)
        )
        potential_energy = potential_fn(position)

    else:
        potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(
            position
        )

    p, unravel_fn = ravel_pytree(position)
    key_mometum, key_slice = jax.random.split(rng_key)
    momentum = unravel_fn(jax.random.normal(key_mometum, p.shape))
    slice = jax.random.uniform(key_slice, minval=-1.0, maxval=1.0)

    return GHMCState(position, momentum, potential_energy, potential_energy_grad, slice)


def kernel(
    noise_gn: Callable = lambda _: 0.0,
    divergence_threshold: float = 1000,
):
    """Build a Generalized HMC kernel.

    The Generalized HMC kernel performs a similar procedure to the standard HMC
    kernel with the difference of a persistent momentum variable and a non-reversible
    Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
    step. This means that; apart from momentum and slice variables that are dependent
    on the previous momentum and slice variables, and a Metropolis-Hastings step
    performed (equivalently) as slice sampling; the standard HMC's implementation can
    be re-used to perform Generalized HMC sampling.

    Parameters
    ----------
    noise_gn
        A function that takes as input the slice variable and outputs a random
        variable used as a noise correction of the persistent slice update.
        The parameter defaults to a random variable with a single atom at 0.
    divergence_threshold
        Value of the difference in energy above which we consider that the
        transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key, a Pytree that contains the current state
    of the chain, and free parameters of the sampling mechanism; and that
    returns a new state of the chain along with information about the transition.
    """

    _, kinetic_energy_fn, _ = metrics.gaussian_euclidean(jnp.ones(1))
    sample_proposal = proposal.nonreversible_slice_sampling

    def one_step(
        rng_key: PRNGKey,
        state: GHMCState,
        logprob_fn: Callable,
        step_size: PyTree,  # float,
        alpha: float,
        delta: float,
        logprob_grad_fn: Optional[Callable] = None,
    ) -> Tuple[GHMCState, hmc.HMCInfo]:
        """Generate new sample with the Generalized HMC kernel.

        Parameters
        ----------
        rng_key
            JAX's pseudo random number generating key.
        state
            Current state of the chain.
        logprob_fn
            (Unnormalized) Log density function being targeted.
        step_size
            Pytree with the same structure as the targeted position variable
            specifying the step size used for each dimension of the target.
        alpha
            Variable specifying the degree of persistent momentum, complementary
            to independent new momentum.
        delta
            Fixed (non-random) amount of translation added at each new iteration
            to the slice variable for non-reversible slice sampling.
        logprob_grad_fn
            Optional function customizing the gradients of the target log density.
        """

        def potential_fn(x):
            return -logprob_fn(x)

        symplectic_integrator = velocity_verlet(
            potential_fn, kinetic_energy_fn, logprob_grad_fn
        )
        proposal_generator = hmc.hmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            divergence_threshold=divergence_threshold,
            sample_proposal=sample_proposal,
        )

        key_momentum, key_noise = jax.random.split(rng_key)
        position, momentum, potential_energy, potential_energy_grad, slice = state
        # New momentum is persistent
        momentum = update_momentum(key_momentum, state, alpha)
        # Slice is non-reversible
        slice = ((slice + 1.0 + delta + noise_gn(key_noise)) % 2) - 1.0

        integrator_state = integrators.IntegratorState(
            position, momentum, potential_energy, potential_energy_grad
        )
        proposal, info = proposal_generator(slice, integrator_state)
        proposal = hmc.flip_momentum(proposal)
        state = GHMCState(
            proposal.position,
            proposal.momentum,
            proposal.potential_energy,
            proposal.potential_energy_grad,
            info.acceptance_probability,
        )

        return state, info

    return one_step


def velocity_verlet(
    potential_fn: Callable,
    kinetic_energy_fn: integrators.EuclideanKineticEnergy,
    logprob_grad_fn: Optional[Callable] = None,
) -> integrators.EuclideanIntegrator:
    """The velocity Verlet (or Verlet-Störmer) integrator.

    Generates an implementation similar to the one in `integrators` except
    with a step_size parameter of the same PyTree structure as the position
    (target) variable that specifies an independent step_size for each dimension
    of the target.
    """

    if logprob_grad_fn:
        potential_and_grad_fn = lambda x: (
            potential_fn(x),
            jax.tree_map(lambda g: -1.0 * g, logprob_grad_fn(x)),
        )
    else:
        potential_and_grad_fn = jax.value_and_grad(potential_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(
        state: integrators.IntegratorState, step_size: PyTree
    ) -> integrators.IntegratorState:
        position, momentum, _, potential_energy_grad = state

        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad, step_size: momentum
            - 0.5 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
            step_size,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad, step_size: position
            + step_size * kinetic_grad,
            position,
            kinetic_grad,
            step_size,
        )

        potential_energy, potential_energy_grad = potential_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, potential_grad, step_size: momentum
            - 0.5 * step_size * potential_grad,
            momentum,
            potential_energy_grad,
            step_size,
        )

        return integrators.IntegratorState(
            position,
            momentum,
            potential_energy,
            potential_energy_grad,
        )

    return one_step


def update_momentum(rng_key, state, alpha):
    """Persistent update of the momentum variable.

    Performs a persistent update of the momentum, taking as input the previous
    momentum, a random number generating key and the parameter alpha. Outputs
    an updated momentum that is a mixture of the previous momentum a new sample
    from a Gaussian density (dependent on alpha). The weights of the mixture of
    these two components are a function of alpha.
    """

    position, momentum, *_ = state

    m, _ = ravel_pytree(momentum)
    momentum_generator, *_ = metrics.gaussian_euclidean(
        1 / alpha * jnp.ones(jnp.shape(m))
    )
    momentum = jax.tree_map(
        lambda prev_momentum, shifted_momentum: prev_momentum * jnp.sqrt(1.0 - alpha)
        + shifted_momentum,
        momentum,
        momentum_generator(rng_key, position),
    )

    return momentum
