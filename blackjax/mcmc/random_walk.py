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

"""
Implements the (basic) user interfaces for Random Walk Rosenbluth-Metropolis-Hastings kernels.
Some interfaces are exposed here for convenience and for entry level users, who might be familiar
with simpler versions of the algorithms, but in all cases they are particular instantiations
of the Random Walk Rosenbluth-Metropolis-Hastings.

Let's note $x_{t-1}$ to the previous position and $x_t$ to the newly sampled one.

The variants offered are:

1. Proposal distribution as addition of random noice from previous position. This means
   $x_t = x_{t-1} + step$.

    Function: `additive_step`

2. Independent proposal distribution: $P(x_t)$ doesn't depend on $x_{t_1}$.

    Function: `irmh`

3. Proposal distribution using a symmetric function. That means $P(x_t|x_{t-1}) = P(x_{t-1}|x_t)$.
   See 'Metropolis Algorithm' in [1].

    Function: `rmh` without proposal_logdensity_fn.

4. Asymmetric proposal distribution. See 'Metropolis-Hastings' Algorithm in [1].

    Function: `rmh` with proposal_logdensity_fn.

Reference: :cite:p:`gelman2014bayesian` Section 11.2

Examples
--------
    The simplest case is:

    .. code::

        random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(sigma))
        state = random_walk.init(position)
        new_state, info = random_walk.step(rng_key, state)

    In all cases we can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(random_walk.step)
        new_state, info = step(rng_key, state)

"""
from typing import Callable, NamedTuple, Optional

import jax
from jax import numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import proposal
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise

__all__ = [
    "build_additive_step",
    "normal",
    "build_irmh",
    "build_rmh",
    "RWInfo",
    "RWState",
    "rmh_proposal",
    "build_rmh_transition_energy",
    "additive_step_random_walk",
    "irmh_as_top_level_api",
    "rmh_as_top_level_api",
    "normal_random_walk",
]


def normal(sigma: Array) -> Callable:
    """Normal Random Walk proposal.

    Propose a new position such that its distance to the current position is
    normally distributed. Suitable for continuous variables.

    Parameter
    ---------
    sigma:
        vector or matrix that contains the standard deviation of the centered
        normal distribution from which we draw the move proposals.

    """
    if jnp.ndim(sigma) > 2:
        raise ValueError("sigma must be a vector or a matrix.")

    def propose(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
        return generate_gaussian_noise(rng_key, position, sigma=sigma)

    return propose


class RWState(NamedTuple):
    """State of the RW chain.

    position
        Current position of the chain.
    log_density
        Current value of the log-density

    """

    position: ArrayTree
    logdensity: float


class RWInfo(NamedTuple):
    """Additional information on the RW chain.

    This additional information can be used for debugging or computing
    diagnostics.

    acceptance_rate
        The acceptance probability of the transition, linked to the energy
        difference between the original and the proposed states.
    is_accepted
        Whether the proposed position was accepted or the original position
        was returned.
    proposal
        The state proposed by the proposal.

    """

    acceptance_rate: float
    is_accepted: bool
    proposal: RWState


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> RWState:
    """Create a chain state from a position.

    Parameters
    ----------
    position: PyTree
        The initial position of the chain
    logdensity_fn: Callable
        Log-probability density function of the distribution we wish to sample
        from.

    """
    return RWState(position, logdensity_fn(position))


def build_additive_step():
    """Build a Random Walk Rosenbluth-Metropolis-Hastings kernel

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def kernel(
        rng_key: PRNGKey, state: RWState, logdensity_fn: Callable, random_step: Callable
    ) -> tuple[RWState, RWInfo]:
        def proposal_generator(key_proposal, position):
            move_proposal = random_step(key_proposal, position)
            new_position = jax.tree_util.tree_map(jnp.add, position, move_proposal)
            return new_position

        inner_kernel = build_rmh()
        return inner_kernel(rng_key, state, logdensity_fn, proposal_generator)

    return kernel


def normal_random_walk(logdensity_fn: Callable, sigma):
    """
    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    sigma
        The value of the covariance matrix of the gaussian proposal distribution.

    Returns
    -------
         A ``SamplingAlgorithm``.
    """
    return additive_step_random_walk(logdensity_fn, normal(sigma))


def additive_step_random_walk(
    logdensity_fn: Callable, random_step: Callable
) -> SamplingAlgorithm:
    """Implements the user interface for the Additive Step RMH

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rw = blackjax.additive_step_random_walk(logdensity_fn, random_step)
        state = rw.init(position)
        new_state, info = rw.step(rng_key, state)

    The specific case of a Gaussian `random_step` is already implemented, either with independent components
    when `covariance_matrix` is a one dimensional array or with dependent components if a two dimensional array:

    .. code::

        rw_gaussian = blackjax.additive_step_random_walk.normal_random_walk(logdensity_fn, covariance_matrix)
        state = rw_gaussian.init(position)
        new_state, info = rw_gaussian.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    random_step
        A Callable that takes a random number generator and the current state and produces a step,
        which will be added to the current position to obtain a new position. Must be symmetric
        to maintain detailed balance. This means that P(step|position) = P(-step | position+step)

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_additive_step()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, random_step)

    return SamplingAlgorithm(init_fn, step_fn)


def build_irmh() -> Callable:
    """
    Build an Independent Random Walk Rosenbluth-Metropolis-Hastings kernel. This implies
    that the proposal distribution does not depend on the particle being mutated :cite:p:`wang2022exact`.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: RWState,
        logdensity_fn: Callable,
        proposal_distribution: Callable,
        proposal_logdensity_fn: Optional[Callable] = None,
    ) -> tuple[RWState, RWInfo]:
        """
        Parameters
        ----------
        proposal_distribution
            A function that, given a PRNGKey, is able to produce a sample in the same
            domain of the target distribution.
        proposal_logdensity_fn:
            For non-symmetric proposals, a function that returns the log-density
            to obtain a given proposal knowing the current state. If it is not
            provided we assume the proposal is symmetric.
        """

        def proposal_generator(rng_key: PRNGKey, position: ArrayTree):
            del position
            return proposal_distribution(rng_key)

        inner_kernel = build_rmh()
        return inner_kernel(
            rng_key, state, logdensity_fn, proposal_generator, proposal_logdensity_fn
        )

    return kernel


def irmh_as_top_level_api(
    logdensity_fn: Callable,
    proposal_distribution: Callable,
    proposal_logdensity_fn: Optional[Callable] = None,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the independent RMH.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.irmh(logdensity_fn, proposal_distribution)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    proposal_distribution
        A Callable that takes a random number generator and produces a new proposal. The
        proposal is independent of the sampler's current state.
    proposal_logdensity_fn:
        For non-symmetric proposals, a function that returns the log-density
        to obtain a given proposal knowing the current state. If it is not
        provided we assume the proposal is symmetric.
    Returns
    -------
    A ``SamplingAlgorithm``.

    """
    kernel = build_irmh()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            proposal_distribution,
            proposal_logdensity_fn,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def build_rmh():
    """Build a Rosenbluth-Metropolis-Hastings kernel.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: RWState,
        logdensity_fn: Callable,
        transition_generator: Callable,
        proposal_logdensity_fn: Optional[Callable] = None,
    ) -> tuple[RWState, RWInfo]:
        """Move the chain by one step using the Rosenbluth Metropolis Hastings
        algorithm.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random
           numbers.
        logdensity_fn:
            A function that returns the log-probability at a given position.
        transition_generator:
            A function that generates a candidate transition for the markov chain.
        proposal_logdensity_fn:
            For non-symmetric proposals, a function that returns the log-density
            to obtain a given proposal knowing the current state. If it is not
            provided we assume the proposal is symmetric.
        state:
            The current state of the chain.

        Returns
        -------
        The next state of the chain and additional information about the current
        step.

        """
        transition_energy = build_rmh_transition_energy(proposal_logdensity_fn)

        compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
            transition_energy
        )

        proposal_generator = rmh_proposal(
            logdensity_fn, transition_generator, compute_acceptance_ratio
        )
        new_state, do_accept, p_accept = proposal_generator(rng_key, state)
        return new_state, RWInfo(p_accept, do_accept, new_state)

    return kernel


def rmh_as_top_level_api(
    logdensity_fn: Callable,
    proposal_generator: Callable[[PRNGKey, ArrayLikeTree], ArrayTree],
    proposal_logdensity_fn: Optional[Callable[[ArrayLikeTree], ArrayTree]] = None,
) -> SamplingAlgorithm:
    """Implements the user interface for the RMH.

    Examples
    --------

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.rmh(logdensity_fn, proposal_generator)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    proposal_generator
        A Callable that takes a random number generator and the current state and produces a new proposal.
    proposal_logdensity_fn
        The logdensity function associated to the proposal_generator. If the generator is non-symmetric,
         P(x_t|x_t-1) is not equal to P(x_t-1|x_t), then this parameter must be not None in order to apply
         the Metropolis-Hastings correction for detailed balance.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_rmh()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            proposal_generator,
            proposal_logdensity_fn,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def build_rmh_transition_energy(proposal_logdensity_fn: Optional[Callable]) -> Callable:
    if proposal_logdensity_fn is None:

        def transition_energy(prev_state, new_state):
            return -new_state.logdensity

    else:

        def transition_energy(prev_state, new_state):
            return -new_state.logdensity - proposal_logdensity_fn(new_state, prev_state)

    return transition_energy


def rmh_proposal(
    logdensity_fn: Callable,
    transition_distribution: Callable,
    compute_acceptance_ratio: Callable,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    def generate(rng_key, previous_state: RWState) -> tuple[RWState, bool, float]:
        key_proposal, key_accept = jax.random.split(rng_key, 2)
        position, _ = previous_state
        new_position = transition_distribution(key_proposal, position)
        proposed_state = RWState(new_position, logdensity_fn(new_position))
        log_p_accept = compute_acceptance_ratio(previous_state, proposed_state)
        accepted_state, info = sample_proposal(
            key_accept, log_p_accept, previous_state, proposed_state
        )
        do_accept, p_accept, _ = info
        return accepted_state, do_accept, p_accept

    return generate
