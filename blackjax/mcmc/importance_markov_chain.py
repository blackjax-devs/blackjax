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
r"""Public API for the Importance Markov Chain.

The Importance Markov Chain (IMC) :cite:p:`andral2024importance` samples a
target :math:`\pi` from draws of an instrumental distribution
:math:`\tilde\pi`. It is necessary to know both densities up to a constant.

Each instrumental draw :math:`\tilde x` is weighted by

.. math::

    \rho(\tilde x) = \kappa \frac{\pi(\tilde x)}{\tilde\pi(\tilde x)},

where :math:`\kappa > 0` is a hyperparameter scaling the number of samples per draw. Given
:math:`\tilde x`, a number of replicas :math:`N` is drawn with
:math:`\mathbb{E}[N \mid \tilde x] = \rho(\tilde x)`. By default
:math:`N \mid \tilde x = \lfloor\rho\rfloor + B` with
:math:`B \sim \mathrm{Bernoulli}(\rho - \lfloor\rho\rfloor)`.

The IMC constructs an augmented chain :math:`((X_i, M_i))`, where :math:`M_i` is the number of
copies of :math:`X_i` still to be returned. From a state :math:`(x, m)`:

1. If :math:`m \geq 1`, move to :math:`(x, m - 1)`.
2. If :math:`m = 0`, draw :math:`\tilde x \sim \tilde\pi` and then
   :math:`N \mid \tilde x`, until :math:`N \geq 1`. Move to
   :math:`(\tilde x, N - 1)`.

The first coordinate of the chain targets :math:`\pi`.
Each instrumental draw can appear multiple times in the chain. Draws of
:math:`\tilde\pi` can be independent, or come from any Markov kernel that
leaves :math:`\tilde\pi` invariant, such as NUTS on a tempered target.

If :math:`\sup(\pi /\tilde\pi) < \infty`, with independent draws and
:math:`\kappa < 1 / \sup(\pi/\tilde\pi)`, IMC is algorithmically equivalent to
rejection sampling.

With unnormalised densities, the expected number of samples per instrumental
draw is :math:`\kappa Z_\pi / Z_{\tilde\pi}`, which is unknown in advance.
:func:`log_kappa_for_num_samples` chooses :math:`\kappa` from the log-weights
of a pilot set of instrumental draws:

.. code::

    pilot_positions = jax.vmap(proposal_distribution)(jax.random.split(pilot_key, 1_000))
    pilot_log_weights = jax.vmap(logdensity_fn)(pilot_positions) - jax.vmap(
        proposal_logdensity_fn
    )(pilot_positions)
    log_kappa = log_kappa_for_num_samples(pilot_log_weights, num_samples=1_000)

With this choice, each new instrumental draw gives one sample on average, up
to the Monte Carlo error of the pilot set.

Examples
--------

With an MCMC sampler of :math:`\tilde\pi`, each step returns one sample of
the chain:

.. code::

    def instrumental_logdensity_fn(x):
        return 0.1 * logdensity_fn(x)

    nuts = blackjax.nuts(instrumental_logdensity_fn, step_size, inverse_mass_matrix)
    imc = blackjax.imc(logdensity_fn, instrumental_logdensity_fn, nuts, log_kappa)

    state = imc.init(initial_position)
    state, info = imc.step(rng_key, state)

With independent proposals, all draws are made at once and returned with
their number of replicas:

.. code::

    positions, num_replicas = blackjax.imc.independent_sampling(
        rng_key,
        logdensity_fn,
        proposal_distribution,
        proposal_logdensity_fn,
        num_proposals=10_000,
        log_kappa=log_kappa,
    )

    mean = jnp.sum(num_replicas * positions) / jnp.sum(num_replicas)
    chain = jnp.repeat(positions, num_replicas, axis=0)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import Info, SamplingAlgorithm, State
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "IMCState",
    "IMCInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
    "optimal_replication",
    "log_kappa_for_num_samples",
    "independent_sampling",
]

# Largest number of replicas for a single draw
MAX_NUM_REPLICAS = 2.0**30


class IMCState(NamedTuple):
    """State of the Importance Markov Chain.

    position
        Current sample of the chain.
    num_remaining
        Number of copies of ``position`` that remain to be returned before a
        new instrumental draw is needed.
    instrumental_state
        State of the sampling algorithm that targets the instrumental
        distribution. Any state with a ``position`` field.
    """

    position: ArrayTree
    num_remaining: Array
    instrumental_state: Any


class IMCInfo(NamedTuple):
    """Information on one transition of the Importance Markov Chain.

    num_instrumental_steps
        Number of steps of the instrumental algorithm taken during this
        transition. It is 0 when the transition returns a copy of the current
        sample.
    is_exhausted
        Whether ``max_num_steps`` instrumental steps were taken without any
        draw receiving a replica. The chain then keeps its current sample.
    """

    num_instrumental_steps: Array
    is_exhausted: Array


def optimal_replication(rng_key: PRNGKey, log_rho: Array) -> Array:
    """Draw numbers of replicas with mean ``exp(log_rho)`` using the variance-minimizing
    kernel (:cite:p:`andral2024importance`, Lemma 1).

    Returns :math:`N = \\lfloor\\rho\\rfloor + B` with
    :math:`B \\sim \\mathrm{Bernoulli}(\\rho - \\lfloor\\rho\\rfloor)`.

    Parameters
    ----------
    rng_key
        The pseudo-random number generator key.
    log_rho
        Logarithm of the expected number of replicas, for one draw or an
        array of draws. A NaN value gives 0 replicas.

    Returns
    -------
    The numbers of replicas as int32, with the shape of ``log_rho``. They are clipped at ``MAX_NUM_REPLICAS`` (:math:`2^{30}`).
    """
    log_rho = jnp.where(jnp.isnan(log_rho), -jnp.inf, log_rho)
    rho = jnp.clip(jnp.exp(log_rho), max=MAX_NUM_REPLICAS)
    integer_part = jnp.floor(rho)
    extra_replica = jax.random.bernoulli(rng_key, rho - integer_part)
    return (integer_part + extra_replica).astype(jnp.int32)


def log_kappa_for_num_samples(log_weights: Array, num_samples: int) -> Array:
    """Compute the hyperparameter ``log_kappa`` from the log-weights
    so that the draws produce, on average, ``num_samples`` samples.

    With :math:`\\kappa = n / \\sum_i w_i`, the expected total number of
    replicas of draws with weights :math:`w_i` is :math:`n`
    (:cite:p:`andral2024importance`, Section 6.1.2).

    Parameters
    ----------
    log_weights
        Log-weights :math:`\\log\\pi(x_i) - \\log\\tilde\\pi(x_i)` of the
        draws, up to an additive constant.
    num_samples
        Expected total number of replicas.

    Returns
    -------
    The value of :math:`\\log\\kappa`.
    """
    return jnp.log(num_samples) - jax.scipy.special.logsumexp(log_weights)


def independent_sampling(
    rng_key: PRNGKey,
    logdensity_fn: Callable,
    proposal_distribution: Callable,
    proposal_logdensity_fn: Callable,
    num_proposals: int,
    log_kappa: float | Array,
) -> tuple[ArrayTree, Array]:
    """Run the Importance Markov Chain with independent proposals.

    Draws ``num_proposals`` i.i.d. proposals, weighs them and draws their
    numbers of replicas with :func:`optimal_replication`. ``log_kappa`` can be
    chosen with :func:`log_kappa_for_num_samples`.

    Estimates use the replicas as weights, for example
    ``jnp.sum(num_replicas * f(positions)) / jnp.sum(num_replicas)``. The chain
    itself is obtained by repeating each proposal, for example
    ``jnp.repeat(positions, num_replicas, axis=0)`` outside of ``jax.jit``.

    Parameters
    ----------
    rng_key
        The pseudo-random number generator key.
    logdensity_fn
        Log-density of the target distribution, up to a constant.
    proposal_distribution
        Function ``rng_key -> position`` that draws one proposal.
    proposal_logdensity_fn
        Log-density of the proposal distribution, up to a constant.
    num_proposals
        Number of proposals to draw.
    log_kappa
        Logarithm of the scaling constant :math:`\\kappa`. The expected number
        of samples per proposal is
        :math:`\\kappa\\, \\mathbb{E}_{\\tilde\\pi}[\\pi/\\tilde\\pi]`.

    Returns
    -------
    The proposals, with a leading axis of size ``num_proposals``, and their numbers of replicas.
    """
    key_proposals, key_replicas = jax.random.split(rng_key)
    proposal_keys = jax.random.split(key_proposals, num_proposals)
    positions = jax.vmap(proposal_distribution)(proposal_keys)

    target_logdensities = jax.vmap(logdensity_fn)(positions)
    proposal_logdensities = jax.vmap(proposal_logdensity_fn)(positions)
    log_weights = target_logdensities - proposal_logdensities

    num_replicas = optimal_replication(key_replicas, log_kappa + log_weights)
    return positions, num_replicas


def init(
    position: ArrayLikeTree,
    instrumental_init_fn: Callable,
    *,
    rng_key: PRNGKey | None = None,
) -> IMCState:
    """Create an Importance Markov Chain state from a position.

    The chain starts with no remaining copies, so the first step draws from
    the instrumental algorithm. To start from an instrumental state that was
    already adapted, build the state directly with
    ``IMCState(instrumental_state.position, jnp.array(0, dtype=jnp.int32),
    instrumental_state)``.

    Parameters
    ----------
    position
        The initial position of the chain.
    instrumental_init_fn
        The ``init`` function of the instrumental algorithm, with signature
        ``(position, rng_key) -> state``.
    rng_key
        Passed to ``instrumental_init_fn``, for algorithms that need one.

    Returns
    -------
    The initial state of the Importance Markov Chain.
    """
    instrumental_state = instrumental_init_fn(position, rng_key)
    num_remaining = jnp.array(0, dtype=jnp.int32)
    return IMCState(instrumental_state.position, num_remaining, instrumental_state)


def build_kernel(
    replication_fn: Callable = optimal_replication,
    max_num_steps: int = 1_000,
) -> Callable:
    """Build an Importance Markov Chain kernel.

    This is Algorithm 2 in :cite:p:`andral2024importance`.

    Parameters
    ----------
    replication_fn
        Function ``(rng_key, log_rho) -> num_replicas`` that draws an integer
        with mean ``exp(log_rho)``. Defaults to :func:`optimal_replication`.
    max_num_steps
        Maximum number of instrumental steps in one transition. If no draw
        receives a replica within this budget, the chain keeps its current
        sample and the next transition carries on from the last instrumental
        state. Reaching the budget is reported in ``IMCInfo.is_exhausted``.
        Adjust ``kappa`` or change the instrumental algorithm if it happens often.

    Returns
    -------
    A kernel ``(rng_key, state, logdensity_fn, instrumental_logdensity_fn,
    instrumental_step_fn, log_kappa) -> (IMCState, IMCInfo)``.
    """

    def kernel(
        rng_key: PRNGKey,
        state: IMCState,
        logdensity_fn: Callable,
        instrumental_logdensity_fn: Callable,
        instrumental_step_fn: Callable,
        log_kappa: float | Array,
    ) -> tuple[IMCState, IMCInfo]:
        """Advances the chain by one step.

        Parameters
        ----------
        rng_key
            The pseudo-random number generator key.
        state
            The current state of the chain.
        logdensity_fn
            Log-density of the target distribution, up to a constant.
        instrumental_logdensity_fn
            Log-density of the instrumental distribution, up to a constant.
        instrumental_step_fn
            The ``step`` function of the instrumental algorithm, with
            signature ``(rng_key, state) -> (state, info)``. Its states must
            have a ``position`` field.
        log_kappa
            Logarithm of the scaling constant :math:`\\kappa`.

        Returns
        -------
        The next state of the chain and information on the transition.
        """

        def return_next_copy():
            new_state = IMCState(
                state.position, state.num_remaining - 1, state.instrumental_state
            )
            info = IMCInfo(jnp.array(0, dtype=jnp.int32), jnp.array(False))
            return new_state, info

        def draw_new_sample():
            return _draw_until_replicated(
                rng_key,
                state,
                logdensity_fn,
                instrumental_logdensity_fn,
                instrumental_step_fn,
                log_kappa,
                replication_fn,
                max_num_steps,
            )

        return jax.lax.cond(state.num_remaining > 0, return_next_copy, draw_new_sample)

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    instrumental_logdensity_fn: Callable,
    instrumental_algorithm: SamplingAlgorithm,
    log_kappa: float | Array,
    *,
    max_num_steps: int = 1_000,
) -> SamplingAlgorithm:
    """Implements the user interface for the Importance Markov Chain.

    Examples
    --------

    Sample a multimodal target by running NUTS on a tempered version of it,
    which moves between modes more easily:

    .. code::

        def instrumental_logdensity_fn(x):
            return 0.1 * logdensity_fn(x)

        nuts = blackjax.nuts(instrumental_logdensity_fn, step_size, inverse_mass_matrix)
        imc = blackjax.imc(logdensity_fn, instrumental_logdensity_fn, nuts, log_kappa)
        state = imc.init(position)
        new_state, info = imc.step(rng_key, state)

    ``log_kappa`` can be chosen from a pilot run of ``nuts`` with
    :func:`log_kappa_for_num_samples`.

    Parameters
    ----------
    logdensity_fn
        Log-density of the target distribution, up to a constant.
    instrumental_logdensity_fn
        Log-density of the instrumental distribution, up to a constant.
    instrumental_algorithm
        A sampling algorithm that targets the instrumental distribution.
    log_kappa
        Logarithm of the scaling constant :math:`\\kappa`. The expected number
        of samples per instrumental draw is
        :math:`\\kappa\\, \\mathbb{E}_{\\tilde\\pi}[\\pi/\\tilde\\pi]`.
    max_num_steps
        Maximum number of instrumental steps in one transition, see
        :func:`build_kernel`.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel(max_num_steps=max_num_steps)

    def init_fn(position: ArrayLikeTree, rng_key: PRNGKey | None = None) -> IMCState:
        return init(position, instrumental_algorithm.init, rng_key=rng_key)

    def step_fn(rng_key: PRNGKey, state: State) -> tuple[State, Info]:
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            instrumental_logdensity_fn,
            instrumental_algorithm.step,
            log_kappa,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def _draw_until_replicated(
    rng_key: PRNGKey,
    state: IMCState,
    logdensity_fn: Callable,
    instrumental_logdensity_fn: Callable,
    instrumental_step_fn: Callable,
    log_kappa: float | Array,
    replication_fn: Callable,
    max_num_steps: int,
) -> tuple[IMCState, IMCInfo]:
    """Advance the instrumental algorithm until a draw receives a replica."""

    def needs_new_draw(loop_state):
        _, _, num_replicas, num_steps = loop_state
        return (num_replicas == 0) & (num_steps < max_num_steps)

    def one_instrumental_step(loop_state):
        key, instrumental_state, _, num_steps = loop_state
        key, key_step, key_replicas = jax.random.split(key, 3)
        instrumental_state, _ = instrumental_step_fn(key_step, instrumental_state)

        position = instrumental_state.position
        log_weight = logdensity_fn(position) - instrumental_logdensity_fn(position)
        num_replicas = replication_fn(key_replicas, log_kappa + log_weight)
        return key, instrumental_state, num_replicas, num_steps + 1

    zero = jnp.array(0, dtype=jnp.int32)
    initial_loop_state = (rng_key, state.instrumental_state, zero, zero)
    _, instrumental_state, num_replicas, num_steps = jax.lax.while_loop(
        needs_new_draw, one_instrumental_step, initial_loop_state
    )

    is_replicated = num_replicas > 0
    position = jax.tree.map(
        lambda new, current: jnp.where(is_replicated, new, current),
        instrumental_state.position,
        state.position,
    )
    num_remaining = jnp.maximum(num_replicas - 1, 0)

    new_state = IMCState(position, num_remaining, instrumental_state)
    info = IMCInfo(num_steps, jnp.logical_not(is_replicated))
    return new_state, info
