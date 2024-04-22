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

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.random
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves
from jax.typing import ArrayLike

from blackjax.base import VIAlgorithm
from blackjax.types import ArrayLikeTree, PRNGKey

__all__ = ["SchrodingerFollmerState", "sample", "init", "step"]


class SchrodingerFollmerState(NamedTuple):
    """State of the Schrödinger-Föllmer algorithm.

    The Schrödinger-Föllmer algorithm gets samples from the target distribution by
    approximating the target distribution as the terminal value of a stochastic differential
    equation (SDE) with a drift term that is evaluated under the running samples.

    position:
        position of the sample
    time:
        Current integration time of the SDE
    """

    position: ArrayLikeTree
    time: ArrayLike


class SchrodingerFollmerInfo(NamedTuple):
    """Extra information returned by the Schrodinger Follmer algorithm.

    drift:
        Approximation of the drift term of the SDE
    """

    drift: ArrayLikeTree


def init(example_position: ArrayLikeTree) -> SchrodingerFollmerState:
    zero = jax.tree.map(jnp.zeros_like, example_position)
    return SchrodingerFollmerState(zero, 0.0)


def step(
    rng_key: PRNGKey,
    state: SchrodingerFollmerState,
    logdensity_fn: Callable,
    step_size: float,
    n_samples: int,
) -> Tuple[SchrodingerFollmerState, SchrodingerFollmerInfo]:
    """
    Runs one step of the Schrödinger-Föllmer algorithm. As per the paper, we only allow for Euler-Maruyama integration.
    It is likely possible to generalize this to other integration schemes but is not considered in the original work
    and we therefore do not consider it here.

    Note that we use the version with Stein's lemma as computing the gradient of the *density* is typically unstable.

    Parameters
    ----------
    rng_key
        PRNG key
    state
        Current state of the algorithm
    logdensity_fn
        Log-density of the target distribution
    step_size
        Step size of the integration scheme
    n_samples
        Number of samples to use to approximate the drift term
    """

    drift_key, sde_key = jax.random.split(rng_key)

    ravelled_position, unravel_fn = ravel_pytree(state.position)
    scale = jnp.sqrt(1 - state.time)

    eps_drift = jax.random.normal(drift_key, (n_samples,) + ravelled_position.shape)
    eps_drift = jax.vmap(unravel_fn)(eps_drift)

    perturbed_position = jax.tree.map(
        lambda a, b: a[None, ...] + scale * b, state.position, eps_drift
    )

    log_pdf = jax.vmap(_log_fn_corrected, in_axes=[0, None])(
        perturbed_position, logdensity_fn
    )
    log_pdf -= jnp.max(log_pdf, axis=0, keepdims=True)
    pdf = jnp.exp(log_pdf)

    num = jax.tree.map(lambda a: pdf @ a, eps_drift)
    den = scale * jnp.sum(pdf, axis=0)

    drift = jax.tree.map(lambda a: a / den, num)

    eps_sde = jax.random.normal(sde_key, ravelled_position.shape)
    eps_sde = unravel_fn(eps_sde)
    next_position = jax.tree.map(
        lambda a, b, c: a + step_size * b + step_size**0.5 * c,
        state.position,
        drift,
        eps_sde,
    )
    next_state = SchrodingerFollmerState(next_position, state.time + step_size)
    return next_state, SchrodingerFollmerInfo(drift)


def sample(
    rng_key: PRNGKey,
    initial_state: SchrodingerFollmerState,
    log_density_fn: Callable,
    n_steps: int,
    n_inner_samples,
    n_samples: int = 1,
):
    """
    Samples from the target distribution using the Schrödinger-Föllmer algorithm.

    Parameters
    ----------
    rng_key
        PRNG key
    initial_state
        Current state of the algorithm
    log_density_fn
        Log-density of the target distribution
    n_steps
        Number of steps to run the algorithm for
    n_inner_samples
        Number of samples to use to approximate the drift term
    n_samples
        Number of samples to draw
    """
    dt = 1.0 / n_steps

    initial_position = initial_state.position
    initial_positions = jax.tree.map(
        lambda a: jnp.zeros([n_samples, *a.shape], dtype=a.dtype), initial_position
    )
    initial_states = SchrodingerFollmerState(initial_positions, jnp.zeros((n_samples,)))

    def body(i, states):
        subkey = jax.random.fold_in(rng_key, i)
        keys = jax.random.split(subkey, n_samples)
        next_states, _ = jax.vmap(step, [0, 0, None, None, None])(
            keys, states, log_density_fn, dt, n_inner_samples
        )
        return next_states

    final_states = jax.lax.fori_loop(0, n_steps, body, initial_states)

    return final_states


def _log_fn_corrected(position, logdensity_fn):
    """
    The Schrödinger-Föllmer algorithm requires the log-density to be given with respect to a standard Gaussian base measure
    but the log-density function passed to the algorithm in BlackJAX is typically given with respect to the Borel measure.
    This corrects the gradient of the log-density function to account for this.
    """
    log_pdf_val = logdensity_fn(position)
    norm = jax.tree.map(lambda a: 0.5 * jnp.sum(a**2), position)
    norm = sum(tree_leaves(norm))
    return log_pdf_val + norm


def as_top_level_api(logdensity_fn: Callable, n_steps: int, n_inner_samples: int) -> VIAlgorithm:  # type: ignore[misc]
    """Implements the (basic) user interface for the Schrödinger-Föllmer algortithm :cite:p:`huang2021schrodingerfollmer`.

    The Schrödinger-Föllmer algorithm obtains (approximate) samples from the target distribution by means of a diffusion with
    approximated drifts.

    Parameters
    ----------
    logdensity_fn
        A function that represents the log-density of the model we want
        to sample from.
    n_steps
        Number of steps used in the SDE
    n_inner_samples
        Number of samples used to approximate the drift term
    Returns
    -------
    A ``VIAlgorithm``.

    """

    def init_fn(position: ArrayLikeTree):
        return init(position)

    def step_fn(
        rng_key: PRNGKey, state: SchrodingerFollmerState
    ) -> tuple[SchrodingerFollmerState, SchrodingerFollmerInfo]:
        return step(rng_key, state, logdensity_fn, 1 / n_steps, n_inner_samples)

    def sample_fn(rng_key: PRNGKey, state: SchrodingerFollmerState, n_samples: int):
        return sample(
            rng_key, state, logdensity_fn, n_steps, n_inner_samples, n_samples
        )

    return VIAlgorithm(init_fn, step_fn, sample_fn)
