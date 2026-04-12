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
"""Skeleton for a new Variational Inference algorithm.

Copy this file to ``blackjax/vi/<your_algorithm>.py``, replace every
occurrence of ``MyVI`` / ``my_vi`` with your algorithm's name, fill
in the blanks, and delete these module-level comments.

See ``docs/developer/new_algorithm_guide.md`` for the complete walkthrough.
"""
from typing import Callable, NamedTuple

import jax
from optax import GradientTransformation

from blackjax.base import VIAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["MyVIState", "MyVIInfo", "init", "step", "sample", "as_top_level_api"]


# ---------------------------------------------------------------------------
# State and info
# ---------------------------------------------------------------------------


class MyVIState(NamedTuple):
    """State of My VI algorithm.

    params
        Variational parameters (mean, log-scale, …).
    opt_state
        Optimizer state (from optax).
    """

    params: ArrayTree
    opt_state: ArrayTree


class MyVIInfo(NamedTuple):
    """Transition information for My VI algorithm.

    elbo
        Evidence lower bound at the current step.
    """

    elbo: float


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def init(
    position: ArrayLikeTree,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
) -> MyVIState:
    """Initialize My VI state.

    Parameters
    ----------
    position
        Initial variational parameters.
    logdensity_fn
        Log-density of the target distribution.
    optimizer
        An optax gradient transformation (e.g. ``optax.adam(1e-3)``).

    Returns
    -------
    The initial ``MyVIState``.
    """
    params = position  # or transform position into variational params
    opt_state = optimizer.init(params)
    return MyVIState(params, opt_state)


def step(
    rng_key: PRNGKey,
    state: MyVIState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 100,
) -> tuple[MyVIState, MyVIInfo]:
    """Advance the VI approximation by one gradient step.

    Parameters
    ----------
    rng_key
        PRNG key for MC sampling inside the ELBO estimator.
    state
        Current VI state.
    logdensity_fn
        Log-density of the target distribution.
    optimizer
        The same optax optimizer passed to ``init``.
    num_samples
        Number of MC samples used to estimate the ELBO gradient.

    Returns
    -------
    Updated ``MyVIState`` and ``MyVIInfo`` diagnostics.
    """

    def elbo_fn(params):
        # Compute the ELBO (or its negation for minimization).
        ...
        return elbo

    elbo, grads = jax.value_and_grad(elbo_fn)(state.params)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = jax.tree.map(lambda p, u: p + u, state.params, updates)
    return MyVIState(new_params, new_opt_state), MyVIInfo(elbo)


def sample(
    rng_key: PRNGKey,
    state: MyVIState,
    num_samples: int,
) -> ArrayTree:
    """Draw samples from the current variational approximation.

    Parameters
    ----------
    rng_key
        PRNG key.
    state
        Current VI state.
    num_samples
        Number of samples to draw.

    Returns
    -------
    A pytree of samples with a leading dimension of ``num_samples``.
    """
    ...


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def as_top_level_api(
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    num_samples: int = 100,
) -> VIAlgorithm:
    """My VI algorithm — user-facing convenience wrapper.

    Examples
    --------

    .. code::

        import optax
        vi = blackjax.my_vi(logdensity_fn, optax.adam(1e-3))
        state = vi.init(initial_position)
        for _ in range(1_000):
            state, info = vi.step(rng_key, state)
        samples = vi.sample(rng_key, state, num_samples=1_000)

    Parameters
    ----------
    logdensity_fn
        The log-density function of the target distribution.
    optimizer
        An optax gradient transformation.
    num_samples
        Number of MC samples per gradient step.

    Returns
    -------
    A ``VIAlgorithm``.
    """

    def init_fn(position: ArrayLikeTree) -> MyVIState:
        return init(position, logdensity_fn, optimizer)

    def step_fn(rng_key: PRNGKey, state: MyVIState) -> tuple[MyVIState, MyVIInfo]:
        return step(rng_key, state, logdensity_fn, optimizer, num_samples)

    def sample_fn(rng_key: PRNGKey, state: MyVIState, num_samples: int) -> ArrayTree:
        return sample(rng_key, state, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)
