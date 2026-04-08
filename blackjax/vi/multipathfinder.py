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

from blackjax.base import VIAlgorithm
from blackjax.diagnostics import psis_weights as _psis_weights
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.vi.pathfinder import PathfinderInfo, PathfinderState, approximate, sample

__all__ = [
    "MultipathfinderState",
    "multi_approximate",
    "psis_weights",
    "as_top_level_api",
]


class MultipathfinderState(NamedTuple):
    """State returned by multi-path Pathfinder.

    path_states
        One ``PathfinderState`` per independent L-BFGS run.
    samples
        Approximate posterior samples drawn from each path's best
        approximation, shape ``(n_paths, num_samples, ...)``.
    logp
        Log target density evaluated at the per-path samples,
        shape ``(n_paths, num_samples)``.
    logq
        Log approximation density at the per-path samples,
        shape ``(n_paths, num_samples)``.
    """

    path_states: PathfinderState
    samples: ArrayTree
    logp: Array
    logq: Array


def multi_approximate(
    rng_key: PRNGKey,
    logdensity_fn: Callable,
    initial_positions: ArrayLikeTree,
    num_samples: int = 200,
    *,
    maxiter: int = 30,
    maxcor: int = 10,
    maxls: int = 1000,
    gtol: float = 1e-08,
    ftol: float = 1e-05,
) -> tuple[MultipathfinderState, PathfinderInfo]:
    """Multi-path Pathfinder variational inference.

    Runs single-path Pathfinder independently from each of the supplied
    initial positions (Algorithm 2 in :cite:p:`zhang2022pathfinder`), then
    collects the per-path samples and log densities needed for importance
    weighting via :func:`psis_weights`.

    Parameters
    ----------
    rng_key
        PRNG key.
    logdensity_fn
        (Un-normalised) log density of the target distribution.
    initial_positions
        Starting points for each L-BFGS run.  Must be a pytree where the
        leading axis indexes the ``n_paths`` paths; e.g. an array of shape
        ``(n_paths, d)``.
    num_samples
        Number of samples drawn per path to estimate ELBO and log weights.
    maxiter
        Maximum L-BFGS iterations per path.
    maxcor
        L-BFGS history size.
    maxls
        Maximum line-search steps per iteration.
    gtol
        Gradient norm convergence tolerance.
    ftol
        Function value convergence tolerance.

    Returns
    -------
    A ``MultipathfinderState`` (all path states + per-path samples and log
    densities) and a ``PathfinderInfo`` wrapping all per-path
    ``PathfinderState``s.
    """
    n_paths = jax.tree.leaves(initial_positions)[0].shape[0]
    approx_key, sample_key = jax.random.split(rng_key)
    rng_keys = jax.random.split(approx_key, n_paths)

    path_states, path_infos = jax.vmap(
        lambda key, x0: approximate(
            key,
            logdensity_fn,
            x0,
            num_samples,
            maxiter=maxiter,
            maxcor=maxcor,
            maxls=maxls,
            gtol=gtol,
            ftol=ftol,
        )
    )(rng_keys, initial_positions)

    def draw_and_eval(key, state):
        """Draw samples from one path and evaluate the target log-density."""
        path_samples, logq = sample(key, state, num_samples)
        logp = jax.vmap(logdensity_fn)(path_samples)
        return path_samples, logp, logq

    sample_keys = jax.random.split(sample_key, n_paths)
    samples, logp, logq = jax.vmap(draw_and_eval)(sample_keys, path_states)

    state = MultipathfinderState(
        path_states=path_states, samples=samples, logp=logp, logq=logq
    )
    return state, PathfinderInfo(path=path_states)


def psis_weights(state: MultipathfinderState) -> tuple[Array, Array]:
    """Compute Pareto-Smoothed Importance Sampling (PSIS) weights.

    Thin wrapper around :func:`blackjax.util.psis_weights` that extracts the
    log importance ratios from a :class:`MultipathfinderState`.

    Parameters
    ----------
    state
        Output of :func:`multi_approximate`.

    Returns
    -------
    log_weights
        Normalised log importance weights, shape ``(n_paths * num_samples,)``.
    pareto_k
        Pareto shape parameter estimate (scalar ``Array``).  Values below 0.5
        indicate reliable importance sampling; above 0.7 may indicate
        unreliable estimates.
    """
    log_ratios = (state.logp - state.logq).ravel()
    return _psis_weights(log_ratios)


def as_top_level_api(logdensity_fn: Callable) -> VIAlgorithm:
    """High-level multi-path Pathfinder interface.

    Returns a ``VIAlgorithm`` whose ``init`` runs multi-path Pathfinder and
    whose ``sample`` draws importance-resampled approximate posterior samples
    using PSIS weights.

    Parameters
    ----------
    logdensity_fn
        (Un-normalised) log density of the target distribution.

    Returns
    -------
    A ``VIAlgorithm``.
    """

    def init_fn(
        rng_key: PRNGKey,
        initial_positions: ArrayLikeTree,
        num_samples: int = 200,
        **lbfgs_parameters,
    ):
        return multi_approximate(
            rng_key,
            logdensity_fn,
            initial_positions,
            num_samples,
            **lbfgs_parameters,
        )

    def step_fn(rng_key: PRNGKey, state: MultipathfinderState):
        """Multi-path Pathfinder is one-shot; this is a no-op."""
        return state, None

    def sample_fn(
        rng_key: PRNGKey,
        state: MultipathfinderState,
        num_samples: int,
    ) -> ArrayTree:
        """Draw samples via PSIS importance resampling.

        Parameters
        ----------
        rng_key
            PRNG key.
        state
            ``MultipathfinderState`` from ``init``.
        num_samples
            Number of output samples.

        Returns
        -------
        Samples drawn from the stored per-path pool, resampled according to
        PSIS weights.  Shape: ``(num_samples, ...)``.
        """
        log_w, _ = psis_weights(state)

        # Flatten the (n_paths, num_samples_per_path, ...) pool to
        # (n_paths * num_samples_per_path, ...) for resampling.
        flat_pool = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), state.samples)

        indices = jax.random.choice(
            rng_key,
            log_w.shape[0],
            shape=(num_samples,),
            replace=True,
            p=jnp.exp(log_w),
        )
        return jax.tree.map(lambda x: x[indices], flat_pool)

    return VIAlgorithm(init_fn, step_fn, sample_fn)
