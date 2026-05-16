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
"""Public API for a pathwise-differentiable random-direction slice sampler."""

import operator
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "ReparameterizedSliceState",
    "ReparameterizedSliceInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]


class ReparameterizedSliceState(NamedTuple):
    """State of the reparameterized random-direction slice sampler.

    position
        Current position of the chain.
    logdensity
        Log density at the current position.
    """

    position: ArrayTree
    logdensity: float


class ReparameterizedSliceInfo(NamedTuple):
    """Additional information about the reparameterized slice transition.

    This additional information can be used for debugging or computing
    diagnostics.

    direction
        The random direction used to define the one-dimensional slice.
    alpha_left
        The negative slice endpoint along ``direction``.
    alpha_right
        The positive slice endpoint along ``direction``.
    log_slice
        The logarithm of the slice level sampled for this transition.
    bracket_steps
        Number of stepping-out updates used to bracket both slice endpoints.
    bisection_steps_left
        Number of bisection iterations used to locate ``alpha_left``.
    bisection_steps_right
        Number of bisection iterations used to locate ``alpha_right``.

    """

    direction: ArrayTree
    alpha_left: float
    alpha_right: float
    log_slice: float
    bracket_steps: int
    bisection_steps_left: int
    bisection_steps_right: int


class SliceRootFindingConfig(NamedTuple):
    """Numerical parameters used by the slice root finder."""

    root_tolerance: float
    near_zero: float
    bracket_log_start: float
    bracket_log_space: float
    bracket_max_steps: int
    bisection_max_steps: int


class SliceStepIntermediates(NamedTuple):
    """Auxiliary data used by the direct-forward helper.

    This container is kept only so `_pathwise_forward` can retain the same
    shape as the reuse-intermediates variant. The all-orders-oriented JVP rule
    below does not consume these intermediates; instead it calls the decorated
    primal function to compute outputs and reconstructs any cheap RNG-derived
    quantities directly from `rng_key`.
    """

    state: ReparameterizedSliceState
    logdensity_params: tuple
    direction: ArrayTree
    alpha_left: float
    alpha_right: float
    interpolation_weight: float


def _call_logdensity(
    logdensity_fn: Callable,
    position: ArrayLikeTree,
    logdensity_params: tuple,
):
    return logdensity_fn(position, *logdensity_params)


def _tree_add_scaled(
    tree: ArrayLikeTree, direction: ArrayLikeTree, scale: float
) -> ArrayTree:
    """Shift a pytree by a scaled direction."""

    return jax.tree.map(lambda x, d: x + scale * d, tree, direction)


def _tree_dot(left: ArrayLikeTree, right: ArrayLikeTree):
    """Euclidean dot product between two pytrees."""

    return jax.tree.reduce(
        operator.add,
        jax.tree.map(lambda x, y: (x * y).sum(), left, right),
    )


def _zero_tangent_like_leaf(leaf):
    """Construct a valid zero tangent for a primal leaf."""

    leaf = jnp.asarray(leaf)
    if jnp.issubdtype(leaf.dtype, jnp.inexact):
        return jnp.zeros_like(leaf)
    return jnp.zeros_like(leaf, dtype=jax.dtypes.float0)


def _zero_tangent_tree(tree: ArrayLikeTree):
    """Construct a pytree of valid zero tangents matching ``tree``."""

    return jax.tree.map(_zero_tangent_like_leaf, tree)


def _linearize_logdensity(
    logdensity_fn: Callable,
    position: ArrayLikeTree,
    logdensity_params: tuple,
):
    """Linearize the log-density with respect to position and parameters."""

    def density_from_position_and_params(pos, params):
        return _call_logdensity(logdensity_fn, pos, params)

    return jax.linearize(
        density_from_position_and_params,
        position,
        logdensity_params,
    )


def init(
    position: ArrayLikeTree,
    logdensity_fn: Callable,
    *logdensity_params,
) -> ReparameterizedSliceState:
    """Create an initial slice-sampling state from a position."""

    logdensity = _call_logdensity(logdensity_fn, position, logdensity_params)
    return ReparameterizedSliceState(position, logdensity)


def _sample_direction(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
    flat_position, unravel_fn = ravel_pytree(position)
    flat_direction = jax.random.normal(
        rng_key, flat_position.shape, dtype=flat_position.dtype
    )
    flat_direction = flat_direction / jnp.linalg.norm(flat_direction)
    return unravel_fn(flat_direction)


def _choose_bracket(
    func: Callable[[float], float],
    config: SliceRootFindingConfig,
) -> tuple[float, float, int]:
    """Expand a symmetric bracket until it crosses the slice on both sides."""

    initial_radius = jnp.power(10.0, config.bracket_log_start)
    left = -initial_radius
    right = initial_radius
    left_val = func(left)
    right_val = func(right)

    def cond_fn(val):
        _, _, left_val, right_val, step = val
        needs_expansion = jnp.logical_or(left_val > 0.0, right_val > 0.0)
        return jnp.logical_and(step < config.bracket_max_steps, needs_expansion)

    def body_fn(val):
        left, right, left_val, right_val, step = val
        next_step = step + 1
        radius = jnp.power(
            10.0, config.bracket_log_start + config.bracket_log_space * next_step
        )

        proposed_left = -radius
        proposed_right = radius
        next_left_val = func(proposed_left)
        next_right_val = func(proposed_right)

        update_left = left_val > 0.0
        update_right = right_val > 0.0

        left = jnp.where(update_left, proposed_left, left)
        right = jnp.where(update_right, proposed_right, right)
        left_val = jnp.where(update_left, next_left_val, left_val)
        right_val = jnp.where(update_right, next_right_val, right_val)
        return left, right, left_val, right_val, next_step

    left, right, _, _, step = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (left, right, left_val, right_val, jnp.array(0, dtype=jnp.int32)),
    )
    return left, right, step


def _bisect_root(
    func: Callable[[float], float],
    left: float,
    right: float,
    tolerance: float,
    max_steps: int,
) -> tuple[float, int]:
    """Bisect a bracketed scalar root."""

    left_val = func(left)
    right_val = func(right)

    def cond_fn(val):
        left, right, _, _, step = val
        interval_large = (right - left) > tolerance
        return jnp.logical_and(step < max_steps, interval_large)

    def body_fn(val):
        left, right, left_val, right_val, step = val
        midpoint = 0.5 * (left + right)
        midpoint_val = func(midpoint)
        same_sign_as_left = jnp.sign(midpoint_val) == jnp.sign(left_val)

        left = jnp.where(same_sign_as_left, midpoint, left)
        left_val = jnp.where(same_sign_as_left, midpoint_val, left_val)
        right = jnp.where(same_sign_as_left, right, midpoint)
        right_val = jnp.where(same_sign_as_left, right_val, midpoint_val)
        return left, right, left_val, right_val, step + 1

    left, right, _, _, step = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (left, right, left_val, right_val, jnp.array(0, dtype=jnp.int32)),
    )
    return 0.5 * (left + right), step


def _pathwise_forward(
    logdensity_fn: Callable,
    config: SliceRootFindingConfig,
    rng_key: PRNGKey,
    state: ReparameterizedSliceState,
    logdensity_params: tuple,
) -> tuple[
    tuple[ReparameterizedSliceState, ReparameterizedSliceInfo], SliceStepIntermediates
]:
    """Sample a slice level, solve the nearest slice roots, and interpolate."""

    position, logdensity = state
    key_slice, key_interp, key_direction = jax.random.split(rng_key, 3)
    slice_uniform = jax.random.uniform(key_slice)
    interpolation_weight = jax.random.uniform(key_interp)
    direction = _sample_direction(key_direction, position)

    log_slice = logdensity + jnp.log(slice_uniform)

    def root_fn(alpha):
        proposal = _tree_add_scaled(position, direction, alpha)
        return _call_logdensity(logdensity_fn, proposal, logdensity_params) - log_slice

    bracket_left, bracket_right, bracket_steps = _choose_bracket(root_fn, config)
    alpha_left, left_steps = _bisect_root(
        root_fn,
        bracket_left,
        -config.near_zero,
        config.root_tolerance,
        config.bisection_max_steps,
    )
    alpha_right, right_steps = _bisect_root(
        root_fn,
        config.near_zero,
        bracket_right,
        config.root_tolerance,
        config.bisection_max_steps,
    )

    alpha = (
        1.0 - interpolation_weight
    ) * alpha_left + interpolation_weight * alpha_right
    new_position = _tree_add_scaled(position, direction, alpha)
    new_logdensity = _call_logdensity(logdensity_fn, new_position, logdensity_params)
    new_state = ReparameterizedSliceState(new_position, new_logdensity)
    info = ReparameterizedSliceInfo(
        direction=direction,
        alpha_left=alpha_left,
        alpha_right=alpha_right,
        log_slice=log_slice,
        bracket_steps=bracket_steps,
        bisection_steps_left=left_steps,
        bisection_steps_right=right_steps,
    )
    residual = SliceStepIntermediates(
        state=state,
        logdensity_params=logdensity_params,
        direction=direction,
        alpha_left=alpha_left,
        alpha_right=alpha_right,
        interpolation_weight=interpolation_weight,
    )
    return (new_state, info), residual


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _pathwise_transition(
    logdensity_fn: Callable,
    config: SliceRootFindingConfig,
    rng_key: PRNGKey,
    state: ReparameterizedSliceState,
    logdensity_params: tuple,
) -> tuple[ReparameterizedSliceState, ReparameterizedSliceInfo]:
    (new_state, info), _ = _pathwise_forward(
        logdensity_fn, config, rng_key, state, logdensity_params
    )
    return new_state, info


@_pathwise_transition.defjvp
def _pathwise_transition_jvp(
    logdensity_fn: Callable,
    config: SliceRootFindingConfig,
    primals,
    tangents,
):
    """Differentiate the slice transition with implicit endpoint JVPs.

    This version follows JAX's documented higher-order-friendly custom-JVP
    pattern: the JVP rule calls the decorated primal function to compute the
    primal outputs. Cheap RNG-only quantities are reconstructed directly from
    ``rng_key`` rather than reused from `_pathwise_forward`.
    """

    rng_key, state, logdensity_params = primals
    rng_key_t, state_t, logdensity_params_t = tangents
    del rng_key_t

    new_state, info = _pathwise_transition(
        logdensity_fn,
        config,
        rng_key,
        state,
        logdensity_params,
    )
    outputs = (new_state, info)

    position = state.position
    position_t = state_t.position
    logdensity_t = state_t.logdensity
    direction = info.direction
    alpha_left = info.alpha_left
    alpha_right = info.alpha_right

    _key_slice, key_interp, _key_direction = jax.random.split(rng_key, 3)
    interpolation_weight = jax.random.uniform(key_interp)

    zero_direction_t = _zero_tangent_tree(direction)
    zero_params_t = _zero_tangent_tree(logdensity_params)

    def alpha_tangent(alpha: float):
        endpoint = _tree_add_scaled(position, direction, alpha)
        _, endpoint_lin = _linearize_logdensity(
            logdensity_fn,
            endpoint,
            logdensity_params,
        )
        numerator = endpoint_lin(position_t, logdensity_params_t) - logdensity_t
        denominator = endpoint_lin(direction, zero_params_t)
        return -numerator / denominator

    alpha_left_t = alpha_tangent(alpha_left)
    alpha_right_t = alpha_tangent(alpha_right)
    alpha_t = (
        1.0 - interpolation_weight
    ) * alpha_left_t + interpolation_weight * alpha_right_t

    new_position_t = _tree_add_scaled(position_t, direction, alpha_t)
    _, new_logdensity_lin = _linearize_logdensity(
        logdensity_fn,
        new_state.position,
        logdensity_params,
    )
    new_logdensity_t = new_logdensity_lin(new_position_t, logdensity_params_t)

    state_t_out = ReparameterizedSliceState(
        position=new_position_t,
        logdensity=new_logdensity_t,
    )
    info_t_out = ReparameterizedSliceInfo(
        direction=zero_direction_t,
        alpha_left=alpha_left_t,
        alpha_right=alpha_right_t,
        log_slice=logdensity_t,
        bracket_steps=_zero_tangent_like_leaf(info.bracket_steps),
        bisection_steps_left=_zero_tangent_like_leaf(info.bisection_steps_left),
        bisection_steps_right=_zero_tangent_like_leaf(info.bisection_steps_right),
    )
    return outputs, (state_t_out, info_t_out)


def build_kernel(
    *,
    root_tolerance: float = 1e-6,
    near_zero: float = 1e-8,
    bracket_log_start: float = -3.0,
    bracket_log_space: float = 0.2,
    bracket_max_steps: int = 100,
    bisection_max_steps: int = 100,
):
    """Build a reparameterized random-direction slice kernel :cite:p:`zoltowski2021slice`.

    Parameters
    ----------
    root_tolerance
        The stopping tolerance used by the endpoint bisection solver.
    near_zero
        Small offset used to keep the left and right root searches away from
        the current point.
    bracket_log_start
        Base-10 logarithm of the initial stepping-out radius.
    bracket_log_space
        Increment in base-10 logarithmic radius applied at each stepping-out
        update.
    bracket_max_steps
        Maximum number of stepping-out updates used to bracket both slice
        endpoints.
    bisection_max_steps
        Maximum number of bisection updates used on each side once a bracket
        has been found.

    Returns
    -------
    A kernel that takes a rng_key, a state, a logdensity function, and any
    explicit positional runtime logdensity parameters, and returns a new state of
    the chain along with information about the transition.

    The returned kernel is pathwise differentiable with respect to the incoming
    state and the explicit positional logdensity parameters.
    """

    config = SliceRootFindingConfig(
        root_tolerance=root_tolerance,
        near_zero=near_zero,
        bracket_log_start=bracket_log_start,
        bracket_log_space=bracket_log_space,
        bracket_max_steps=bracket_max_steps,
        bisection_max_steps=bisection_max_steps,
    )

    def kernel(
        rng_key: PRNGKey,
        state: ReparameterizedSliceState,
        logdensity_fn: Callable,
        *logdensity_params,
    ) -> tuple[ReparameterizedSliceState, ReparameterizedSliceInfo]:
        return _pathwise_transition(
            logdensity_fn,
            config,
            rng_key,
            state,
            logdensity_params,
        )

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    *,
    root_tolerance: float = 1e-6,
    near_zero: float = 1e-8,
    bracket_log_start: float = -3.0,
    bracket_log_space: float = 0.2,
    bracket_max_steps: int = 100,
    bisection_max_steps: int = 100,
) -> SamplingAlgorithm:
    """Implements the user interface for the reparameterized slice kernel.

    The implementation follows :cite:p:`zoltowski2021slice` and is adapted from
    the reference research code :cite:p:`princetonlips2021slicereparam`.

    Examples
    --------

    If the log-density function takes positional parameters in addition to the
    position,
    these can be passed explicitly to both `init` and `step`.

    .. code::

        algo = blackjax.reparameterized_slice(logdensity_fn)
        state = algo.init(position, param1, param2, etc.)

        def one_step(state, rng_key):
            new_state, info = algo.step(rng_key, state, param1, param2, etc.)
            return new_state, info

        keys = jax.random.split(rng_key, num_steps)
        final_state, infos = jax.lax.scan(one_step, state, keys)

    If the log-density function only takes position as an argument, the same sampler
    can also be used like any other blackjax sampler.

    .. code::

        algo = blackjax.reparameterized_slice(logdensity_fn)
        state = algo.init(position)
        new_state, info = algo.step(rng_key, state)

    The log-density parameters are positional-only; you can bundle several
    differentiable quantities into one pytree or pass them as several
    positional arguments.

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from. It should accept
        the chain position as its first argument and any differentiable parameters
        as additional positional arguments.
    root_tolerance
        The stopping tolerance used by the endpoint bisection solver.
    near_zero
        Small offset used to keep the left and right root searches away from
        the current point.
    bracket_log_start
        Base-10 logarithm of the initial stepping-out radius.
    bracket_log_space
        Increment in base-10 logarithmic radius applied at each stepping-out
        update.
    bracket_max_steps
        Maximum number of stepping-out updates used to bracket both slice
        endpoints.
    bisection_max_steps
        Maximum number of bisection updates used on each side once a bracket
        has been found.

    Returns
    -------
    A ``SamplingAlgorithm`` whose ``init`` and ``step`` methods accept
    additional positional runtime logdensity parameters.
    """

    kernel = build_kernel(
        root_tolerance=root_tolerance,
        near_zero=near_zero,
        bracket_log_start=bracket_log_start,
        bracket_log_space=bracket_log_space,
        bracket_max_steps=bracket_max_steps,
        bisection_max_steps=bisection_max_steps,
    )

    def init_fn(
        position: ArrayLikeTree,
        *logdensity_params,
        rng_key=None,
    ):
        del rng_key
        return init(position, logdensity_fn, *logdensity_params)

    def step_fn(
        rng_key: PRNGKey,
        state: ReparameterizedSliceState,
        *logdensity_params,
    ):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            *logdensity_params,
        )

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
