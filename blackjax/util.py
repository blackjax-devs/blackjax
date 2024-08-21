"""Utility functions for BlackJax."""

from functools import partial
from typing import Callable, Union

import jax.numpy as jnp
from jax import jit, lax
from jax.flatten_util import ravel_pytree
from jax.random import normal, split
from jax.tree_util import tree_leaves, tree_map

from blackjax.base import SamplingAlgorithm, VIAlgorithm
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey


@partial(jit, static_argnames=("precision",), inline=True)
def linear_map(diag_or_dense_a, b, *, precision="highest"):
    """Perform a linear map of the form y = Ax.

    Dispatch matrix multiplication to either jnp.dot or jnp.multiply.

    Unlike jax.numpy.dot, this function output an Array that match the dtype
    and shape of the 2nd input:
    - diag_or_dense_a is a scalar or 1d vector, `diag_or_dense_a * b` is returned
    - diag_or_dense_a is a 2d matrix, `diag_or_dense_a @ b` is returned

    Note that unlike jax.numpy.dot, here we defaults to full (highest)
    precision. This is more useful for numerical algorithms and will be the
    default for jax.numpy in the future:
    https://github.com/google/jax/pull/7859

    Parameters
    ----------
    diag_or_dense_a:
        A diagonal (1d vector) or dense matrix (2d square matrix).
    b:
        A vector.
    precision:
        The precision of the computation. See jax.lax.dot_general for
        more details.

    Returns
    -------
        The result vector of the matrix multiplication.
    """
    dtype = jnp.result_type(diag_or_dense_a.dtype, b.dtype)
    diag_or_dense_a = diag_or_dense_a.astype(dtype)
    b = b.astype(dtype)
    ndim = jnp.ndim(diag_or_dense_a)

    if ndim <= 1:
        return lax.mul(diag_or_dense_a, b)
    else:
        return lax.dot(diag_or_dense_a, b, precision=precision)


# TODO(https://github.com/blackjax-devs/blackjax/issues/376)
# Refactor this function to not use ravel_pytree might be more performant.
def generate_gaussian_noise(
    rng_key: PRNGKey,
    position: ArrayLikeTree,
    mu: Union[float, Array] = 0.0,
    sigma: Union[float, Array] = 1.0,
) -> ArrayTree:
    """Generate N(mu, sigma) noise with output structure that match a given PyTree.

    Parameters
    ----------
    rng_key:
        The pseudo-random number generator key used to generate random numbers.
    position:
        PyTree that the structure the output should to match.
    mu:
        The mean of the Gaussian distribution.
    sigma:
        The standard deviation of the Gaussian distribution.

    Returns
    -------
    Gaussian noise following N(mu, sigma) that match the structure of position.
    """
    p, unravel_fn = ravel_pytree(position)
    sample = normal(rng_key, shape=p.shape, dtype=p.dtype)
    return unravel_fn(mu + linear_map(sigma, sample))


def generate_unit_vector(
    rng_key: PRNGKey,
    position: ArrayLikeTree,
) -> Array:
    """Generate a random unit vector with output structure that match a given PyTree.

    Parameters
    ----------
    rng_key:
        The pseudo-random number generator key used to generate random numbers.
    position:
        PyTree that the structure the output should to match.

    Returns
    -------
    Random unit vector that match the structure of position.
    """
    p, unravel_fn = ravel_pytree(position)
    sample = normal(rng_key, shape=p.shape, dtype=p.dtype)
    return unravel_fn(sample / jnp.linalg.norm(sample))


def pytree_size(pytree: ArrayLikeTree) -> int:
    """Return the dimension of the flatten PyTree."""
    return sum(jnp.size(value) for value in tree_leaves(pytree))


def index_pytree(input_pytree: ArrayLikeTree) -> ArrayTree:
    """Builds a PyTree with elements indicating its corresponding index on a flat array.

    Various algorithms in BlackJAX take as input a 1 or 2 dimensional array which somehow
    affects the sampling or approximation of a PyTree. For instance, in HMC a 1 or 2
    dimensional inverse mass matrix is used when simulating Hamilonian dynamics on
    PyTree position and momentum variables. It is usually unclear how the elements of the
    array interact with the PyTree. This function demonstrates how all algorithms map an
    array to a PyTree of equivalent dimension.

    The function returns the index of a 1 dimensional array corresponding to each element of
    the PyTree. This way the user can tell which element in the PyTree corresponds to which
    column (and row) of a 1 dimensional (or 2 dimensional) array.

    Parameters
    ----------
    input_pytree:
        Example PyTree.

    Returns
    -------
    PyTree mapping each individual element of an arange array to elements in the PyTree.
    """
    flat_input, unravel_fn = ravel_pytree(input_pytree)
    (dim_input,) = flat_input.shape
    array = jnp.arange(dim_input, dtype=flat_input.dtype)
    return unravel_fn(array)


def run_inference_algorithm(
    rng_key: PRNGKey,
    inference_algorithm: Union[SamplingAlgorithm, VIAlgorithm],
    num_steps: int,
    initial_state: ArrayLikeTree = None,
    initial_position: ArrayLikeTree = None,
    progress_bar: bool = False,
    transform: Callable = lambda state, info: (state, info),
) -> tuple:
    """Wrapper to run an inference algorithm.

    Note that this utility function does not work for Stochastic Gradient MCMC samplers
    like sghmc, as SG-MCMC samplers require additional control flow for batches of data
    to be passed in during each sample.

    Parameters
    ----------
    rng_key
        The random state used by JAX's random numbers generator.
    initial_state
        The initial state of the inference algorithm.
    initial_position
        The initial position of the inference algorithm. This is used when the initial state is not provided.
    inference_algorithm
        One of blackjax's sampling algorithms or variational inference algorithms.
    num_steps
        Number of MCMC steps.
    progress_bar
        Whether to display a progress bar.
    transform
        A transformation of the trace of states (and info) to be returned. This is useful for
        computing determinstic variables, or returning a subset of the states.
        By default, the states are returned as is.

    Returns
    -------
        1. The final state.
        2. The history of states.
    """

    if initial_state is None and initial_position is None:
        raise ValueError(
            "Either `initial_state` or `initial_position` must be provided."
        )
    if initial_state is not None and initial_position is not None:
        raise ValueError(
            "Only one of `initial_state` or `initial_position` must be provided."
        )

    if initial_state is None:
        rng_key, init_key = split(rng_key, 2)
        initial_state = inference_algorithm.init(initial_position, init_key)

    keys = split(rng_key, num_steps)

    def one_step(state, xs):
        _, rng_key = xs
        state, info = inference_algorithm.step(rng_key, state)
        return state, transform(state, info)

    scan_fn = gen_scan_fn(num_steps, progress_bar)

    xs = jnp.arange(num_steps), keys
    final_state, history = scan_fn(one_step, initial_state, xs)

    return final_state, history


def store_only_expectation_values(
    sampling_algorithm,
    state_transform=lambda x: x,
    incremental_value_transform=lambda x: x,
    burn_in=0,
):
    """Takes a sampling algorithm and constructs from it a new sampling algorithm object. The new sampling algorithm has the same
     kernel but only stores the streaming expectation values of some observables, not the full states; to save memory.

    It saves incremental_value_transform(E[state_transform(x)]) at each step i, where expectation is computed with samples up to i-th sample.

    Example:

    .. code::

         init_key, state_key, run_key = jax.random.split(jax.random.PRNGKey(0),3)
         model = StandardNormal(2)
         initial_position = model.sample_init(init_key)
         initial_state = blackjax.mcmc.mclmc.init(
             position=initial_position, logdensity_fn=model.logdensity_fn, rng_key=state_key
         )
         integrator_type = "mclachlan"
         L = 1.0
         step_size = 0.1
         num_steps = 4

         integrator = map_integrator_type_to_integrator['mclmc'][integrator_type]
         state_transform = lambda state: state.position
         memory_efficient_sampling_alg, transform = store_only_expectation_values(
             sampling_algorithm=sampling_alg,
             state_transform=state_transform)

         initial_state = memory_efficient_sampling_alg.init(initial_state)

         final_state, trace_at_every_step = run_inference_algorithm(

             rng_key=run_key,
             initial_state=initial_state,
             inference_algorithm=memory_efficient_sampling_alg,
             num_steps=num_steps,
             transform=transform,
             progress_bar=True,
         )
    """

    def init_fn(state):
        averaging_state = (0.0, state_transform(state))
        return (state, averaging_state)

    def update_fn(rng_key, state_and_incremental_val):
        state, averaging_state = state_and_incremental_val
        state, info = sampling_algorithm.step(
            rng_key, state
        )  # update the state with the sampling algorithm
        averaging_state = incremental_value_update(
            state_transform(state),
            averaging_state,
            weight=(
                averaging_state[0] >= burn_in
            ),  # If we want to eliminate some number of steps as a burn-in
            zero_prevention=1e-10 * (burn_in > 0),
        )
        # update the expectation value with the running average
        return (state, averaging_state), info

    def transform(state_and_incremental_val, info):
        (state, (_, incremental_value)) = state_and_incremental_val
        return incremental_value_transform(incremental_value), info

    return SamplingAlgorithm(init_fn, update_fn), transform


def safediv(x, y):
    return jnp.where(x == 0.0, 0.0, x / y)


def incremental_value_update(
    expectation, incremental_val, weight=1.0, zero_prevention=0.0
):
    """Compute the streaming average of a function O(x) using a weight.
    Parameters:
    ----------
        expectation
            the value of the expectation at the current timestep
        incremental_val
            tuple of (total, average) where total is the sum of weights and average is the current average
        weight
            weight of the current state
        zero_prevention
            small value to prevent division by zero
    Returns:
    ----------
        new streaming average
    """

    total, average = incremental_val
    average = tree_map(
        lambda exp, av: safediv(
            total * av + weight * exp, (total + weight + zero_prevention)
        ),
        expectation,
        average,
    )
    total += weight
    return total, average
