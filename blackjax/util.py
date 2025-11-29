"""Utility functions for BlackJax."""

from functools import partial
from typing import Callable, Union

import jax.numpy as jnp
from jax import device_put, jit, lax, vmap
from jax.experimental.shard_map import shard_map
from jax.flatten_util import ravel_pytree
from jax.random import normal, split
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_leaves, tree_map

from blackjax.base import SamplingAlgorithm, VIAlgorithm
from blackjax.progress_bar import gen_scan_fn
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.diagnostics import splitR
import time
import jax

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
    # tic = time.time()
    final_state, history = scan_fn(one_step, initial_state, xs)
    # toc = time.time()

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


def eca_step(
    kernel, summary_statistics_fn, adaptation_update, num_chains, superchain_size = None, all_chains_info=None
):
    """
    Construct a single step of ensemble chain adaptation (eca) to be performed in parallel on multiple devices.
    """

    def step(state_all, xs):
        """This function operates on a single device."""
        (
            state,
            adaptation_state,
        ) = state_all  # state is an array of states, one for each chain on this device. adaptation_state is the same for all chains, so it is not an array.
        (
            _,
            keys_sampling,
            key_adaptation,
        ) = xs  # keys_sampling.shape = (chains_per_device, )

        # update the state of all chains on this device
        state, info = vmap(kernel, (0, 0, None))(keys_sampling, state, adaptation_state)

        # combine all the chains to compute expectation values
        summary_statistics = vmap(summary_statistics_fn, (0, 0, None))(state, info, key_adaptation)
        expected_value_summary_statistics = tree_map(
            lambda summary_statistics: lax.psum(jnp.sum(summary_statistics, axis=0), axis_name="chains")
            / num_chains,
            summary_statistics,
        )

        # use these to adapt the hyperparameters of the dynamics
        adaptation_state, info_to_be_stored = adaptation_update(
            adaptation_state, expected_value_summary_statistics
        )

        return (state, adaptation_state), info_to_be_stored


    return add_all_chains_info(add_splitR(step, num_chains, superchain_size), all_chains_info)


def add_splitR(step, num_chains, superchain_size):

    
    def _step_with_R(state_all, xs):

        state_all, info_to_be_stored  = step(state_all, xs)
        
        state, adaptation_state = state_all

        R = splitR(state.position, num_chains, superchain_size)
        split_bavg = jnp.average(jnp.square(R) - 1)
        split_bmax = jnp.max(jnp.square(R) - 1)

        info_to_be_stored['R_avg'] = split_bavg
        info_to_be_stored['R_max'] = split_bmax

        return (state, adaptation_state), info_to_be_stored

    def _step_with_R_1(state_all, xs):

        state_all, info_to_be_stored  = step(state_all, xs)
        
        info_to_be_stored['R_avg'] = 0.
        info_to_be_stored['R_max'] = 0.

        return state_all, info_to_be_stored
    
    if superchain_size == None:
        return step
    
    if superchain_size == 1:
        return _step_with_R_1
    
    else:
        return _step_with_R
    
    
def add_all_chains_info(step, all_chains_info):

    def _step(state_all, xs):
        (state, adaptation_state), info_to_be_stored = step(state_all, xs)
        info_to_be_stored['all_chains_info'] = vmap(all_chains_info)(state.position)

        return (state, adaptation_state), info_to_be_stored

    return _step if all_chains_info is not None else step




def while_with_info(step, init, xs, length, while_cond):
    """Same syntax and usage as jax.lax.scan, but it is run as a while loop that is terminated if not while_cond(state).
        len(xs) determines the maximum number of iterations.
    """

    get_i = lambda tree, i: jax.tree_util.tree_map(lambda arr: arr[i], tree)

    info1 = step(init, get_i(xs, 0))[1] # call the step once to determine the shape of info
    info = jax.lax.scan(lambda x, _: (x, info1), init= 0, length= length)[1] # allocate the full info by repeating values

    init_val = (init, info, 0, while_cond(info1, 0))

    def body_fun(val):
        x, info_old, counter, cond = val
        
        x_new, info_new = step(x, get_i(xs, counter))
        
        # update the full info by adding the new one
        info_full = jax.tree_util.tree_map(lambda arr, val: arr.at[counter].set(val), info_old, info_new) 

        cond = while_cond(info_new, counter)

        return x_new, info_full, counter + 1, cond


    def cond_fun(val):
        _, _, counter, cond = val
        return cond & (counter < length)
    
    
    final, info, counter, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return final, info, counter


def run_eca(
    rng_key,
    initial_state,
    kernel,
    adaptation,
    num_steps,
    num_chains,
    mesh,
    superchain_size= None,
    all_chains_info=None,
    early_stop=False,
):
    """
    Run ensemble chain adaptation (eca) in parallel on multiple devices.
    -----------------------------------------------------
    Args:
        rng_key: random key
        initial_state: initial state of the system
        kernel: kernel for the dynamics
        adaptation: adaptation object
        num_steps: number of steps to run
        num_chains: number of chains
        mesh: mesh for parallelization
        all_chains_info: function that takes the state of the system and returns some summary statistics. Will be applied and stored for all the chains at each step so it can be memory intensive.
        early_stop: whether to stop early
    Returns:
        final_state: final state of the system
        final_adaptation_state: final adaptation state
        info_history: history of the information that was stored at each step (if early_stop is False, then this is None)
    """

    step = eca_step(
        kernel,
        adaptation.summary_statistics_fn,
        adaptation.update,
        num_chains,
        superchain_size= superchain_size,
        all_chains_info = all_chains_info,
    )

    def all_steps(initial_state, keys_sampling, keys_adaptation):
        """This function operates on a single device. key is a random key for this device."""

        initial_state_all = (initial_state, adaptation.initial_state)

        # run sampling
        xs = (
            jnp.arange(num_steps),
            keys_sampling.T,
            keys_adaptation,
        )  # keys for all steps that will be performed. keys_sampling.shape = (num_steps, chains_per_device), keys_adaptation.shape = (num_steps, )


        if early_stop:
            final_state_all, info_history, counter = while_with_info(step, initial_state_all, xs, num_steps, adaptation.while_cond)

        else:
            final_state_all, info_history = lax.scan(step, initial_state_all, xs)
            counter = num_steps

        final_state, final_adaptation_state = final_state_all
        
        return final_state, final_adaptation_state, info_history, counter  # info history is composed of averages over all chains, so it is a couple of scalars


    p, pscalar = PartitionSpec("chains"), PartitionSpec()
    parallel_execute = shard_map(
        all_steps,
        mesh=mesh,
        in_specs=(p, p, pscalar),
        out_specs=(p, pscalar, pscalar, pscalar),
        check_rep=False,
    )

    # produce all random keys that will be needed

    key_sampling, key_adaptation = split(rng_key)
    num_steps = jnp.array(num_steps).item()
    keys_adaptation = split(key_adaptation, num_steps)
    distribute_keys = lambda key, shape: device_put(
        split(key, shape), NamedSharding(mesh, p)
    )  # random keys, distributed across devices
    keys_sampling = distribute_keys(key_sampling, (num_chains, num_steps))

    # run sampling in parallel
    final_state, final_adaptation_state, info_history, counter = parallel_execute(
        initial_state, keys_sampling, keys_adaptation
    )

    # info_history has a static size, determined by num_steps, but if early_stop = True, the values after the while condition has been violated are nonsense. Remove them:
    info_history = jax.tree_util.tree_map(lambda arr: arr[:int(counter)], info_history)

    return final_state, final_adaptation_state, info_history


def ensemble_execute_fn(
    func,
    rng_key,
    num_chains,
    mesh,
    x=None,
    args=None,
    summary_statistics_fn=lambda y: 0.0,
    superchain_size = None
):
    """Given a sequential function
     func(rng_key, x, args) = y,
    evaluate it with an ensemble and also compute some summary statistics E[theta(y)], where expectation is taken over ensemble.
    Args:
         x: array distributed over all decvices
         args: additional arguments for func, not distributed.
         summary_statistics_fn: operates on a chain and returns some summary statistics.
         rng_key: a single random key, which will then be split, such that chain will get a different random key.

    Returns:
         y: array distributed over all decvices. Need not be of the same shape as x.
         Etheta: expected values of the summary statistics
    """
    p, pscalar = PartitionSpec("chains"), PartitionSpec()

    if x is None:
        X = device_put(jnp.zeros(num_chains), NamedSharding(mesh, p))
    else:
        X = x

    adaptation_update = lambda _, Etheta: (Etheta, None)

    _F = eca_step(
        func,
        lambda y, info, key: summary_statistics_fn(y),
        adaptation_update,
        num_chains,
    )

    def F(x, keys):
        """This function operates on a single device. key is a random key for this device."""
        y, summary_statistics = _F((x, args), (None, keys, None))[0]
        return y, summary_statistics

    parallel_execute = shard_map(
        F, mesh=mesh, in_specs=(p, p), out_specs=(p, pscalar), check_rep=False
    )

    if superchain_size == 1:
        _keys = split(rng_key, num_chains)
    
    else:
        _keys = jnp.repeat(split(rng_key, num_chains // superchain_size), superchain_size)


    keys = device_put(_keys, NamedSharding(mesh, p))  # random keys, distributed across devices
    
    # apply F in parallel
    return parallel_execute(X, keys)




