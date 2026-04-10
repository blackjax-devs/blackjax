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
"""Ensemble Chain Adaptation (ECA) utilities for multi-device parallel sampling."""

import jax
import jax.numpy as jnp
from jax import device_put, lax, shard_map, vmap
from jax.random import split
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_map

from blackjax.diagnostics import splitR


def eca_step(
    kernel,
    summary_statistics_fn,
    adaptation_update,
    num_chains,
    superchain_size=None,
    all_chains_info=None,
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
        summary_statistics = vmap(summary_statistics_fn, (0, 0, None))(
            state, info, key_adaptation
        )
        expected_value_summary_statistics = tree_map(
            lambda summary_statistics: lax.psum(
                jnp.sum(summary_statistics, axis=0), axis_name="chains"
            )
            / num_chains,
            summary_statistics,
        )

        # use these to adapt the hyperparameters of the dynamics
        adaptation_state, info_to_be_stored = adaptation_update(
            adaptation_state, expected_value_summary_statistics
        )

        return (state, adaptation_state), info_to_be_stored

    return add_all_chains_info(
        add_splitR(step, num_chains, superchain_size), all_chains_info
    )


def add_splitR(step, num_chains, superchain_size):
    def _step_with_R(state_all, xs):
        state_all, info_to_be_stored = step(state_all, xs)

        state, adaptation_state = state_all

        R = splitR(state.position, num_chains, superchain_size)
        split_bavg = jnp.average(jnp.square(R) - 1)
        split_bmax = jnp.max(jnp.square(R) - 1)

        info_to_be_stored["R_avg"] = split_bavg
        info_to_be_stored["R_max"] = split_bmax

        return (state, adaptation_state), info_to_be_stored

    def _step_with_R_1(state_all, xs):
        state_all, info_to_be_stored = step(state_all, xs)

        info_to_be_stored["R_avg"] = 0.0
        info_to_be_stored["R_max"] = 0.0

        return state_all, info_to_be_stored

    if superchain_size is None:
        return step

    if superchain_size == 1:
        return _step_with_R_1

    else:
        return _step_with_R


def add_all_chains_info(step, all_chains_info):
    def _step(state_all, xs):
        (state, adaptation_state), info_to_be_stored = step(state_all, xs)
        info_to_be_stored["all_chains_info"] = vmap(all_chains_info)(state.position)

        return (state, adaptation_state), info_to_be_stored

    return _step if all_chains_info is not None else step


def while_with_info(step, init, xs, length, while_cond):
    """Same syntax and usage as jax.lax.scan, but it is run as a while loop that is terminated if not while_cond(state).
    len(xs) determines the maximum number of iterations.
    """

    get_i = lambda tree, i: jax.tree.map(lambda arr: arr[i], tree)

    info1 = step(init, get_i(xs, 0))[
        1
    ]  # call the step once to determine the shape of info
    info = jax.lax.scan(lambda x, _: (x, info1), init=0, length=length)[
        1
    ]  # allocate the full info by repeating values

    init_val = (init, info, 0, while_cond(info1, 0))

    def body_fun(val):
        x, info_old, counter, cond = val

        x_new, info_new = step(x, get_i(xs, counter))

        # update the full info by adding the new one
        info_full = jax.tree.map(
            lambda arr, val: arr.at[counter].set(val), info_old, info_new
        )

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
    superchain_size=None,
    all_chains_info=None,
    early_stop=False,
):
    """
    Run ensemble chain adaptation (eca) in parallel on multiple devices.

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
        superchain_size=superchain_size,
        all_chains_info=all_chains_info,
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
            final_state_all, info_history, counter = while_with_info(
                step, initial_state_all, xs, num_steps, adaptation.while_cond
            )

        else:
            final_state_all, info_history = lax.scan(step, initial_state_all, xs)
            counter = num_steps

        final_state, final_adaptation_state = final_state_all

        return (
            final_state,
            final_adaptation_state,
            info_history,
            counter,
        )  # info history is composed of averages over all chains, so it is a couple of scalars

    p, pscalar = PartitionSpec("chains"), PartitionSpec()
    parallel_execute = shard_map(
        all_steps,
        mesh=mesh,
        in_specs=(p, p, pscalar),
        out_specs=(p, pscalar, pscalar, pscalar),
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
    info_history = jax.tree.map(lambda arr: arr[: int(counter)], info_history)

    return final_state, final_adaptation_state, info_history


def ensemble_execute_fn(
    func,
    rng_key,
    num_chains,
    mesh,
    x=None,
    args=None,
    summary_statistics_fn=lambda y: 0.0,
    superchain_size=None,
):
    """Given a sequential function ``func(rng_key, x, args) = y``, evaluate it
    with an ensemble and compute summary statistics ``E[theta(y)]``.

    Args:
        x: array distributed over all devices
        args: additional arguments for func, not distributed.
        summary_statistics_fn: operates on a chain and returns some summary statistics.
        rng_key: a single random key, which will then be split so each chain gets a different key.

    Returns:
        y: array distributed over all devices. Need not be of the same shape as x.
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

    parallel_execute = shard_map(F, mesh=mesh, in_specs=(p, p), out_specs=(p, pscalar))

    if superchain_size == 1:
        _keys = split(rng_key, num_chains)

    else:
        _keys = jnp.repeat(
            split(rng_key, num_chains // superchain_size), superchain_size
        )

    keys = device_put(
        _keys, NamedSharding(mesh, p)
    )  # random keys, distributed across devices

    # apply F in parallel
    return parallel_execute(X, keys)
