import jax
import jax.numpy as jnp
from jax import device_put, jit, lax, vmap
from jax.experimental.shard_map import shard_map
from jax.flatten_util import ravel_pytree
from jax.random import normal, split
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_leaves, tree_map

import blackjax.adaptation.ensemble_umclmc as umclmc


def eca_step(
    kernel, summary_statistics_fn, adaptation_update, num_chains, ensemble_info=None
):
    """
    Construct a single step of ensemble chain adaptation (eca) to be performed in parallel on multiple devices.
    """

    def _step(state_all, xs):
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
        theta = vmap(summary_statistics_fn, (0, 0, None))(state, info, key_adaptation)
        Etheta = tree_map(
            lambda theta: lax.psum(jnp.sum(theta, axis=0), axis_name="chains")
            / num_chains,
            theta,
        )

        # use these to adapt the hyperparameters of the dynamics
        adaptation_state, info_to_be_stored = adaptation_update(
            adaptation_state, Etheta
        )

        return (state, adaptation_state), info_to_be_stored

    if ensemble_info is not None:

        def step(state_all, xs):
            (state, adaptation_state), info_to_be_stored = _step(state_all, xs)
            return (state, adaptation_state), (
                info_to_be_stored,
                vmap(ensemble_info)(state.position),
            )

        return step

    else:
        return _step




def ensemble_execute_fn(
    func,
    rng_key,
    num_chains,
    mesh,
    x=None,
    args=None,
    summary_statistics_fn=lambda y: 0.0,
):
    """Given a sequential function
     func(rng_key, x, args) = y,
    evaluate it with an ensemble and also compute some summary statistics E[theta(y)], where expectation is taken over ensemble.
    Args:
         x: array distributed over all decvices
         args: additional arguments for func, not distributed.
         summary_statistics_fn: operates on a single member of ensemble and returns some summary statistics.
         rng_key: a single random key, which will then be split, such that each member of an ensemble will get a different random key.

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

    keys = device_put(
        split(rng_key, num_chains), NamedSharding(mesh, p)
    )  # random keys, distributed across devices
    # apply F in parallel
    return parallel_execute(X, keys)

def run_eca(
    rng_key,
    initial_state,
    kernel,
    adaptation,
    num_steps,
    num_chains,
    mesh,
    ensemble_info=None,
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
        ensemble_info: function that takes the state of the system and returns some information about the ensemble
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
        ensemble_info,
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

        # ((a, Int) -> (a, Int))
        def step_while(a):
            x, i, _ = a

            auxilliary_input = (xs[0][i], xs[1][i], xs[2][i])

            output, info = step(x, auxilliary_input)

            return (output, i + 1, info[0].get("while_cond"))

        if early_stop:
            final_state_all, i, _ = lax.while_loop(
                lambda a: ((a[1] < num_steps) & a[2]),
                step_while,
                (initial_state_all, 0, True),
            )
            info_history = None

        else:
            final_state_all, info_history = lax.scan(step, initial_state_all, xs)

        final_state, final_adaptation_state = final_state_all
        return (
            final_state,
            final_adaptation_state,
            info_history,
        )  # info history is composed of averages over all chains, so it is a couple of scalars

    p, pscalar = PartitionSpec("chains"), PartitionSpec()
    parallel_execute = shard_map(
        all_steps,
        mesh=mesh,
        in_specs=(p, p, pscalar),
        out_specs=(p, pscalar, pscalar),
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
    final_state, final_adaptation_state, info_history = parallel_execute(
        initial_state, keys_sampling, keys_adaptation
    )

    return final_state, final_adaptation_state, info_history


mesh = jax.sharding.Mesh(devices=jax.devices(),axis_names= "chains")

key_init, key_umclmc, key_mclmc = jax.random.split(jax.random.key(0), 3)

num_chains = 128
ndims = 2

def logdensity_fn(x):
        mu2 = 0.03 * (x[0] ** 2 - 100)
        return -0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))

def transform(x):
        return x

def sample_init(key):
        z = jax.random.normal(key, shape=(2,))
        x0 = 10.0 * z[0]
        x1 = 0.03 * (x0**2 - 100) + z[1]
        return jnp.array([x0, x1])

# initialize the chains
initial_state = umclmc.initialize(
    key_init, logdensity_fn, sample_init, num_chains, mesh
)

alpha = 1.9
C = 0.1
r_end=5e-3
ensemble_observables=lambda x: x

# burn-in with the unadjusted method #
kernel = umclmc.build_kernel(logdensity_fn)
save_num = 20 # (int)(jnp.rint(save_frac * num_steps1))
adap = umclmc.Adaptation(
    ndims,
    alpha=alpha,
    bias_type=3,
    save_num=save_num,
    C=C,
    power=3.0 / 8.0,
    r_end=r_end,
    observables_for_bias=lambda position: jnp.square(
        transform(jax.flatten_util.ravel_pytree(position)[0])
    ),
)


final_state, final_adaptation_state, info1 = run_eca(
        key_umclmc,
        initial_state,
        kernel,
        adap,
        100,
        num_chains,
        mesh,
        ensemble_observables,
        early_stop=True,
    )


# a = jnp.array([8.0, 4.0])

# def f(rng_key, x, args):
#     return x + normal(rng_key, x.shape) + a, a

# out = ensemble_execute_fn(
#     func = f,
#     rng_key = jax.random.PRNGKey(0),
#     num_chains = 4,
#     mesh = mesh,
#     x = None,
#     args = None,
#     summary_statistics_fn = lambda y: a,
# )

# print(out)