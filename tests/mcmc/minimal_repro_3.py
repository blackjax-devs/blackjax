

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

import blackjax.adaptation.ensemble_umclmc as umclmc
from blackjax.adaptation.ensemble_umclmc import (
    equipartition_diagonal,
    equipartition_diagonal_loss,
)
from blackjax.adaptation.step_size import bisection_monotonic_fn
from blackjax.mcmc.adjusted_mclmc import build_kernel as build_kernel_malt
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import (
    generate_isokinetic_integrator,
    mclachlan_coefficients,
    omelyan_coefficients,
)
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

# from blackjax.util import run_eca



class AdaptationState(NamedTuple):
    steps_per_sample: float
    step_size: float
    stepsize_adaptation_state: (
        Any  # the state of the bisection algorithm to find a stepsize
    )
    iteration: int


build_kernel = lambda logdensity_fn, integrator, inverse_mass_matrix: lambda key, state, adap: build_kernel_malt(
    logdensity_fn=logdensity_fn,
    integrator=integrator,
    inverse_mass_matrix=inverse_mass_matrix,
)(
    rng_key=key,
    state=state,
    step_size=adap.step_size,
    num_integration_steps=adap.steps_per_sample,
    L_proposal_factor=1.25,
)


class Adaptation:
    def __init__(
        self,
        adaptation_state,
        num_adaptation_samples,  # amount of tuning in the adjusted phase before fixing params
        steps_per_sample=15,  # L/eps
        acc_prob_target=0.8,
        observables=lambda x: 0.0,  # just for diagnostics: some function of a given chain at given timestep
        observables_for_bias=lambda x: 0.0,  # just for diagnostics: the above, but averaged over all chains
        contract=lambda x: 0.0,  # just for diagnostics: observabiels for bias, contracted over dimensions
    ):
        self.num_adaptation_samples = num_adaptation_samples
        self.observables = observables
        self.observables_for_bias = observables_for_bias
        self.contract = contract

        # Determine the initial hyperparameters #

        # stepsize #
        # if we switched to the more accurate integrator we can use longer step size
        # integrator_factor = jnp.sqrt(10.) if mclachlan else 1.
        # Let's use the stepsize which will be optimal for the adjusted method. The energy variance after N steps scales as sigma^2 ~ N^2 eps^6 = eps^4 L^2
        # In the adjusted method we want sigma^2 = 2 mu = 2 * 0.41 = 0.82
        # With the current eps, we had sigma^2 = EEVPD * d for N = 1.
        # Combining the two we have EEVPD * d / 0.82 = eps^6 / eps_new^4 L^2
        # adjustment_factor = jnp.power(0.82 / (ndims * adaptation_state.EEVPD), 0.25) / jnp.sqrt(steps_per_sample)
        step_size = adaptation_state.step_size

        # Initialize the bisection for finding the step size
        self.epsadap_update = bisection_monotonic_fn(acc_prob_target)
        stepsize_adaptation_state = (jnp.array([-jnp.inf, jnp.inf]), False)

        self.initial_state = AdaptationState(
            steps_per_sample, step_size, stepsize_adaptation_state, 0
        )

    def summary_statistics_fn(self, state, info, rng_key):
        return {
            "acceptance_probability": info.acceptance_rate,
            "equipartition_diagonal": equipartition_diagonal(
                state
            ),  # metric for bias: equipartition theorem gives todo...
            "observables": self.observables(state.position),
            "observables_for_bias": self.observables_for_bias(state.position),
        }

    def update(self, adaptation_state, Etheta):
        acc_prob = Etheta["acceptance_probability"]
        equi_diag = equipartition_diagonal_loss(Etheta["equipartition_diagonal"])
        true_bias = self.contract(Etheta["observables_for_bias"])  # remove

        info_to_be_stored = {
            "L": adaptation_state.step_size * adaptation_state.steps_per_sample,
            "steps_per_sample": adaptation_state.steps_per_sample,
            "step_size": adaptation_state.step_size,
            "acc_prob": acc_prob,
            "equi_diag": equi_diag,
            "bias": true_bias,
            "observables": Etheta["observables"],
        }

        # Bisection to find step size
        stepsize_adaptation_state, step_size = self.epsadap_update(
            adaptation_state.stepsize_adaptation_state,
            adaptation_state.step_size,
            acc_prob,
        )

        return (
            AdaptationState(
                adaptation_state.steps_per_sample,
                step_size,
                stepsize_adaptation_state,
                adaptation_state.iteration + 1,
            ),
            info_to_be_stored,
        )


def bias(model):
    """should be transfered to benchmarks/"""

    def observables(position):
        return jnp.square(model.transform(position))

    def contract(sampler_E_x2):
        bsq = jnp.square(sampler_E_x2 - model.E_x2) / model.Var_x2
        return jnp.array([jnp.max(bsq), jnp.average(bsq)])

    return observables, contract


def while_steps_num(cond):
    if jnp.all(cond):
        return len(cond)
    else:
        return jnp.argmin(cond) + 1



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

num_chains = 128

mesh = jax.sharding.Mesh(devices=jax.devices(),axis_names= "chains")

key_init, key_umclmc, key_mclmc = jax.random.split(jax.random.key(0), 3)

integrator_coefficients = mclachlan_coefficients

acc_prob = None

# initialize the chains
initial_state = umclmc.initialize(
    key_init, logdensity_fn, sample_init, num_chains, mesh
)

diagonal_preconditioning = False
ndims = 2

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

# refine the results with the adjusted method
_acc_prob = acc_prob
if integrator_coefficients is None:
    high_dims = ndims > 200
    _integrator_coefficients = (
        omelyan_coefficients if high_dims else mclachlan_coefficients
    )
    if acc_prob is None:
        _acc_prob = 0.9 if high_dims else 0.7

else:
    _integrator_coefficients = integrator_coefficients
    if acc_prob is None:
        _acc_prob = 0.9

integrator = generate_isokinetic_integrator(_integrator_coefficients)
gradient_calls_per_step = (
    len(_integrator_coefficients) // 2
)  # scheme = BABAB..AB scheme has len(scheme)//2 + 1 Bs. The last doesn't count because that gradient can be reused in the next step.

if diagonal_preconditioning:
    inverse_mass_matrix = jnp.sqrt(final_adaptation_state.inverse_mass_matrix)

    # scale the stepsize so that it reflects averag scale change of the preconditioning
    average_scale_change = jnp.sqrt(jnp.average(inverse_mass_matrix))
    final_adaptation_state = final_adaptation_state._replace(
        step_size=final_adaptation_state.step_size / average_scale_change
    )

else:
    inverse_mass_matrix = 1.0

kernel = build_kernel(
    logdensity_fn, integrator, inverse_mass_matrix=inverse_mass_matrix
)
steps_per_sample = 15
num_steps2 = 100


initial_state = HMCState(
        final_state.position, final_state.logdensity, final_state.logdensity_grad
    )

print(initial_state.position.shape, "bar\n\n")

# pos = jax.random.normal(key_mclmc, shape=(num_chains, ndims))



# print("baz", logdensity_fn(pos))

# initial_state = HMCState(
#     pos, logdensity_fn(pos[0]), jax.grad(logdensity_fn)(pos[0])
# )


num_samples = num_steps2 // (gradient_calls_per_step * steps_per_sample)
num_adaptation_samples = (
    num_samples // 2
)  # number of samples after which the stepsize is fixed.

adap = Adaptation(
    final_adaptation_state,
    num_adaptation_samples,
    steps_per_sample,
    _acc_prob,
)



final_state, final_adaptation_state, info2 = run_eca(
    key_mclmc,
    initial_state,
    kernel,
    adap,
    num_samples,
    num_chains,
    mesh,
    ensemble_observables,
)

