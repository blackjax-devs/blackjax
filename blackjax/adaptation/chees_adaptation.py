"""Public API for ChEES-HMC"""

from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import blackjax.mcmc.dynamic_hmc as dynamic_hmc
import blackjax.optimizers.dual_averaging as dual_averaging
from blackjax.adaptation.base import AdaptationResults, return_all_adapt_info
from blackjax.base import AdaptationAlgorithm
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.util import pytree_size

# optimal tuning for HMC, see https://arxiv.org/abs/1001.4460
OPTIMAL_TARGET_ACCEPTANCE_RATE = 0.651
# Clip the final log-space update like the original implementation in TFP (~log(2)/2 ≈ 0.35).
LOG_UPDATE_CLIP = 0.35
# Small constant to avoid division by zero or log of zero
EPS_FLOAT = 1e-20


class ChEESAdaptationState(NamedTuple):
    """State of the ChEES-HMC adaptation scheme.

    step_size
        Value of the step_size parameter of the HMC algorithm.
    log_step_size_moving_average
        Running moving average of the log step_size parameter.
    trajectory_length
        Value of the num_integration_steps * step_size parameter of
        the HMC algorithm.
    log_trajectory_length_moving_average
        Running moving average of the log num_integration_steps / step_size
        parameter.
    optim_state
        Optax optimizing state for used to maximize the ChEES criterion.
    random_generator_arg
        Utility array for generating a pseudo or quasi-random sequence of
        numbers.
    step
        Current iteration number.

    """

    step_size: float
    log_step_size_moving_average: float
    trajectory_length: float
    log_trajectory_length_moving_average: float
    da_state: dual_averaging.DualAveragingState
    optim_state: optax.OptState
    random_generator_arg: Array
    step: int


def weighted_empirical_mean(x, w):
    # x: (num_chains, dim), w: (num_chains,)
    x_safe = jnp.where(jnp.isfinite(x), x, 0.0)
    w = jnp.where(jnp.isfinite(x).all(axis=-1), w, 0.0)

    w_exp = w.reshape((w.shape[0],) + (1,) * (x.ndim - 1))
    num = jnp.sum(w_exp * x_safe, axis=0)
    den = jnp.sum(w_exp, axis=0) + EPS_FLOAT
    return jax.lax.stop_gradient(num / den)


def base(
    jitter_generator: Callable,
    next_random_arg_fn: Callable,
    optim: optax.GradientTransformation,
    target_acceptance_rate: float,
    decay_rate: float,
    max_leapfrog_steps: int,
) -> Tuple[Callable, Callable]:
    """Maximizing the Change in the Estimator of the Expected Square criterion
    (trajectory length) and dual averaging procedure (step size) for the jittered
    Hamiltonian Monte Carlo kernel :cite:p:`hoffman2021adaptive`.

    This adaptation algorithm tunes the step size and trajectory length, i.e.
    number of integration steps / step size, of the jittered HMC algorithm based
    on statistics collected from a population of many chains. It maximizes the Change
    in the Estimator of the Expected Square (ChEES) criterion to tune the trajectory
    length and uses dual averaging targeting an acceptance rate of 0.651 of the harmonic
    mean of the chain's acceptance probabilities to tune the step size.

    Parameters
    ----------
    jitter_generator
        Optional function that generates a value in [0, 1] used to jitter the trajectory
        lengths given a PRNGKey, used to propose the number of integration steps. If None,
        then a quasi-random Halton is used to jitter the trajectory length.
    next_random_arg_fn
        Function that generates the next `random_generator_arg` from its previous value.
    optim
        Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol.
    target_acceptance_rate
        Average acceptance rate to target with dual averaging.
    decay_rate
        Float representing how much to favor recent iterations over earlier ones in the optimization
        of step size and trajectory length.


    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.

    """

    da_init, da_update, _ = dual_averaging.dual_averaging()

    def compute_parameters(
        proposed_positions: ArrayLikeTree,
        proposed_momentums: ArrayLikeTree,
        initial_positions: ArrayLikeTree,
        acceptance_probabilities: Array,
        is_divergent: Array,
        initial_adaptation_state: ChEESAdaptationState,
    ) -> ChEESAdaptationState:
        """Compute values for the parameters based on statistics collected from
        multiple chains.

        Parameters
        ----------
        proposed_positions:
            A PyTree that contains the position proposed by the HMC algorithm of
            every chain (proposal that is accepted or rejected using MH).
        proposed_momentums:
            A PyTree that contains the momentum variable proposed by the HMC algorithm
            of every chain (proposal that is accepted or rejected using MH).
        initial_positions:
            A PyTree that contains the initial position at the start of the HMC
            algorithm of every chain.
        acceptance_probabilities:
            Metropolis-Hastings acceptance probabilty of proposals of every chain.
        initial_adaptation_state:
            ChEES adaptation step used to generate proposals and acceptance probabilities.

        Returns
        -------
        New values of the step size and trajectory length of the jittered HMC algorithm.

        """
        (
            step_size,
            log_step_size_ma,
            trajectory_length,
            log_trajectory_length_ma,
            da_state,
            optim_state,
            random_generator_arg,
            step,
        ) = initial_adaptation_state

        harmonic_mean = 1.0 / jnp.mean(
            1.0 / acceptance_probabilities, where=~is_divergent
        )
        # Replace inf/nan harmonic mean as zero to avoid issues in dual averaging
        harmonic_mean = jnp.where(jnp.isfinite(harmonic_mean), harmonic_mean, 0.0)
        da_state_ = da_update(da_state, target_acceptance_rate - harmonic_mean)
        step_size_ = jnp.exp(da_state_.log_x)
        new_step_size, new_da_state, new_log_step_size = jax.lax.cond(
            jnp.isfinite(step_size_),
            lambda _: (step_size_, da_state_, da_state_.log_x),
            lambda _: (step_size, da_state, da_state.log_x),
            None,
        )
        update_weight = step ** (-decay_rate)
        new_log_step_size_ma = (
            1.0 - update_weight
        ) * log_step_size_ma + update_weight * new_log_step_size

        w = jnp.where(~is_divergent, acceptance_probabilities, 0.0)
        proposals_mean = jax.tree_util.tree_map(
            lambda p: weighted_empirical_mean(p, w), proposed_positions
        )
        # The above weighted mean is presumably better than the simple mean:
        # proposals_mean = jax.tree_util.tree_map(
        #     lambda p: jnp.nanmean(p, axis=0), proposed_positions
        # )
        initials_mean = jax.tree_util.tree_map(
            lambda p: jnp.nanmean(p, axis=0), initial_positions
        )
        proposals_centered = jax.tree_util.tree_map(
            lambda p, pm: p - pm, proposed_positions, proposals_mean
        )
        initials_centered = jax.tree_util.tree_map(
            lambda p, pm: p - pm, initial_positions, initials_mean
        )

        vmap_flatten_op = jax.vmap(lambda p: jax.flatten_util.ravel_pytree(p)[0])
        proposals_matrix = vmap_flatten_op(proposals_centered)
        initials_matrix = vmap_flatten_op(initials_centered)
        momentums_matrix = vmap_flatten_op(proposed_momentums)

        trajectory_gradients = (
            jitter_generator(random_generator_arg)
            * trajectory_length  # this effectively make this gradient w.r.t. log_trajectory_length
            * jax.vmap(
                lambda pm, im, mm: (jnp.dot(pm, pm) - jnp.dot(im, im)) * jnp.dot(pm, mm)
            )(proposals_matrix, initials_matrix, momentums_matrix)
        )

        trajectory_gradient = jnp.sum(
            acceptance_probabilities * trajectory_gradients,
            where=~is_divergent,
        ) / jnp.sum(acceptance_probabilities + EPS_FLOAT, where=~is_divergent)

        log_trajectory_length = jnp.log(trajectory_length)
        updates, optim_state_ = optim.update(
            trajectory_gradient, optim_state, log_trajectory_length
        )

        updates = jax.tree_util.tree_map(
            lambda u: jnp.clip(u, -LOG_UPDATE_CLIP, LOG_UPDATE_CLIP), updates
        )
        log_trajectory_length_ = optax.apply_updates(log_trajectory_length, updates)
        new_log_trajectory_length, new_optim_state = jax.lax.cond(
            jnp.isfinite(
                jax.flatten_util.ravel_pytree(log_trajectory_length_)[0]
            ).all(),
            lambda _: (log_trajectory_length_, optim_state_),
            lambda _: (log_trajectory_length, optim_state),
            None,
        )
        new_log_trajectory_length_ma = (
            1.0 - update_weight
        ) * log_trajectory_length_ma + update_weight * new_log_trajectory_length
        new_trajectory_length = jnp.exp(new_log_trajectory_length_ma)

        # clip new trajectory length to avoid too large trajectories, also the
        # minimum trajectory length is one integrator step
        new_trajectory_length = jnp.clip(
            new_trajectory_length,
            max=max_leapfrog_steps * new_step_size,
            min=new_step_size,
        )
        return ChEESAdaptationState(
            new_step_size,
            new_log_step_size_ma,
            new_trajectory_length,
            new_log_trajectory_length_ma,
            new_da_state,
            new_optim_state,
            next_random_arg_fn(random_generator_arg),
            step + 1,
        )

    def init(random_generator_arg: Array, step_size: float):
        return ChEESAdaptationState(
            step_size=step_size,
            log_step_size_moving_average=0.0,
            trajectory_length=step_size,
            log_trajectory_length_moving_average=0.0,
            da_state=da_init(step_size),
            optim_state=optim.init(step_size),
            random_generator_arg=random_generator_arg,
            step=1,
        )

    def update(
        adaptation_state: ChEESAdaptationState,
        proposed_positions: ArrayLikeTree,
        proposed_momentums: ArrayLikeTree,
        initial_positions: ArrayLikeTree,
        acceptance_probabilities: Array,
        is_divergent: Array,
    ):
        """Update the adaptation state and parameter values.

        Parameters
        ----------
        adaptation_state
            The current state of the adaptation algorithm
        proposed_positions:
            The position proposed by the HMC algorithm of every chain.
        proposed_momentums:
            The momentum variable proposed by the HMC algorithm of every chain.
        initial_positions:
            The initial position at the start of the HMC algorithm of every chain.
        acceptance_probabilities:
            Metropolis-Hastings acceptance probabilty of proposals of every chain.

        Returns
        -------
        New adaptation state that contains the step size and trajectory length of the
        jittered HMC algorithm.

        """
        new_state = compute_parameters(
            proposed_positions,
            proposed_momentums,
            initial_positions,
            acceptance_probabilities,
            is_divergent,
            adaptation_state,
        )

        return new_state

    return init, update


def chees_adaptation(
    logdensity_fn: Callable,
    num_chains: int,
    *,
    jitter_generator: Optional[Callable] = None,
    jitter_amount: float = 1.0,
    target_acceptance_rate: float = OPTIMAL_TARGET_ACCEPTANCE_RATE,
    decay_rate: float = 0.5,
    max_leapfrog_steps: int = 1000,
    adaptation_info_fn: Callable = return_all_adapt_info,
) -> AdaptationAlgorithm:
    """Adapt the step size and trajectory length (number of integration steps / step size)
    parameters of the jittered HMC algorthm.

    The jittered HMC algorithm depends on the value of a step size, controlling
    the discretization step of the integrator, and a trajectory length, given by the
    number of integration steps / step size, jittered by using only a random percentage
    of this trajectory length.

    This adaptation algorithm tunes the trajectory length by heuristically maximizing
    the Change in the Estimator of the Expected Square (ChEES) criterion over
    an ensamble of parallel chains. At equilibrium, the algorithm aims at eliminating
    correlations between target dimensions, making the HMC algorithm efficient.

    Jittering requires generating a random sequence of uniform variables in [0, 1].
    However, this adds another source of variance to the sampling procedure,
    which may slow adaptation or lead to suboptimal mixing. To alleviate this,
    rather than use uniform random noise to jitter the trajectory lengths, we use a
    quasi-random Halton sequence, which ensures a more even distribution of trajectory
    lengths.

    Examples
    --------

    An HMC adapted kernel can be learned and used with the following code:

    .. code::

        warmup = blackjax.chees_adaptation(logdensity_fn, num_chains)
        key_warmup, key_sample = jax.random.split(rng_key)
        optim = optax.adam(learning_rate)
        (last_states, parameters), _ = warmup.run(
            key_warmup,
            positions, #PyTree where each leaf has shape (num_chains, ...)
            initial_step_size,
            optim,
            num_warmup_steps,
        )
        kernel = blackjax.dynamic_hmc(logdensity_fn, **parameters).step
        new_states, info = jax.vmap(kernel)(key_sample, last_states)

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to sample.
    num_chains
        Number of chains used for cross-chain warm-up training.
    jitter_generator
        Optional function that generates a value in [0, 1] used to jitter the trajectory
        lengths given a PRNGKey, used to propose the number of integration steps. If None,
        then a quasi-random Halton is used to jitter the trajectory length.
    jitter_value
        A percentage in [0, 1] representing how much of the calculated trajectory should be jitted.
    target_acceptance_rate
        Average acceptance rate to target with dual averaging. Defaults to optimal tuning for HMC.
    decay_rate
        Float representing how much to favor recent iterations over earlier ones in the optimization
        of step size and trajectory length. A value of 1 gives equal weight to all history. A value
        of 0 gives weight only to the most recent iteration.
    adaptation_info_fn
        Function to select the adaptation info returned. See return_all_adapt_info
        and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
        information is saved - this can result in excessive memory usage if the
        information is unused.

    Returns
    -------
    A function that returns the last cross-chain state, a sampling kernel with the
    tuned parameter values, and all the warm-up states for diagnostics.

    """

    def run(
        rng_key: PRNGKey,
        positions: ArrayLikeTree,
        step_size: float,
        optim: optax.GradientTransformation,
        num_steps: int = 1000,
        *,
        max_sampling_steps: int = 1000,
    ):
        assert all(
            jax.tree_util.tree_flatten(
                jax.tree_util.tree_map(lambda p: p.shape[0] == num_chains, positions)
            )[0]
        ), "initial `positions` leading dimension must be equal to the `num_chains`"
        num_dim = pytree_size(positions) // num_chains

        next_random_arg_fn = lambda i: i + 1
        init_random_arg = 0

        if jitter_generator is not None:
            rng_key, carry_key = jax.random.split(rng_key)
            jitter_gn = lambda i: jitter_generator(
                jax.random.fold_in(carry_key, i)
            ) * jitter_amount + (1.0 - jitter_amount)
        else:
            max_bits = np.ceil(np.log2(num_steps + max_sampling_steps))
            jitter_gn = lambda i: dynamic_hmc.halton_sequence(
                i, max_bits
            ) * jitter_amount + (1.0 - jitter_amount)

        def integration_steps_fn(random_generator_arg, num_leapfrog_steps):
            return jnp.asarray(
                jnp.ceil(jitter_gn(random_generator_arg) * num_leapfrog_steps),
                dtype=int,
            )

        step_fn = dynamic_hmc.build_kernel(
            next_random_arg_fn=next_random_arg_fn,
            integration_steps_fn=integration_steps_fn,
        )

        init, update = base(
            jitter_gn,
            next_random_arg_fn,
            optim,
            target_acceptance_rate,
            decay_rate,
            max_leapfrog_steps,
        )

        def one_step(carry, rng_key):
            states, adaptation_state = carry

            keys = jax.random.split(rng_key, num_chains)
            _step_fn = partial(
                step_fn,
                logdensity_fn=logdensity_fn,
                step_size=adaptation_state.step_size,
                inverse_mass_matrix=jnp.ones(num_dim),
                num_leapfrog_steps=adaptation_state.trajectory_length
                / adaptation_state.step_size,
            )
            new_states, info = jax.vmap(_step_fn)(keys, states)
            new_adaptation_state = update(
                adaptation_state,
                info.proposal.position,
                info.proposal.momentum,
                states.position,
                info.acceptance_rate,
                info.is_divergent,
            )

            return (new_states, new_adaptation_state), adaptation_info_fn(
                new_states, info, new_adaptation_state
            )

        batch_init = jax.vmap(
            lambda p: dynamic_hmc.init(p, logdensity_fn, init_random_arg)
        )
        init_states = batch_init(positions)
        init_adaptation_state = init(init_random_arg, step_size)

        keys_step = jax.random.split(rng_key, num_steps)
        (last_states, last_adaptation_state), info = jax.lax.scan(
            one_step, (init_states, init_adaptation_state), keys_step
        )

        num_leapfrog_steps = jnp.exp(
            last_adaptation_state.log_trajectory_length_moving_average
            - last_adaptation_state.log_step_size_moving_average
        )
        parameters = {
            "step_size": jnp.exp(last_adaptation_state.log_step_size_moving_average),
            "inverse_mass_matrix": jnp.ones(num_dim),
            "next_random_arg_fn": next_random_arg_fn,
            "integration_steps_fn": lambda arg: integration_steps_fn(
                arg, num_leapfrog_steps
            ),
        }

        return AdaptationResults(last_states, parameters), info

    return AdaptationAlgorithm(run)  # type: ignore[arg-type]
