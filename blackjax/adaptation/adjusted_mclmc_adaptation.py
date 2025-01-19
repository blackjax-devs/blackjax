import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.diagnostics import effective_sample_size
from blackjax.util import incremental_value_update, pytree_size

Lratio_lowerbound = 0.0
Lratio_upperbound = 2.0


def adjusted_mclmc_find_L_and_step_size(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    target,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.0,
    diagonal_preconditioning=True,
    params=None,
    max="avg",
    num_windows=1,
    tuning_factor=1.3,
):
    """
    Finds the optimal value of the parameters for the MH-MCHMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function used for the MCMC algorithm.
    num_steps
        The number of MCMC steps that will subsequently be run, after tuning.
    state
        The initial state of the MCMC algorithm.
    rng_key
        The random number generator key.
    target
        The target acceptance rate for the step size adaptation.
    frac_tune1
        The fraction of tuning for the first step of the adaptation.
    frac_tune2
        The fraction of tuning for the second step of the adaptation.
    frac_tune3
        The fraction of tuning for the third step of the adaptation.
    diagonal_preconditioning
        Whether to do diagonal preconditioning (i.e. a mass matrix)
    params
        Initial params to start tuning from (optional)
    max
        whether to calculate L from maximum or average eigenvalue. Average is advised.
    num_windows
        how many iterations of the tuning are carried out
    tuning_factor
        multiplicative factor for L


    Returns
    -------
    A tuple containing the final state of the MCMC algorithm and the final hyperparameters.
    """

    frac_tune1 /= num_windows
    frac_tune2 /= num_windows
    frac_tune3 /= num_windows

    dim = pytree_size(state.position)
    if params is None:
        params = MCLMCAdaptationState(
            jnp.sqrt(dim), jnp.sqrt(dim) * 0.2, inverse_mass_matrix=jnp.ones((dim,))
        )

    part1_key, part2_key = jax.random.split(rng_key, 2)

    total_num_tuning_integrator_steps = 0
    for i in range(num_windows):
        window_key = jax.random.fold_in(part1_key, i)
        (
            state,
            params,
            eigenvector,
            num_tuning_integrator_steps,
        ) = adjusted_mclmc_make_L_step_size_adaptation(
            kernel=mclmc_kernel,
            dim=dim,
            frac_tune1=frac_tune1,
            frac_tune2=frac_tune2,
            target=target,
            diagonal_preconditioning=diagonal_preconditioning,
            max=max,
            tuning_factor=tuning_factor,
        )(
            state, params, num_steps, window_key
        )
        total_num_tuning_integrator_steps += num_tuning_integrator_steps

    if frac_tune3 != 0:
        for i in range(num_windows):
            part2_key = jax.random.fold_in(part2_key, i)
            part2_key1, part2_key2 = jax.random.split(part2_key, 2)

            (
                state,
                params,
                num_tuning_integrator_steps,
            ) = adjusted_mclmc_make_adaptation_L(
                mclmc_kernel,
                frac=frac_tune3,
                Lfactor=0.5,
                max=max,
                eigenvector=eigenvector,
            )(
                state, params, num_steps, part2_key1
            )

            total_num_tuning_integrator_steps += num_tuning_integrator_steps

            (
                state,
                params,
                _,
                num_tuning_integrator_steps,
            ) = adjusted_mclmc_make_L_step_size_adaptation(
                kernel=mclmc_kernel,
                dim=dim,
                frac_tune1=frac_tune1,
                frac_tune2=0,
                target=target,
                fix_L_first_da=True,
                diagonal_preconditioning=diagonal_preconditioning,
                max=max,
                tuning_factor=tuning_factor,
            )(
                state, params, num_steps, part2_key2
            )

            total_num_tuning_integrator_steps += num_tuning_integrator_steps

    return state, params, total_num_tuning_integrator_steps


def adjusted_mclmc_make_L_step_size_adaptation(
    kernel,
    dim,
    frac_tune1,
    frac_tune2,
    target,
    diagonal_preconditioning,
    fix_L_first_da=False,
    max="avg",
    tuning_factor=1.0,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for adjusted MCLMC"""

    def dual_avg_step(fix_L, update_da):
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""

        def step(iteration_state, weight_and_key):
            mask, rng_key = weight_and_key
            (
                previous_state,
                params,
                (adaptive_state, step_size_max),
                previous_weight_and_average,
            ) = iteration_state

            avg_num_integration_steps = params.L / params.step_size

            state, info = kernel(
                rng_key=rng_key,
                state=previous_state,
                avg_num_integration_steps=avg_num_integration_steps,
                step_size=params.step_size,
                inverse_mass_matrix=params.inverse_mass_matrix,
            )

            # step updating
            success, state, step_size_max, energy_change = handle_nans(
                previous_state,
                state,
                params.step_size,
                step_size_max,
                info.energy,
            )

            with_mask = lambda x, y: mask * x + (1 - mask) * y

            log_step_size, log_step_size_avg, step, avg_error, mu = update_da(
                adaptive_state, info.acceptance_rate
            )

            adaptive_state = DualAveragingAdaptationState(
                with_mask(log_step_size, adaptive_state.log_step_size),
                with_mask(log_step_size_avg, adaptive_state.log_step_size_avg),
                with_mask(step, adaptive_state.step),
                with_mask(avg_error, adaptive_state.avg_error),
                with_mask(mu, adaptive_state.mu),
            )

            step_size = jax.lax.clamp(
                1e-5, jnp.exp(adaptive_state.log_step_size), params.L / 1.1
            )
            adaptive_state = adaptive_state._replace(log_step_size=jnp.log(step_size))

            x = ravel_pytree(state.position)[0]

            # update the running average of x, x^2
            previous_weight_and_average = incremental_value_update(
                expectation=jnp.array([x, jnp.square(x)]),
                incremental_val=previous_weight_and_average,
                weight=(1 - mask) * success * step_size,
                zero_prevention=mask,
            )

            params = params._replace(step_size=with_mask(step_size, params.step_size))
            if not fix_L:
                params = params._replace(
                    L=with_mask(params.L * (step_size / params.step_size), params.L),
                )

            state_position = state.position

            return (
                state,
                params,
                (adaptive_state, step_size_max),
                previous_weight_and_average,
            ), (
                info,
                state_position,
            )

        return step

    def step_size_adaptation(mask, state, params, keys, fix_L, initial_da, update_da):
        return jax.lax.scan(
            dual_avg_step(fix_L, update_da),
            init=(
                state,
                params,
                (initial_da(params.step_size), jnp.inf),  # step size max
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=(mask, keys),
        )

    def L_step_size_adaptation(state, params, num_steps, rng_key):
        num_steps1, num_steps2 = int(num_steps * frac_tune1), int(
            num_steps * frac_tune2
        )

        check_key, rng_key = jax.random.split(rng_key, 2)

        rng_key_pass1, rng_key_pass2 = jax.random.split(rng_key, 2)
        L_step_size_adaptation_keys_pass1 = jax.random.split(
            rng_key_pass1, num_steps1 + num_steps2
        )
        L_step_size_adaptation_keys_pass2 = jax.random.split(rng_key_pass2, num_steps1)

        # determine which steps to ignore in the streaming average
        mask = 1 - jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        initial_da, update_da, final_da = dual_averaging_adaptation(target=target)

        (
            (state, params, (dual_avg_state, step_size_max), (_, average)),
            (info, position_samples),
        ) = step_size_adaptation(
            mask,
            state,
            params,
            L_step_size_adaptation_keys_pass1,
            fix_L=fix_L_first_da,
            initial_da=initial_da,
            update_da=update_da,
        )

        num_tuning_integrator_steps = info.num_integration_steps.sum()
        final_stepsize = final_da(dual_avg_state)
        params = params._replace(step_size=final_stepsize)

        # determine L
        eigenvector = None
        if num_steps2 != 0.0:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)

            if max == "max":
                contract = lambda x: jnp.sqrt(jnp.max(x) * dim) * tuning_factor

            elif max == "avg":
                contract = lambda x: jnp.sqrt(jnp.sum(x)) * tuning_factor

            else:
                raise ValueError("max should be either 'max' or 'avg'")

            change = jax.lax.clamp(
                Lratio_lowerbound,
                contract(variances) / params.L,
                Lratio_upperbound,
            )
            params = params._replace(
                L=params.L * change, step_size=params.step_size * change
            )
            if diagonal_preconditioning:
                params = params._replace(inverse_mass_matrix=variances, L=jnp.sqrt(dim))

            initial_da, update_da, final_da = dual_averaging_adaptation(target=target)
            (
                (state, params, (dual_avg_state, step_size_max), (_, average)),
                (info, params_history),
            ) = step_size_adaptation(
                jnp.ones(num_steps1),
                state,
                params,
                L_step_size_adaptation_keys_pass2,
                fix_L=True,
                update_da=update_da,
                initial_da=initial_da,
            )

            num_tuning_integrator_steps += info.num_integration_steps.sum()

            params = params._replace(step_size=final_da(dual_avg_state))

        return state, params, eigenvector, num_tuning_integrator_steps

    return L_step_size_adaptation


def adjusted_mclmc_make_adaptation_L(
    kernel, frac, Lfactor, max="avg", eigenvector=None
):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps = int(num_steps * frac)
        adaptation_L_keys = jax.random.split(key, num_steps)

        def step(state, key):
            next_state, info = kernel(
                rng_key=key,
                state=state,
                step_size=params.step_size,
                avg_num_integration_steps=params.L / params.step_size,
                inverse_mass_matrix=params.inverse_mass_matrix,
            )
            return next_state, (next_state.position, info)

        state, (samples, info) = jax.lax.scan(
            f=step,
            init=state,
            xs=adaptation_L_keys,
        )

        if max == "max":
            contract = jnp.min
        else:
            contract = jnp.mean

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)

        if eigenvector is not None:
            flat_samples = jnp.expand_dims(
                jnp.einsum("ij,j", flat_samples, eigenvector), 1
            )

        # number of effective samples per 1 actual sample
        ess = contract(effective_sample_size(flat_samples[None, ...])) / num_steps

        return (
            state,
            params._replace(
                L=jnp.clip(
                    Lfactor * params.L / jnp.mean(ess), max=params.L * Lratio_upperbound
                )
            ),
            info.num_integration_steps.sum(),
        )

    return adaptation_L


def handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change):
    """if there are nans, let's reduce the stepsize, and not update the state. The
    function returns the old state in this case."""

    reduced_step_size = 0.8
    p, unravel_fn = ravel_pytree(next_state.position)
    nonans = jnp.all(jnp.isfinite(p))
    state, step_size, kinetic_change = jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (next_state, step_size_max, kinetic_change),
        (previous_state, step_size * reduced_step_size, 0.0),
    )

    return nonans, state, step_size, kinetic_change
