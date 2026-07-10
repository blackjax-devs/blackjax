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
"""Algorithms to adapt the MCLMC kernel parameters, namely step size and L."""

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.diagnostics import effective_sample_size
from blackjax.util import generate_unit_vector, incremental_value_update, pytree_size

# ---------------------------------------------------------------------------
# Diagnostic warning callbacks — safe to call under JIT via jax.debug.callback.
# Each function must NEVER raise: wrap the body in try/except so that a
# warnings.simplefilter("error") environment does not corrupt the JAX runtime.
# ---------------------------------------------------------------------------


def _cb_frozen_outcome_warning(step_size, position_std, num_steps):
    """Warn when the adapted step size cannot traverse the warmup-scale posterior.

    Triggered on mixing infeasibility (step_size * num_steps << warmup position
    std, or zero position variance from complete divergence).  Scale/budget-
    relative — no magic absolute constant.  Frozen-vs-healthy separation is
    ~6 orders of magnitude, so a 3-order margin is highly robust.
    """
    try:
        ss = float(step_size)
        pos_std = float(position_std)
        n = int(num_steps)
        # Two conditions for "frozen":
        #  (a) pos_std ≈ 0 — all warmup steps diverged, zero position variance.
        #  (b) step_size * num_steps << pos_std — step too small to traverse.
        is_zero_scale = pos_std < 1e-10
        is_step_frozen = (ss * n) < pos_std * 1e-3
        if is_zero_scale or is_step_frozen:
            ss_str = format(ss, ".2g")
            pos_str = format(pos_std, ".2g")
            warnings.warn(
                f"MCLMC warmup: adapted step_size {ss_str} cannot traverse "
                f"the warmup-scale posterior (scale {pos_str}) in {n} "
                "steps — warmup likely collapsed, check initial step size / "
                "model support (see blackjax#973).",
                UserWarning,
                stacklevel=2,
            )
    except Exception:
        pass


def _cb_grad_finiteness_guard(grad_flat):
    """Warn when the init-state gradient is non-finite.

    Value-finiteness at init is insufficient (e.g. lotka-volterra: finite
    logdensity but NaN gradient at the ODE solver's stiff regime).  This guard
    fires once at adaptation start and is nearly free — the gradient was already
    computed by ``mclmc.init``.
    """
    try:
        if not bool(jnp.all(jnp.isfinite(grad_flat))):
            warnings.warn(
                "MCLMC warmup: gradient non-finite at init — this is a "
                "model/solver/support issue, not a step-size issue. "
                "Check model code, solver tolerances, and parameter support "
                "(see blackjax#973).",
                UserWarning,
                stacklevel=2,
            )
    except Exception:
        pass


class MCLMCAdaptationState(NamedTuple):
    """Represents the tunable parameters for MCLMC adaptation.

    L
        The momentum decoherent rate for the MCLMC algorithm.
    step_size
        The step size used for the MCLMC algorithm.
    inverse_mass_matrix
        A matrix used for preconditioning.
    """

    L: float
    step_size: float
    inverse_mass_matrix: float


def mclmc_find_L_and_step_size(
    mclmc_kernel,
    num_steps,
    state,
    rng_key,
    logdensity_fn=None,
    frac_tune1=0.1,
    frac_tune2=0.1,
    frac_tune3=0.1,
    desired_energy_var=5e-4,
    trust_in_estimate=1.5,
    num_effective_samples=150,
    diagonal_preconditioning=True,
    params=None,
    l_factor=0.4,
):
    """
    Finds the optimal value of the parameters for the MCLMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function built by ``mclmc.build_kernel``.  Its call signature
        must be ``kernel(rng_key, state, logdensity_fn, inverse_mass_matrix, L,
        step_size)``, matching the standard BlackJAX kernel pattern.
    num_steps
        The number of MCMC steps that will subsequently be run, after tuning.
    state
        The initial state of the MCMC algorithm.
    rng_key
        The random number generator key.
    logdensity_fn
        The log-density function of the target distribution.
    frac_tune1
        The fraction of tuning for the first step of the adaptation.
    frac_tune2
        The fraction of tuning for the second step of the adaptation.
    frac_tune3
        The fraction of tuning for the third step of the adaptation.
    desired_energy_var
        The desired energy variance for the MCMC algorithm.
    trust_in_estimate
        The trust in the estimate of optimal stepsize.
    num_effective_samples
        The number of effective samples for the MCMC algorithm.
    diagonal_preconditioning
        Whether to do diagonal preconditioning (i.e. a mass matrix)
    params
        Initial params to start tuning from (optional)
    l_factor
        The factor scaling the estimated autocorrelation length to obtain momentum decoherence length L.

    Returns
    -------
    final_state
        The final integrator state after the three tuning phases.
    final_params
        An ``MCLMCAdaptationState`` containing the adapted ``L``,
        ``step_size``, and ``inverse_mass_matrix``.
    total_num_tuning_integrator_steps
        The total number of integrator steps consumed across all three
        tuning phases (frac_tune1 + frac_tune2 + frac_tune3 of
        ``num_steps``).

    Example
    -------
    .. code-block:: python

        kernel = blackjax.mcmc.mclmc.build_kernel(integrator=integrator)

        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
            num_tuning_steps,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=logdensity_fn,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            diagonal_preconditioning=preconditioning,
        )

    Notes
    -----
    **Live divergence monitoring (jax-tap >= 0.3.0)**

    The internal tuning scan exposes a per-step divergence flag as its ``ys``
    output (``True`` = divergence on that step).  Users who install
    ``jax-tap >= 0.3.0`` can observe this stream with no changes to BlackJAX::

        import jaxtap  # pip install "jax-tap>=0.3.0"

        with jaxtap.record(
            select_ys=lambda ys: ys[0],  # the single divergence-flag leaf
            alert_ys=lambda e: "divergence" if e.value else None,
            alert_ys_once=True,  # one stderr line then silence; drop for per-step
        ) as rec:
            state, params, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel, num_steps=N, state=init_state,
                rng_key=key, logdensity_fn=logdensity_fn,
            )
        divergence_steps = [
            e.step for e in rec.events if e.kind == "output" and e.value
        ]
    """
    if logdensity_fn is None:
        raise ValueError(
            "logdensity_fn is required. Pass the log-density function of the "
            "target distribution."
        )

    # Guard: check gradient finiteness at init (nearly free — already computed
    # by mclmc.init; fires once via jax.debug.callback so it works under JIT).
    flat_grad, _ = ravel_pytree(state.logdensity_grad)
    jax.debug.callback(_cb_grad_finiteness_guard, flat_grad)

    dim = pytree_size(state.position)
    if params is None:
        params = MCLMCAdaptationState(
            jnp.sqrt(dim), jnp.sqrt(dim) * 0.25, inverse_mass_matrix=jnp.ones((dim,))
        )

    part1_key, part2_key = jax.random.split(rng_key, 2)
    total_num_tuning_integrator_steps = 0

    num_steps1, num_steps2 = round(num_steps * frac_tune1), round(
        num_steps * frac_tune2
    )
    num_steps2 += diagonal_preconditioning * (num_steps2 // 3)
    num_steps3 = round(num_steps * frac_tune3)

    state, params = make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        logdensity_fn=logdensity_fn,
        dim=dim,
        frac_tune1=frac_tune1,
        frac_tune2=frac_tune2,
        desired_energy_var=desired_energy_var,
        trust_in_estimate=trust_in_estimate,
        num_effective_samples=num_effective_samples,
        diagonal_preconditioning=diagonal_preconditioning,
    )(state, params, num_steps, part1_key)
    total_num_tuning_integrator_steps += num_steps1 + num_steps2

    if num_steps3 >= 2:  # at least 2 samples for ESS estimation
        state, params = make_adaptation_L(
            mclmc_kernel, logdensity_fn, frac=frac_tune3, l_factor=l_factor
        )(state, params, num_steps, part2_key)
        total_num_tuning_integrator_steps += num_steps3

    return state, params, total_num_tuning_integrator_steps


def make_L_step_size_adaptation(
    kernel,
    logdensity_fn,
    dim,
    frac_tune1,
    frac_tune2,
    diagonal_preconditioning,
    desired_energy_var=1e-3,
    trust_in_estimate=1.5,
    num_effective_samples=150,
):
    """Adapts the stepsize and L of the MCLMC kernel. Designed for unadjusted MCLMC"""

    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def predictor(previous_state, params, adaptive_state, rng_key):
        """does one step with the dynamics and updates the prediction for the optimal stepsize
        Designed for the unadjusted MCHMC"""

        time, x_average, step_size_max = adaptive_state

        rng_key, nan_key = jax.random.split(rng_key)

        # dynamics
        next_state, info = kernel(
            rng_key=rng_key,
            state=previous_state,
            logdensity_fn=logdensity_fn,
            inverse_mass_matrix=params.inverse_mass_matrix,
            L=params.L,
            step_size=params.step_size,
        )

        # step updating — thread info so handle_nans can use the kernel's truthful
        # nonans flag (#969) instead of re-deriving from the already-reverted next_state.
        success, state, step_size_max, energy_change = handle_nans(
            previous_state,
            next_state,
            params.step_size,
            step_size_max,
            info.energy_change,
            info.nonans,
            nan_key,
        )

        # The step-size adaptation exploits the scaling relation Var[E] = O(eps^6)
        # for the leapfrog integrator (see Bou-Rabee & Sanz-Serna, 2018).
        # xi measures the energy-variance ratio relative to the target; the
        # exponent 6.0 throughout this block originates from that relation.
        xi = (
            jnp.square(energy_change) / (dim * desired_energy_var)
        ) + 1e-8  # small offset to prevent log(0) divergence
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # Gaussian weight that down-weights step sizes far from the optimum

        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # invert the Var[E] = O(eps^6) relation to obtain the optimal step size
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        params_new = params._replace(step_size=step_size)

        adaptive_state = (time, x_average, step_size_max)

        return state, params_new, adaptive_state, success

    def step(iteration_state, weight_and_key):
        """does one step of the dynamics and updates the estimate of the posterior size and optimal stepsize"""

        mask, rng_key = weight_and_key
        state, params, adaptive_state, streaming_avg = iteration_state

        state, params, adaptive_state, success = predictor(
            state, params, adaptive_state, rng_key
        )

        x = ravel_pytree(state.position)[0]
        # update the running average of x, x^2
        streaming_avg = incremental_value_update(
            expectation=jnp.array([x, jnp.square(x)]),
            incremental_val=streaming_avg,
            weight=mask * success * params.step_size,
        )

        # Enabling seam: per-step divergence flag (True = diverged) is the scan ys.
        # jaxtap y-taps observe it via select_ys=lambda ys: ys[0] — see Notes in
        # mclmc_find_L_and_step_size.
        return (state, params, adaptive_state, streaming_avg), jnp.logical_not(success)

    def run_steps(xs, state, params):
        """Run adaptation steps via scan; return (final_carry, per_step_div_flags)."""
        carry, div_flags = jax.lax.scan(
            step,
            init=(
                state,
                params,
                (0.0, 0.0, jnp.inf),
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=xs,
        )
        return carry, div_flags

    def L_step_size_adaptation(state, params, num_steps, rng_key):
        num_steps1, num_steps2 = round(num_steps * frac_tune1), round(
            num_steps * frac_tune2
        )

        L_step_size_adaptation_keys = jax.random.split(
            rng_key, num_steps1 + num_steps2 + 1
        )
        L_step_size_adaptation_keys, final_key = (
            L_step_size_adaptation_keys[:-1],
            L_step_size_adaptation_keys[-1],
        )

        # we use the last num_steps2 to compute the diagonal preconditioner
        mask = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        # run the steps; ys (per-step divergence flags) available to jaxtap y-taps
        (state, params, _, (_, average)), _ = run_steps(
            xs=(mask, L_step_size_adaptation_keys), state=state, params=params
        )

        L = params.L
        # determine L; track warmup position scale for the frozen-outcome warning
        inverse_mass_matrix = params.inverse_mass_matrix
        warmup_position_std = jnp.zeros(())  # updated below when data is available
        if num_steps2 > 1:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)
            warmup_position_std = jnp.sqrt(jnp.sum(variances))
            L = warmup_position_std

            if diagonal_preconditioning:
                inverse_mass_matrix = variances
                params = params._replace(inverse_mass_matrix=inverse_mass_matrix)
                L = jnp.sqrt(dim)

                # readjust the stepsize
                steps = round(num_steps2 / 3)  # we do some small number of steps
                keys = jax.random.split(final_key, steps)
                (state, params, _, _), _ = run_steps(
                    xs=(jnp.ones(steps), keys), state=state, params=params
                )

        # Emit endpoint diagnostic warnings (safe under JIT via jax.debug.callback)
        jax.debug.callback(
            _cb_frozen_outcome_warning,
            params.step_size,
            warmup_position_std,
            jnp.array(num_steps, dtype=jnp.int32),
        )

        return state, MCLMCAdaptationState(L, params.step_size, inverse_mass_matrix)

    return L_step_size_adaptation


def make_adaptation_L(kernel, logdensity_fn, frac, l_factor):
    """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

    def adaptation_L(state, params, num_steps, key):
        num_steps_3 = round(num_steps * frac)
        adaptation_L_keys = jax.random.split(key, num_steps_3)

        def step(state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                logdensity_fn=logdensity_fn,
                inverse_mass_matrix=params.inverse_mass_matrix,
                L=params.L,
                step_size=params.step_size,
            )

            return next_state, next_state.position

        state, samples = jax.lax.scan(
            f=step,
            init=state,
            xs=adaptation_L_keys,
        )

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        ess = effective_sample_size(flat_samples[None, ...])

        return state, params._replace(
            L=l_factor * params.step_size * jnp.mean(num_steps_3 / ess)
        )

    return adaptation_L


def handle_nans(
    previous_state,
    next_state,
    step_size,
    step_size_max,
    kinetic_change,
    kernel_nonans,
    key,
):
    """Adaptation-level NaN handler.

    If the kernel reported a divergence (via its truthful ``info.nonans`` after
    #969 fix), reduce ``step_size_max`` and return the pre-step state.  The
    kernel's own ``handle_nans`` already sanitises ``next_state`` for both
    divergence signatures:

    * Case-1: NaN position or momentum (position overshoot through a hard boundary).
    * Case-2: finite position + momentum but NaN ``logdensity`` (dominant under
      ``velocity_verlet`` at moderate overshoot on bounded targets).

    Parameters
    ----------
    kernel_nonans
        ``info.nonans`` from the MCLMC kernel — truthful after the #969 fix.

    Returns
    -------
    success
        ``True`` when the step was clean (no divergence and finite energy change).
    """
    reduced_step_size = 0.8  # multiplicative shrinkage applied on NaN recovery

    # Consume the kernel's truthful flag; AND with energy finiteness as a
    # defense-in-depth guard that catches any residual NaN propagation.
    nonans = jnp.logical_and(kernel_nonans, jnp.isfinite(kinetic_change))

    state, step_size, kinetic_change = jax.tree.map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (next_state, step_size_max, kinetic_change),
        (previous_state, step_size * reduced_step_size, 0.0),
    )

    state = jax.lax.cond(
        jnp.isnan(next_state.logdensity),
        lambda: state._replace(
            momentum=generate_unit_vector(key, previous_state.position)
        ),
        lambda: state,
    )

    return nonans, state, step_size, kinetic_change
