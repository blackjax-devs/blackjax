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
"""GIST instance (a): self-tuning step size, autoStep-style.

The tuning parameter is ``alpha = (a, b, j)``: ``(a, b)`` are soft
acceptance-ratio thresholds freshly drawn every transition from the uniform
distribution on the triangle ``Delta = {(a, b) in (0, 1)^2 : a < b}``, and
``j`` is the integer log2 step-size index (``step_size = initial_step_size *
2**j``) selected *deterministically* given ``(theta, rho, a, b)`` by the
doubling/halving selector :func:`step_size_selector` (section 2.1.2).

Because ``p(a, b, j | theta, rho) = Uniform_Delta(a, b) . 1{j = mu(theta, rho,
a, b)}`` is a Dirac measure in its ``j``-argument, the GIST tuning-density
ratio (eq. 9) collapses to an indicator that the *reverse* selection ``j' =
mu(theta', rho', a, b)`` (re-run at the proposal, with the *same* ``(a, b)``)
matches the forward ``j`` -- the "reversibility check" of
[autoMALA]/[AutoStep], derived directly from the GIST framework rather than
bolted on as a separate correctness patch (section 2.1.3).

References
----------
.. [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for locally
   adaptive Hamiltonian Monte Carlo", arXiv:2404.15253, section 2 (mapping
   onto the general kernel).
.. [2] Liu, Surjanovic, Biron-Lattes, Bouchard-Cote, Campbell, "AutoStep:
   Locally adaptive involutive MCMC", arXiv:2410.18929, Algorithm 2 (the
   symmetric selector, default here).
.. [3] Biron-Lattes, Surjanovic, Syed, Campbell, Bouchard-Cote, "autoMALA:
   Locally adaptive Metropolis-adjusted Langevin algorithm",
   arXiv:2310.16782, Algorithm 2 (the asymmetric selector, ``criterion=
   "asymmetric"``, provided for cross-validation against the original paper
   only -- can get stuck near the mode/in the tails).
"""
from typing import Callable, NamedTuple, cast

import jax
import jax.numpy as jnp

import blackjax.mcmc.gist as gist
import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.trajectory as trajectory
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.proposal import safe_energy_diff
from blackjax.types import Array, PRNGKey

__all__ = [
    "GISTStepSizeInfo",
    "StepSizeTuningParameter",
    "init",
    "step_size_selector",
    "build_kernel",
    "as_top_level_api",
]

init = gist.init


class StepSizeTuningParameter(NamedTuple):
    """The GIST tuning parameter ``alpha = (a, b, j)``, section 2.1.1.

    a, b
        Soft acceptance-ratio thresholds, freshly drawn ~ Uniform on
        ``{(a, b) in (0, 1)^2 : a < b}`` every transition (the Gibbs refresh
        of the tuning parameter). The uniform density cancels in the
        acceptance ratio since ``g = identity`` carries ``(a, b)`` through
        the involution unchanged (section 2.1.3) -- any consistent
        (deterministic-given-key) way of drawing them would do, but the
        uniform-on-the-triangle draw is what the paper's own ``p(alpha |
        theta, rho)`` factorization assumes.
    step_index
        ``j``, the integer log2 step-size index:
        ``step_size = initial_step_size * 2**j``.
    """

    a: Array
    b: Array
    step_index: Array


class _StepSizeExtra(NamedTuple):
    """Extra info computed by ``apply_fn``, threaded into
    :class:`GISTStepSizeInfo` without recomputation."""

    num_integration_steps: Array
    reverse_step_index: Array
    search_exhausted: Array
    step_size: Array


class GISTStepSizeInfo(NamedTuple):
    """Additional information for a ``gist_step_size`` transition.

    momentum, tuning_parameter, is_accepted, is_divergent, acceptance_rate,
    energy, num_integration_steps
        Same as :class:`~blackjax.mcmc.gist.GISTInfo` (this instance's Info
        extends it with flat, named fields rather than nesting -- matches
        the ``NUTSInfo``-extends-``HMCInfo``-fields precedent, not literal
        NamedTuple inheritance).
    step_index, reverse_step_index
        ``j`` (selected forward) and ``j'`` (re-selected at the proposal,
        the "reversibility check", section 2.1.3). ``is_accepted`` is False
        whenever ``reverse_step_index != step_index``, in addition to the
        ordinary energy-based rejection path -- both are folded into
        ``is_accepted``; these two fields let you tell them apart.
    search_exhausted
        True if the doubling/halving search (forward OR reverse) hit
        ``max_search_steps`` without the selection criterion terminating.
        When True, the transition was forced to reject regardless of the
        energy term (section 2.1.2).
    step_size
        The step size ``epsilon = initial_step_size * 2**step_index``
        actually used to build the proposal.
    """

    momentum: Array
    tuning_parameter: StepSizeTuningParameter
    is_accepted: Array
    is_divergent: Array
    acceptance_rate: Array
    energy: float
    num_integration_steps: Array
    step_index: Array
    reverse_step_index: Array
    search_exhausted: Array
    step_size: Array


def step_size_selector(
    integrator: Callable,
    num_integration_steps: int,
    initial_step_size: float,
    max_search_steps: int = 10,
    criterion: str = "symmetric",
) -> Callable:
    """Build the ``mu(state, a, b, logdensity_fn, metric) -> (step_index,
    search_exhausted)`` selector, section 2.1.2.

    (``logdensity_fn``/``metric`` extend the paper's ``mu(state, a, b)``
    shorthand: evaluating ``F(alpha)`` fundamentally requires both, they are
    not a free design choice.)

    Parameters
    ----------
    integrator
        Symplectic integrator used for the ``num_integration_steps``-step
        trial trajectories.
    num_integration_steps
        ``L``, the fixed number of leapfrog steps per trial trajectory.
    initial_step_size
        ``epsilon_init``, the fixed base step size the doubling/halving
        search starts from and reports its selection relative to.
    max_search_steps
        Cap on doubling/halving iterations.
    criterion
        ``"symmetric"`` ([AutoStep] Algorithm 2, default -- proven
        irreducible and aperiodic) or ``"asymmetric"`` ([autoMALA]'s
        original criterion, provided for cross-validation against the
        original paper only).

    Returns
    -------
    The selector ``mu``.
    """
    if criterion not in ("symmetric", "asymmetric"):
        raise ValueError(
            "criterion must be 'symmetric' or 'asymmetric', got " f"{criterion!r}"
        )
    is_symmetric = criterion == "symmetric"

    def log_acceptance_ratio(state, step_size, logdensity_fn, metric):
        """``ell(theta, rho, epsilon)`` -- ``-delta_energy`` at one trial
        step size, section 2.1.2. Renamed from the paper's terse ``ell`` to
        a descriptive name, per the naming-conventions checklist.
        """
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        build_trajectory = trajectory.static_integration(symplectic_integrator)
        end_state = build_trajectory(state, step_size, num_integration_steps)
        end_state = hmc.flip_momentum(end_state)
        initial_energy = -state.logdensity + metric.kinetic_energy(state.momentum)
        new_energy = -end_state.logdensity + metric.kinetic_energy(end_state.momentum)
        return safe_energy_diff(initial_energy, new_energy)

    def mu(state: IntegratorState, a, b, logdensity_fn, metric):
        log_a = jnp.log(a)
        log_b = jnp.log(b)
        ell0 = log_acceptance_ratio(state, initial_step_size, logdensity_fn, metric)

        if is_symmetric:
            do_expand = jnp.abs(ell0) < jnp.abs(log_b)  # too small a step
            do_shrink = jnp.abs(ell0) > jnp.abs(log_a)  # too large a step
        else:
            do_expand = ell0 >= log_b
            do_shrink = ell0 <= log_a
        v = jnp.where(do_expand, 1, jnp.where(do_shrink, -1, 0)).astype(jnp.int32)

        def cond_fn(carry):
            _, n, terminated = carry
            return jnp.logical_not(terminated) & (n < max_search_steps)

        def body_fn(carry):
            j, n, _ = carry
            j_next = j + v
            step_size = initial_step_size * 2.0 ** j_next.astype(jnp.float32)
            ell = log_acceptance_ratio(state, step_size, logdensity_fn, metric)
            if is_symmetric:
                term_expand = (v == 1) & (jnp.abs(ell) >= jnp.abs(log_b))
                term_shrink = (v == -1) & (jnp.abs(ell) <= jnp.abs(log_a))
            else:
                term_expand = (v == 1) & (ell < log_b)
                term_shrink = (v == -1) & (ell > log_a)
            return j_next, n + 1, term_expand | term_shrink

        init_carry = (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            v == 0,
        )
        j_final, _, terminated_final = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        search_exhausted = jnp.logical_not(terminated_final) & (v != 0)
        # "Final halving": on a successful *expansion*, report one step back
        # (section 2.1.2) -- necessary for the reversibility check to ever
        # pass in the doubling sub-case ([autoMALA] p.4).
        step_index = jnp.where(terminated_final & (v == 1), j_final - 1, j_final)
        return step_index, search_exhausted

    return mu


def _tuning_parameter_fn(selector: Callable) -> Callable:
    def tuning_parameter_fn(rng_key, state, logdensity_fn, metric):
        u = jax.random.uniform(rng_key, shape=(2,))
        a = jnp.minimum(u[0], u[1])
        b = jnp.maximum(u[0], u[1])
        step_index, search_exhausted = selector(state, a, b, logdensity_fn, metric)
        alpha = StepSizeTuningParameter(a, b, step_index)
        return alpha, search_exhausted

    return tuning_parameter_fn


def _apply_fn(
    integrator: Callable,
    num_integration_steps: int,
    initial_step_size: float,
    selector: Callable,
) -> Callable:
    def apply_fn(state, alpha, aux, logdensity_fn, metric):
        a, b, step_index = alpha
        search_exhausted_forward = aux
        step_size = initial_step_size * 2.0 ** step_index.astype(jnp.float32)

        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        build_trajectory = trajectory.static_integration(symplectic_integrator)
        proposal_state = build_trajectory(state, step_size, num_integration_steps)
        proposal_state = hmc.flip_momentum(proposal_state)

        reverse_step_index, search_exhausted_reverse = selector(
            proposal_state, a, b, logdensity_fn, metric
        )
        search_exhausted = search_exhausted_forward | search_exhausted_reverse
        is_reversible = reverse_step_index == step_index
        log_tuning_density_ratio = jnp.where(
            is_reversible & jnp.logical_not(search_exhausted), 0.0, -jnp.inf
        )
        extra_info = _StepSizeExtra(
            num_integration_steps=jnp.asarray(num_integration_steps),
            reverse_step_index=reverse_step_index,
            search_exhausted=search_exhausted,
            step_size=step_size,
        )
        return proposal_state, log_tuning_density_ratio, extra_info

    return apply_fn


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
    criterion: str = "symmetric",
    max_search_steps: int = 10,
) -> Callable:
    """Build a ``gist_step_size`` kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian
        dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the
        transition is divergent.
    criterion
        ``"symmetric"`` (default) or ``"asymmetric"``, see
        :func:`step_size_selector`.
    max_search_steps
        Cap on doubling/halving iterations (both the forward selection and
        the reversibility-check re-selection).

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current
    state of the chain and that returns a new state of the chain along with
    information about the transition.
    """
    if criterion not in ("symmetric", "asymmetric"):
        raise ValueError(
            "criterion must be 'symmetric' or 'asymmetric', got " f"{criterion!r}"
        )
    gist_step = gist._step

    def kernel(
        rng_key: PRNGKey,
        state: gist.GISTState,
        logdensity_fn: Callable,
        initial_step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
        num_integration_steps: int = 1,
    ) -> tuple[gist.GISTState, GISTStepSizeInfo]:
        """Generate a new sample with the ``gist_step_size`` kernel."""
        selector = step_size_selector(
            integrator,
            num_integration_steps,
            initial_step_size,
            max_search_steps,
            criterion,
        )
        tuning_parameter_fn = _tuning_parameter_fn(selector)
        apply_fn = _apply_fn(
            integrator, num_integration_steps, initial_step_size, selector
        )

        new_state, info, raw_extra_info = gist_step(
            rng_key,
            state,
            logdensity_fn,
            tuning_parameter_fn,
            apply_fn,
            inverse_mass_matrix,
            divergence_threshold,
        )
        # `info.tuning_parameter`/`raw_extra_info` are declared as the generic
        # `ArrayTree` in the general spine (opaque to it by design); cast back
        # to the concrete instance types this kernel actually produces.
        tuning_parameter = cast(StepSizeTuningParameter, info.tuning_parameter)
        extra_info = cast(_StepSizeExtra, raw_extra_info)
        step_size_info = GISTStepSizeInfo(
            info.momentum,
            tuning_parameter,
            info.is_accepted,
            info.is_divergent,
            info.acceptance_rate,
            info.energy,
            info.num_integration_steps,
            tuning_parameter.step_index,
            extra_info.reverse_step_index,
            extra_info.search_exhausted,
            extra_info.step_size,
        )
        return new_state, step_size_info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    inverse_mass_matrix: metrics.MetricTypes,
    initial_step_size: float,
    num_integration_steps: int = 1,
    *,
    criterion: str = "symmetric",
    max_search_steps: int = 10,
    divergence_threshold: float = 1000,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:
    """``blackjax.gist_step_size`` -- GIST self-tuning step size (autoStep-style).

    Examples
    --------

    A new ``gist_step_size`` kernel can be initialized and used with the
    following code:

    .. code::

        gist_step_size = blackjax.gist_step_size(
            logdensity_fn, inverse_mass_matrix, initial_step_size=0.1
        )
        state = gist_step_size.init(position)
        new_state, info = gist_step_size.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value
        for the momentum and computing the kinetic energy.
    initial_step_size
        ``epsilon_init``, section 2.1.1; the fixed base step size the
        doubling/halving search starts from and reports its selection
        relative to (``epsilon = initial_step_size * 2**j``). NOT re-tuned
        by this kernel -- round-based re-tuning of ``initial_step_size``
        ([AutoStep]/[autoMALA] Algorithm 3) is out of scope; it belongs in
        ``blackjax/adaptation/``.
    num_integration_steps
        ``L``, the (fixed) number of leapfrog steps per proposal at the
        selected step size. Default 1 reproduces the MALA-equivalent
        single-leapfrog-step case both source papers analyze.
    criterion
        ``"symmetric"`` (default, [AutoStep] Algorithm 2 -- proven
        irreducible and aperiodic) or ``"asymmetric"`` ([autoMALA]'s
        original criterion -- provided for cross-validation against the
        original paper only; can get stuck near the mode/in the tails, see
        the module docstring).
    max_search_steps
        Cap on doubling/halving iterations (both the forward selection and
        the reversibility-check re-selection). On exhaustion the transition
        is forced to reject; see ``GISTStepSizeInfo.search_exhausted``.
    divergence_threshold
        The absolute value of the difference in energy between two states
        above which we say that the transition is divergent.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate
        the trajectory.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel(integrator, divergence_threshold, criterion, max_search_steps)
    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        kernel_args=(initial_step_size, inverse_mass_matrix, num_integration_steps),
    )
