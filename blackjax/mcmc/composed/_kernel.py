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
"""GIST instance (c): composed step size then trajectory length (``h`` then
``L``).

Selects ``h`` via the same reversibility-checked ladder as
:mod:`~blackjax.mcmc.composed.step_size` (run at a frozen trial length,
never the sampled ``L`` -- that would make the selection circular), then
draws ``L`` from the no-U-turn interval built at that ``h``
(:mod:`~blackjax.mcmc.composed.trajectory_length`, reusing its
buffered-rollout caching so the accepted move is a gather, not a
re-integration). The acceptance ratio multiplies three factors, all
evaluated at the forward-selected ``h`` on both endpoints: the h-selection
reversibility indicator, the L-interval width ratio, and the L-interval
membership indicator -- dropping any one breaks stationarity (see this
module's mutation tests). Rejects if either the forward or reverse
step-size search exhausts its budget. ``path_fraction``, ``max_num_steps``,
and ``h_selector_trial_length`` are validated eagerly at construction.

Composition principle: adaptNUTS (arXiv:2408.08259) + the GIST survey
(arXiv:2404.15253); this module claims the instance, not the principle.
"""
from typing import Callable, NamedTuple, cast

import jax
import jax.numpy as jnp

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.composed import _seam as seam
from blackjax.mcmc.composed import step_size, trajectory_length
from blackjax.mcmc.integrators import IntegratorState
from blackjax.types import Array, PRNGKey

__all__ = [
    "ComposedTuningParameter",
    "GISTComposedInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
]

init = seam.init


class ComposedTuningParameter(NamedTuple):
    """The composed GIST tuning parameter ``alpha = (a, b, j, L)``.

    a, b
        Soft acceptance-ratio thresholds for the ``h``-selection ladder,
        freshly drawn ~ Uniform on ``{(a, b) in (0, 1)^2 : a < b}`` every
        transition, exactly as in
        :class:`~blackjax.mcmc.composed.step_size.StepSizeTuningParameter`.
    step_index
        ``j``, the integer log2 step-size index selected by the ladder at
        the frozen trial length ``h_selector_trial_length``:
        ``h = initial_step_size * 2**j``.
    num_integration_steps
        ``L``, the number of leapfrog steps at the selected ``h`` drawn
        uniformly from the psi-jittered no-U-turn interval built at that
        ``h``.
    """

    a: Array
    b: Array
    step_index: Array
    num_integration_steps: Array


class _ComposedAux(NamedTuple):
    """Threaded from ``tuning_parameter_fn`` to ``apply_fn`` (GIST's
    ``aux``): everything computed while drawing ``alpha`` that ``apply_fn``
    would otherwise have to recompute."""

    step_size: Array
    h_search_exhausted_forward: Array
    rollout: "trajectory_length._ForwardRollout"


class _ComposedExtra(NamedTuple):
    """Extra info computed by ``apply_fn``, threaded into
    :class:`GISTComposedInfo` without recomputation."""

    num_integration_steps: Array
    step_size: Array
    reverse_step_index: Array
    h_search_exhausted: Array
    num_steps_to_uturn_forward: Array
    num_steps_to_uturn_reverse: Array
    is_no_return_rejected: Array


class GISTComposedInfo(NamedTuple):
    """Additional information for a ``gist_composed`` transition.

    momentum, tuning_parameter, is_accepted, is_divergent, acceptance_rate,
    energy, num_integration_steps
        Same convention as the sibling instances' ``Info`` types (flat
        extension of :class:`~blackjax.mcmc.composed._seam.GISTInfo`'s
        fields, not nesting).
    step_index, reverse_step_index
        ``j`` (forward-selected) and ``j'`` (re-selected at the proposal,
        the h-selection reversibility check).
    step_size
        The step size ``h = initial_step_size * 2**step_index`` actually
        used to build both the forward and the reverse no-U-turn rollouts.
    h_search_exhausted
        True if the step-size ladder search (forward OR the reversibility
        re-check) hit its iteration budget without terminating. When True,
        the transition was forced to reject regardless of the energy term.
    num_steps_to_uturn_forward, num_steps_to_uturn_reverse
        ``U(theta, rho, h_j)`` and ``U(theta', rho', h_j)`` -- the
        (possibly capped) leapfrog-step counts to the no-return condition,
        forward and reverse, both evaluated at the forward-selected ``h_j``.
    is_no_return_rejected
        True when the drawn ``L`` fell outside the reverse endpoint's
        psi-jittered interval -- the length instance's own "no-return"
        rejection category, tracked separately from an ordinary
        energy-based Metropolis rejection and from the (structurally
        distinct) h-selection reversibility failure.
    """

    momentum: Array
    tuning_parameter: ComposedTuningParameter
    is_accepted: Array
    is_divergent: Array
    acceptance_rate: Array
    energy: float
    num_integration_steps: Array
    step_index: Array
    reverse_step_index: Array
    step_size: Array
    h_search_exhausted: Array
    num_steps_to_uturn_forward: Array
    num_steps_to_uturn_reverse: Array
    is_no_return_rejected: Array


def _tuning_parameter_fn(
    integrator: Callable,
    initial_step_size: float,
    h_selector_trial_length: int,
    max_search_steps: int,
    criterion: str,
    max_num_steps: int,
    path_fraction: float,
) -> Callable:
    selector = step_size.step_size_selector(
        integrator,
        h_selector_trial_length,
        initial_step_size,
        max_search_steps,
        criterion,
    )

    def tuning_parameter_fn(rng_key, state, logdensity_fn, metric):
        # the ladder is deterministic given (a, b), so the reserved subkey
        # is unused; split off anyway so a future randomized criterion
        # inherits independence for free.
        key_ab, _key_ladder_reserved, key_L = jax.random.split(rng_key, 3)
        u = jax.random.uniform(key_ab, shape=(2,))
        a = jnp.minimum(u[0], u[1])
        b = jnp.maximum(u[0], u[1])

        step_index, h_search_exhausted_forward = selector(
            state, a, b, logdensity_fn, metric
        )
        h = initial_step_size * 2.0 ** step_index.astype(jnp.float32)

        rollout_fn = trajectory_length._num_steps_to_uturn_with_rollout(
            integrator, h, metric, max_num_steps
        )
        rollout = rollout_fn(state, logdensity_fn)
        lo, _ = trajectory_length._step_distribution(
            rollout.num_steps_to_uturn, path_fraction
        )
        num_steps = jax.random.randint(
            key_L, shape=(), minval=lo, maxval=rollout.num_steps_to_uturn + 1
        )

        alpha = ComposedTuningParameter(a, b, step_index, num_steps)
        aux = _ComposedAux(h, h_search_exhausted_forward, rollout)
        return alpha, aux

    return tuning_parameter_fn


def _apply_fn(
    integrator: Callable,
    initial_step_size: float,
    h_selector_trial_length: int,
    max_search_steps: int,
    criterion: str,
    max_num_steps: int,
    path_fraction: float,
) -> Callable:
    selector = step_size.step_size_selector(
        integrator,
        h_selector_trial_length,
        initial_step_size,
        max_search_steps,
        criterion,
    )

    def apply_fn(state, alpha, aux, logdensity_fn, metric):
        a, b, step_index, num_steps = alpha
        h, h_search_exhausted_forward, rollout = aux
        forward_uturn = rollout.num_steps_to_uturn

        # gather (not re-integrate) the buffered forward-rollout state at L.
        proposal_state = cast(
            IntegratorState,
            jax.tree.map(
                lambda buf: jax.lax.dynamic_index_in_dim(
                    buf, num_steps - 1, axis=0, keepdims=False
                ),
                rollout.states,
            ),
        )
        proposal_state = hmc.flip_momentum(proposal_state)

        # h reversibility check: same ladder, same (a, b), at the proposal.
        reverse_step_index, h_search_exhausted_reverse = selector(
            proposal_state, a, b, logdensity_fn, metric
        )
        h_search_exhausted = h_search_exhausted_forward | h_search_exhausted_reverse
        is_h_reversible = reverse_step_index == step_index

        # reverse rollout at the SAME forward-selected h (not h0, not h').
        reverse_uturn_fn = trajectory_length.num_steps_to_uturn(
            integrator, h, metric, max_num_steps
        )
        reverse_uturn = reverse_uturn_fn(proposal_state, logdensity_fn)

        _, width_forward = trajectory_length._step_distribution(
            forward_uturn, path_fraction
        )
        lo_reverse, width_reverse = trajectory_length._step_distribution(
            reverse_uturn, path_fraction
        )
        is_in_reverse_interval = (num_steps >= lo_reverse) & (
            num_steps <= reverse_uturn
        )

        is_valid = (
            is_h_reversible
            & is_in_reverse_interval
            & jnp.logical_not(h_search_exhausted)
        )
        log_tuning_density_ratio = jnp.where(
            is_valid,
            jnp.log(width_forward.astype(jnp.float32))
            - jnp.log(width_reverse.astype(jnp.float32)),
            -jnp.inf,
        )
        extra_info = _ComposedExtra(
            num_integration_steps=num_steps,
            step_size=h,
            reverse_step_index=reverse_step_index,
            h_search_exhausted=h_search_exhausted,
            num_steps_to_uturn_forward=forward_uturn,
            num_steps_to_uturn_reverse=reverse_uturn,
            is_no_return_rejected=jnp.logical_not(is_in_reverse_interval),
        )
        return proposal_state, log_tuning_density_ratio, extra_info

    return apply_fn


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
    criterion: str = "symmetric",
    max_search_steps: int = 10,
    h_selector_trial_length: int = 1,
    path_fraction: float = 0.2849,
    max_num_steps: int = 1024,
) -> Callable:
    """Build a ``gist_composed`` kernel: select ``h`` via the step-size
    ladder, then ``L | h`` via the no-U-turn interval built at that ``h``.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian
        dynamics. Must be reversible and volume-preserving for every fixed
        step size.
    divergence_threshold
        Value of the difference in energy above which we consider that the
        transition is divergent.
    criterion
        ``"symmetric"`` (default, proven irreducible and aperiodic) or
        ``"asymmetric"``, see
        :func:`blackjax.mcmc.composed.step_size.step_size_selector`.
    max_search_steps
        Cap on doubling/halving iterations for the step-size ladder (both
        the forward selection and the reversibility-check re-selection).
    h_selector_trial_length
        The step-size ladder's frozen trial trajectory length -- fixed
        independently of the sampled ``L`` to avoid a circular definition.
        Must be >= 1.
    path_fraction
        ``psi`` in ``[0, 1]``, the no-U-turn interval's jitter fraction.
        Defaults to ``0.2849`` (closed form, ``1 - 4.4934 / (2 * pi)``;
        favors second-moment accuracy over first-moment accuracy relative to
        the standalone ``gist_trajectory_length`` instance's ``0.5``
        default).
    max_num_steps
        Hard cap on each no-U-turn rollout (forward and reverse), both built
        at the selected ``h``; also sizes the forward-rollout buffer. Must
        be >= 1.

    Returns
    -------
    A kernel with signature ``kernel(rng_key, state, logdensity_fn,
    initial_step_size, inverse_mass_matrix) -> (GISTState, GISTComposedInfo)``.
    """
    if criterion not in ("symmetric", "asymmetric"):
        raise ValueError(
            "criterion must be 'symmetric' or 'asymmetric', got " f"{criterion!r}"
        )
    if not (0.0 <= path_fraction <= 1.0):
        raise ValueError(f"path_fraction (psi) must lie in [0, 1], got {path_fraction}")
    if max_num_steps < 1:
        raise ValueError(f"max_num_steps must be >= 1, got {max_num_steps}")
    if h_selector_trial_length < 1:
        raise ValueError(
            "h_selector_trial_length (the h-selector's frozen trial length) "
            f"must be >= 1, got {h_selector_trial_length}"
        )
    if max_search_steps < 0:
        raise ValueError(f"max_search_steps must be >= 0, got {max_search_steps}")

    def kernel(
        rng_key: PRNGKey,
        state: seam.GISTState,
        logdensity_fn: Callable,
        initial_step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
    ) -> tuple[seam.GISTState, GISTComposedInfo]:
        """Generate a new sample with the ``gist_composed`` kernel."""
        tuning_parameter_fn = _tuning_parameter_fn(
            integrator,
            initial_step_size,
            h_selector_trial_length,
            max_search_steps,
            criterion,
            max_num_steps,
            path_fraction,
        )
        apply_fn = _apply_fn(
            integrator,
            initial_step_size,
            h_selector_trial_length,
            max_search_steps,
            criterion,
            max_num_steps,
            path_fraction,
        )

        new_state, info, raw_extra_info = seam._step(
            rng_key,
            state,
            logdensity_fn,
            tuning_parameter_fn,
            apply_fn,
            inverse_mass_matrix,
            divergence_threshold,
        )
        tuning_parameter = cast(ComposedTuningParameter, info.tuning_parameter)
        extra_info = cast(_ComposedExtra, raw_extra_info)
        composed_info = GISTComposedInfo(
            info.momentum,
            tuning_parameter,
            info.is_accepted,
            info.is_divergent,
            info.acceptance_rate,
            info.energy,
            info.num_integration_steps,
            tuning_parameter.step_index,
            extra_info.reverse_step_index,
            extra_info.step_size,
            extra_info.h_search_exhausted,
            extra_info.num_steps_to_uturn_forward,
            extra_info.num_steps_to_uturn_reverse,
            extra_info.is_no_return_rejected,
        )
        return new_state, composed_info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    inverse_mass_matrix: metrics.MetricTypes,
    initial_step_size: float,
    *,
    criterion: str = "symmetric",
    max_search_steps: int = 10,
    h_selector_trial_length: int = 1,
    path_fraction: float = 0.2849,
    max_num_steps: int = 1024,
    divergence_threshold: float = 1000,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:
    """``blackjax.gist_composed`` -- GIST composed step-size x trajectory-length
    instance: select ``h`` via the step-size ladder, then ``L | h`` via the
    no-U-turn interval built at that ``h``.

    Examples
    --------

    A new ``gist_composed`` kernel can be initialized and used with the
    following code:

    .. code::

        gist_composed = blackjax.gist_composed(
            logdensity_fn, inverse_mass_matrix, initial_step_size=0.1
        )
        state = gist_composed.init(position)
        new_state, info = gist_composed.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value
        for the momentum and computing the kinetic energy.
    initial_step_size
        ``epsilon_init`` / ``h0``, the fixed base step size the step-size
        ladder's doubling/halving search starts from and reports its
        selection relative to (``h = initial_step_size * 2**j``).
    criterion
        ``"symmetric"`` (default) or ``"asymmetric"``, see
        :func:`blackjax.mcmc.composed.step_size.step_size_selector`.
    max_search_steps
        Cap on doubling/halving iterations for the step-size ladder (both
        the forward selection and the reversibility-check re-selection). On
        exhaustion (forward OR reverse) the transition is forced to reject;
        see ``GISTComposedInfo.h_search_exhausted``.
    h_selector_trial_length
        The step-size ladder's frozen trial trajectory length -- fixed
        independently of the sampled trajectory length ``L`` to avoid a
        circular definition. Default 1 reproduces the MALA-equivalent
        single-leapfrog-step trial the standalone ``gist_step_size``
        instance also defaults to.
    path_fraction
        ``psi`` in ``[0, 1]``. Default ``0.2849``.
    max_num_steps
        Hard cap on each no-U-turn rollout (forward and reverse), both built
        at the selected ``h``. Also sizes the forward-rollout buffer --
        ``O(max_num_steps)`` leapfrog states are allocated per transition.
    divergence_threshold
        The absolute value of the difference in energy between two states
        above which we say that the transition is divergent.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate
        the trajectory. Must be reversible and volume-preserving for every
        fixed step size.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    kernel = build_kernel(
        integrator,
        divergence_threshold,
        criterion,
        max_search_steps,
        h_selector_trial_length,
        path_fraction,
        max_num_steps,
    )
    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        kernel_args=(initial_step_size, inverse_mass_matrix),
    )
