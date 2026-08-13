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
"""GIST instance (b): self-tuning trajectory length (no-U-turn, GIST paper
section 5) -- NOT NUTS's recursive doubling.

The tuning parameter is ``alpha = L``, the number of leapfrog steps at a
*fixed* step size ``h``. ``L`` is drawn uniformly from ``[Lo(theta, rho) :
U(theta, rho)]`` where ``U(theta, rho)`` is the number of leapfrog steps
until the no-U-turn condition first fires (a single forward ``while_loop``
rollout, materially simpler than NUTS's recursive-doubling tree, section
2.2.2) and ``Lo`` shifts the lower end of that range by a fixed path
fraction ``psi``. Because ``g = identity`` on ``L``, the reversibility of the
resulting kernel is automatic (Corollary 4); the only thing the acceptance
ratio has to account for is that the forward and reverse draws are uniform
over *different-width* intervals containing the same ``L`` (section 2.2.4).

Forward-rollout caching
------------------------
The forward ``U(theta, rho)`` search already integrates every state on
``[1 : U]`` one leapfrog step at a time (the ``while_loop`` in
:func:`num_steps_to_uturn`); the selected proposal is always one of those
states (some ``L <= U``). Rather than throw them away and re-integrate ``L``
steps from scratch via ``trajectory.static_integration`` (the original
implementation), :func:`_tuning_parameter_fn` buffers every rolled-out state
as it goes (:func:`_num_steps_to_uturn_with_rollout`) and threads the buffer
to :func:`_apply_fn` as GIST's ``aux``, so building the proposal is an
``O(1)`` gather rather than an ``O(L)`` re-integration. Per transition this
drops the cost from roughly ``U + L + U_rev`` leapfrog/gradient evaluations
(``~2.75 U`` empirically, averaging over ``L``'s distribution) to roughly
``U + U_rev`` (``~1.25 U``) -- see Issue#1058. The reverse rollout
``U_rev(theta', rho')`` (section 2.2.4) has nothing to gather from (its own
states are never selected as the proposal) and is left as an ordinary,
unbuffered :func:`num_steps_to_uturn` call.

The buffer is ``max_num_steps`` copies of the full leapfrog state
(position, momentum, logdensity, logdensity_grad) -- the same asymptotic
memory scale as the states NUTS's own trajectory-doubling machinery holds
live during tree building (:mod:`blackjax.mcmc.trajectory`), and it is
allocated fresh on every kernel call (not carried across transitions), so
steady-state memory is ``O(max_num_steps * dim)`` on top of the chain state
itself, not ``O(num_samples * max_num_steps * dim)``. This makes
``max_num_steps`` (default 1024, mirroring NUTS's ``max_num_doublings=10``
cap of 1023 steps) a *memory* knob as well as a compute-cap knob: sizing it
much larger than the rollout lengths a target actually needs pays for unused
buffer capacity on every transition, so prefer capping it near the largest
``U`` the warmup/pilot run actually observes rather than leaving the default
untouched on very high-dimensional targets.

References
----------
.. [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for locally
   adaptive Hamiltonian Monte Carlo", arXiv:2404.15253, section 5 (p.20-25),
   Algorithm 2 (p.21), eq. 33 (the no-U-turn condition), eq. 34-35 (the step
   distributions).
"""
from typing import Callable, NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.composed import _seam as seam
from blackjax.mcmc.integrators import IntegratorState
from blackjax.types import Array, PRNGKey

__all__ = [
    "GISTTrajectoryLengthInfo",
    "init",
    "num_steps_to_uturn",
    "build_kernel",
    "as_top_level_api",
]

init = seam.init


class _TrajectoryLengthExtra(NamedTuple):
    """Extra info computed by ``apply_fn``, threaded into
    :class:`GISTTrajectoryLengthInfo` without recomputation."""

    num_integration_steps: Array
    num_steps_to_uturn_forward: Array
    num_steps_to_uturn_reverse: Array
    is_no_return_rejected: Array


class GISTTrajectoryLengthInfo(NamedTuple):
    """Additional information for a ``gist_trajectory_length`` transition.

    momentum, tuning_parameter, is_accepted, is_divergent, acceptance_rate,
    energy, num_integration_steps
        Same convention as :class:`~blackjax.mcmc.composed.step_size.GISTStepSizeInfo`
        (flat extension, not nesting). ``tuning_parameter`` is ``L`` itself
        (an integer, not a compound PyTree, since ``g = identity`` is the
        only tuning-parameter component here).
    num_steps_to_uturn_forward, num_steps_to_uturn_reverse
        ``U(theta, rho) = M`` and ``U(theta', rho') = N``, section 2.2.2 --
        the (possibly capped) leapfrog-step counts to the no-return
        condition, forward from the current state and from the proposal.
    is_no_return_rejected
        True when ``L`` fell outside ``[Lo(theta', rho'), N]`` -- the
        paper's own "no-return" rejection category (Fig. 5, section 2.2.4),
        tracked separately from an ordinary energy-based Metropolis
        rejection.
    """

    momentum: Array
    tuning_parameter: Array
    is_accepted: Array
    is_divergent: Array
    acceptance_rate: Array
    energy: float
    num_integration_steps: Array
    num_steps_to_uturn_forward: Array
    num_steps_to_uturn_reverse: Array
    is_no_return_rejected: Array


def num_steps_to_uturn(
    integrator: Callable,
    step_size: float,
    metric: metrics.Metric,
    max_num_steps: int,
) -> Callable:
    """Build the ``U(theta, rho)`` forward-rollout function, section 2.2.2.

    A single forward ``while_loop``, one leapfrog step at a time, checking
    the sign of the (metric-corrected) no-return condition after every step
    -- no tree/doubling data structure, no recursion, no sub-U-turn
    bookkeeping, unlike NUTS's ``1sub-U-turn`` criterion (section 3.5).

    The dot product pairs the position displacement with the **raw momentum**
    ``rho`` (GIST paper eq. 33), so the criterion generalizes correctly to
    a non-identity ``inverse_mass_matrix`` (diagonal / dense / low-rank) in
    an affine-equivariant manner: with p ~ N(0, G^-1) and K(p) = ½ pᵀGp,
    whitening φ = G^{-1/2}θ and p_φ = G^{1/2}p gives (Δθ)ᵀp = Δφᵀp_φ,
    which is exactly invariant under any change of basis G. This is a
    *different* criterion from NUTS's momentum-sum ``check_turning``, which
    uses the momentum sum (not position displacement) in the pairing. This
    changes no formula for the identity-metric case the paper's own
    experiments use.

    ``max_num_steps`` is a hard cap, exactly analogous to NUTS's
    ``max_num_doublings`` (except here it bounds a *linear* rollout, not
    ``2**max`` leapfrog steps). Capping is not an approximation: as long as
    ``p(L|theta,rho)`` and the acceptance ratio consistently use this capped
    ``U``, the resulting ``p(L|theta,rho)`` is still an exact, well-defined,
    strictly-positive conditional density (the GIST reversibility guarantee,
    Theorem 3, does not care that ``U`` is a capped version of the "true"
    U-turn step count).

    Parameters
    ----------
    integrator
        Symplectic integrator used for the one-leapfrog-at-a-time rollout.
    step_size
        ``h``, the fixed step size (not GIST-adapted in this instance).
    metric
        The (already-resolved) :class:`~blackjax.mcmc.metrics.Metric`.
    max_num_steps
        Hard cap on the rollout length.

    Returns
    -------
    ``uturn_fn(state: IntegratorState, logdensity_fn) -> Array``, the
    (possibly capped) number of leapfrog steps to the no-return condition.
    """

    def uturn_fn(state: IntegratorState, logdensity_fn: Callable) -> Array:
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        theta0, _ = ravel_pytree(state.position)

        def cond_fn(carry):
            n, _, no_return = carry
            return jnp.logical_not(no_return) & (n < max_num_steps)

        def body_fn(carry):
            n, current, _ = carry
            nxt = symplectic_integrator(current, step_size)
            delta = ravel_pytree(nxt.position)[0] - theta0
            momentum, _ = ravel_pytree(nxt.momentum)
            no_return = jnp.dot(delta, momentum) < 0.0
            return n + 1, nxt, no_return

        n_final, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (jnp.asarray(0), state, jnp.asarray(False))
        )
        return n_final

    return uturn_fn


class _ForwardRollout(NamedTuple):
    """Cached forward rollout, threaded from :func:`_tuning_parameter_fn` to
    :func:`_apply_fn` as GIST's ``aux`` (Issue#1058).

    num_steps_to_uturn
        ``U(theta, rho)`` -- the same value :func:`num_steps_to_uturn` would
        return for this state.
    states
        ``IntegratorState`` pytree with an added leading ``max_num_steps``
        axis. ``states[k]`` is the state after ``k + 1`` leapfrog steps from
        the initial state, written for every ``k < num_steps_to_uturn``.
        Entries at or beyond ``num_steps_to_uturn`` are never written by the
        rollout (left at their zero fill) and must not be read -- every
        ``alpha`` this module's own ``tuning_parameter_fn`` ever draws
        satisfies ``alpha <= num_steps_to_uturn`` by construction (see
        ``_step_distribution``), so this precondition always holds for
        proposals built by the shipped kernel.
    """

    num_steps_to_uturn: Array
    states: IntegratorState


def _num_steps_to_uturn_with_rollout(
    integrator: Callable,
    step_size: float,
    metric: metrics.Metric,
    max_num_steps: int,
) -> Callable:
    """Forward-only variant of :func:`num_steps_to_uturn` that additionally
    buffers every rolled-out state (Issue#1058), so :func:`_apply_fn` can
    gather the selected-``L`` proposal instead of re-integrating it.

    Only used for the forward rollout, in :func:`_tuning_parameter_fn`: the
    reverse rollout in :func:`_apply_fn` has no selected state to gather
    (its own count is the only thing used) and keeps calling the plain,
    unbuffered :func:`num_steps_to_uturn` directly -- buffering it would
    only add memory traffic for states that are never read.

    Same control flow, same floating-point operations, same iteration order
    as :func:`num_steps_to_uturn` (the buffer writes are a side effect that
    does not influence ``no_return`` or the final count), so the returned
    ``num_steps_to_uturn`` is bit-identical to calling the unbuffered
    version on the same state, and ``states[k]`` is bit-identical to what
    ``trajectory.static_integration`` would produce after ``k + 1`` steps
    from the same initial state at the same step size.
    """

    def uturn_fn(state: IntegratorState, logdensity_fn: Callable) -> _ForwardRollout:
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        theta0, _ = ravel_pytree(state.position)
        rollout_buffer = jax.tree.map(
            lambda leaf: jnp.zeros(
                (max_num_steps,) + jnp.shape(leaf), dtype=leaf.dtype
            ),
            state,
        )

        def cond_fn(carry):
            n, _, no_return, _ = carry
            return jnp.logical_not(no_return) & (n < max_num_steps)

        def body_fn(carry):
            n, current, _, buffer = carry
            nxt = symplectic_integrator(current, step_size)
            delta = ravel_pytree(nxt.position)[0] - theta0
            momentum, _ = ravel_pytree(nxt.momentum)
            no_return = jnp.dot(delta, momentum) < 0.0
            buffer = jax.tree.map(
                lambda buf, leaf: jax.lax.dynamic_update_index_in_dim(
                    buf, leaf, n, axis=0
                ),
                buffer,
                nxt,
            )
            return n + 1, nxt, no_return, buffer

        n_final, _, _, rollout_buffer = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (jnp.asarray(0), state, jnp.asarray(False), rollout_buffer),
        )
        return _ForwardRollout(n_final, rollout_buffer)

    return uturn_fn


def _step_distribution(num_steps_to_uturn_value: Array, path_fraction: float):
    """``Lo(theta,rho)`` and ``W(theta,rho)``, section 2.2.3 (eq. 34-35)."""
    lo = jnp.maximum(
        1, jnp.floor(path_fraction * num_steps_to_uturn_value).astype(jnp.int32)
    )
    width = num_steps_to_uturn_value - lo + 1
    return lo, width


def _tuning_parameter_fn(
    integrator: Callable, step_size: float, max_num_steps: int, path_fraction: float
) -> Callable:
    def tuning_parameter_fn(rng_key, state, logdensity_fn, metric):
        uturn_fn = _num_steps_to_uturn_with_rollout(
            integrator, step_size, metric, max_num_steps
        )
        rollout = uturn_fn(state, logdensity_fn)
        lo, _ = _step_distribution(rollout.num_steps_to_uturn, path_fraction)
        num_steps = jax.random.randint(
            rng_key, shape=(), minval=lo, maxval=rollout.num_steps_to_uturn + 1
        )
        return num_steps, rollout

    return tuning_parameter_fn


def _apply_fn(
    integrator: Callable, step_size: float, max_num_steps: int, path_fraction: float
) -> Callable:
    def apply_fn(state, alpha, aux, logdensity_fn, metric):
        num_steps = alpha
        rollout: _ForwardRollout = aux
        forward = rollout.num_steps_to_uturn

        # GATHER, not a re-integration (Issue#1058): `num_steps` (`alpha`) is
        # always <= `forward` by construction of `_tuning_parameter_fn`'s
        # draw, so `rollout.states[num_steps - 1]` was written by the
        # forward rollout above and is bit-identical to what
        # `trajectory.static_integration` would (re-)compute here.
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

        uturn_fn = num_steps_to_uturn(integrator, step_size, metric, max_num_steps)
        reverse = uturn_fn(proposal_state, logdensity_fn)

        _, width_forward = _step_distribution(forward, path_fraction)
        lo_reverse, width_reverse = _step_distribution(reverse, path_fraction)

        is_in_reverse_interval = (num_steps >= lo_reverse) & (num_steps <= reverse)
        log_tuning_density_ratio = jnp.where(
            is_in_reverse_interval,
            jnp.log(width_forward.astype(jnp.float32))
            - jnp.log(width_reverse.astype(jnp.float32)),
            -jnp.inf,
        )
        extra_info = _TrajectoryLengthExtra(
            num_integration_steps=num_steps,
            num_steps_to_uturn_forward=forward,
            num_steps_to_uturn_reverse=reverse,
            is_no_return_rejected=jnp.logical_not(is_in_reverse_interval),
        )
        return proposal_state, log_tuning_density_ratio, extra_info

    return apply_fn


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
    path_fraction: float = 0.5,
    max_num_steps: int = 1024,
) -> Callable:
    """Build a ``gist_trajectory_length`` kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian
        dynamics.
    divergence_threshold
        Value of the difference in energy above which we consider that the
        transition is divergent.
    path_fraction
        ``psi`` in ``[0, 1]``, section 2.2.3. Default 0.5 per the paper's
        own recommendation (comparable leapfrog-step counts to NUTS;
        ``psi=0`` is the simpler eq. 34 special case).
    max_num_steps
        Hard cap on each U-turn rollout (forward and reverse); analogous to
        NUTS's ``max_num_doublings``, but bounds a *linear* rollout here,
        not ``2**max`` leapfrog steps. Also sizes the forward-rollout buffer
        (module docstring, "Forward-rollout caching") -- ``O(max_num_steps)``
        leapfrog states are allocated per transition.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current
    state of the chain and that returns a new state of the chain along with
    information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: seam.GISTState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: metrics.MetricTypes,
    ) -> tuple[seam.GISTState, GISTTrajectoryLengthInfo]:
        """Generate a new sample with the ``gist_trajectory_length`` kernel."""
        tuning_parameter_fn = _tuning_parameter_fn(
            integrator, step_size, max_num_steps, path_fraction
        )
        apply_fn = _apply_fn(integrator, step_size, max_num_steps, path_fraction)

        new_state, info, raw_extra_info = seam._step(
            rng_key,
            state,
            logdensity_fn,
            tuning_parameter_fn,
            apply_fn,
            inverse_mass_matrix,
            divergence_threshold,
        )
        # `raw_extra_info` is declared as the generic `ArrayTree` in the
        # general spine (opaque to it by design); cast back to the concrete
        # type this kernel actually produces.
        extra_info = cast(_TrajectoryLengthExtra, raw_extra_info)
        trajectory_length_info = GISTTrajectoryLengthInfo(
            info.momentum,
            info.tuning_parameter,
            info.is_accepted,
            info.is_divergent,
            info.acceptance_rate,
            info.energy,
            info.num_integration_steps,
            extra_info.num_steps_to_uturn_forward,
            extra_info.num_steps_to_uturn_reverse,
            extra_info.is_no_return_rejected,
        )
        return new_state, trajectory_length_info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    inverse_mass_matrix: metrics.MetricTypes,
    step_size: float,
    *,
    path_fraction: float = 0.5,
    max_num_steps: int = 1024,
    divergence_threshold: float = 1000,
    integrator: Callable = integrators.velocity_verlet,
) -> SamplingAlgorithm:
    """``blackjax.gist_trajectory_length`` -- GIST self-tuning path length
    (no-U-turn condition, section 2.2; NOT NUTS's recursive doubling).

    Examples
    --------

    A new ``gist_trajectory_length`` kernel can be initialized and used with
    the following code:

    .. code::

        gist_trajectory_length = blackjax.gist_trajectory_length(
            logdensity_fn, inverse_mass_matrix, step_size=0.1
        )
        state = gist_trajectory_length.init(position)
        new_state, info = gist_trajectory_length.step(rng_key, state)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value
        for the momentum and computing the kinetic energy.
    step_size
        ``h``, fixed (not GIST-adapted in this instance; see the module
        docstring for composing with ``gist_step_size``).
    path_fraction
        ``psi`` in ``[0, 1]``, section 2.2.3. Default 0.5 per the paper's
        own recommendation (comparable leapfrog-step counts to NUTS;
        ``psi=0`` is the simpler eq. 34 special case).
    max_num_steps
        Hard cap on each U-turn rollout (forward and reverse); analogous to
        NUTS's ``max_num_doublings``, but bounds a *linear* rollout here,
        not ``2**max`` leapfrog steps -- size accordingly (NUTS's default of
        10 doublings caps at 1023 steps; a directly-comparable cap here is
        ``max_num_steps ~= 1024``). Also sizes the forward-rollout buffer
        (module docstring, "Forward-rollout caching") -- ``O(max_num_steps)``
        leapfrog states are allocated per transition, so an oversized cap on
        a high-dimensional target trades unused memory for compute.
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
    kernel = build_kernel(
        integrator, divergence_threshold, path_fraction, max_num_steps
    )
    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        kernel_args=(step_size, inverse_mass_matrix),
    )
