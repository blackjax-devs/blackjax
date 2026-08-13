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

The tuning parameter is ``alpha = (a, b, j, L)``: ``(a, b, j)`` select a
step size ``h = h0 * 2**j`` exactly as :mod:`~blackjax.mcmc.composed.step_size`
does (the doubling/halving selector, section 2.1.2, at a *fixed* trial
trajectory length ``L0`` -- never the sampled ``L``, since evaluating the
selector's own trial trajectory at a length that depends on the not-yet-drawn
``L`` would make the ``j``-then-``L|j`` factorization circular); ``L`` is then
drawn exactly as :mod:`~blackjax.mcmc.composed.trajectory_length` does, but
from the no-U-turn rollout built *at the selected* ``h`` rather than at a
fixed step size.

Both selections are Gibbs draws from a joint conditional that factors
sequentially::

    p(a, b, j, L | theta, rho)
      = Uniform_Delta(a, b) . 1{j = mu_L0(theta, rho, a, b)}
        . 1{L in I(U(theta, rho, h_j))} / W(theta, rho, h_j)

with ``h_j = h0 * 2**j``, ``mu_L0`` the step-size selector run with the fixed
trial length ``L0``, and ``U(., h)``/``I(.)``/``W(.)`` the no-U-turn rollout,
its psi-jittered interval, and that interval's width, all evaluated at ``h``
(:func:`blackjax.mcmc.composed.trajectory_length.num_steps_to_uturn`,
:func:`~blackjax.mcmc.composed.trajectory_length._step_distribution`).
Because ``g = identity`` carries all of ``(a, b, j, L)`` unchanged through the
involution (as both shipped instances already do), conditioning the reverse
draw's density on the same forward-drawn ``(a, b)`` collapses the GIST
tuning-density ratio (eq. 9) to three multiplicative factors evaluated at the
*forward-selected* ``h_j`` on both sides:

1. **the h-selection reversibility indicator** ``1{j' = j}``, where ``j'`` is
   the step-size selector re-run at the proposal with the *same* ``(a, b)``
   (:mod:`~blackjax.mcmc.composed.step_size`'s own reversibility check,
   section 2.1.3);
2. **the L-interval width ratio** ``W(theta, rho, h_j) /
   W(theta', rho', h_j)`` -- both endpoints evaluated at the *same*, forward-
   selected ``h_j``, not a base step size and not the proposal's own
   re-selected ``h'``: mixing step sizes across the forward/reverse rollout
   evaluates the wrong conditional density, since the declared ``p(L | x, j)``
   is defined in terms of ``h_j`` specifically;
3. **the L-interval membership indicator** ``1{L in I(U(theta', rho', h_j))}``
   -- a uniform distribution on a state-dependent interval contributes both
   its width *and* whether the realized draw actually falls inside the
   interval computed at the other endpoint; the width ratio alone is not the
   full density ratio of a interval-supported uniform law.

Omitting any one of these three factors breaks stationarity; see this
module's own falsification tests (single-factor-ablation mutants of each) for
the confirmed empirical failure modes and the standalone Gibbs-conditional
correctness argument for a full derivation.

Forward and reverse U-turn rollouts both run at the selected ``h_j``
--------------------------------------------------------------------
Because both the forward rollout (used to draw ``L``) and the reverse
rollout (used to build the width/membership factors above) are evaluated at
the *same* ``h_j``, the forward one reuses
:mod:`~blackjax.mcmc.composed.trajectory_length`'s buffered-rollout caching
(:func:`~blackjax.mcmc.composed.trajectory_length._num_steps_to_uturn_with_rollout`)
exactly as the standalone length instance does: the proposal for the drawn
``L`` is a gather from the buffer built while searching for ``U``, not a
fresh re-integration. The reverse rollout has nothing to gather (its own
states are never a candidate proposal) and stays an ordinary, unbuffered
:func:`~blackjax.mcmc.composed.trajectory_length.num_steps_to_uturn` call, as
in the standalone instance.

Symmetric exhaustion veto
--------------------------
The step-size selector's doubling/halving search can exhaust its iteration
budget without terminating (``search_exhausted``,
:mod:`~blackjax.mcmc.composed.step_size`). Because that search is
deterministic given its inputs, exhaustion is not itself a factor of the
displayed conditional density above -- it has to be handled as a forced
rejection outside it, exactly as the standalone step-size instance already
does. Rejecting on the search exhausting in *either* direction (forward or
at the reversibility re-check) rather than only the forward direction is
what keeps this forced rejection symmetric: for an involutive proposal map
``T`` and any measurable exhaustion predicate ``E``, the veto
``c(x, alpha) = 1{not(E(x, alpha) or E(T(x, alpha)))}`` satisfies
``c(x, alpha) = c(T(x, alpha), alpha)`` by construction (the union of the two
predicates is symmetric under swapping its two arguments), so vetoing on the
union preserves detailed balance where a forward-only veto would not.

Construction-time validation
-----------------------------
``path_fraction`` (``psi``) and ``max_num_steps`` are validated eagerly at
:func:`build_kernel` construction, not silently clamped or left to fail deep
inside a traced ``while_loop``: the displayed conditional density is only a
well-defined, normalized, strictly-positive law when ``0 <= psi <= 1`` (so
the interval ``I(U) = [max(1, floor(psi . U)), U]`` is well-formed) and
``max_num_steps >= 1`` (so every rollout visits at least one step and ``U``
is a positive integer). The step-size selector's own ``h_selector_trial_length``
(the frozen ``L0`` above) is validated the same way: a value below 1 leapfrog
step is not a trajectory.

Integrator assumption
----------------------
The correctness argument above assumes the supplied ``integrator`` is
reversible and volume-preserving for every fixed step size (a property of
the *algorithm*, not something this module can check at runtime for an
arbitrary callable). The default, :func:`blackjax.mcmc.integrators.velocity_verlet`,
has this property; a custom integrator substituted for it must too.

Estimand-aware ``psi`` default
--------------------------------
This instance defaults ``path_fraction`` (``psi``) to
``1 - 4.4934 / (2 * pi) ~= 0.2849`` (closed form; ``4.4934`` is the first
positive root of ``tan(x) = x``, the same constant behind the No-U-Turn
step-size heuristic in [3]) rather than the standalone length instance's
``0.5``. This value sits inside a flat cost-model-dependent plateau
(``psi`` roughly ``0.25`` to ``0.30`` all perform comparably once the
forward-rollout cost is ``psi``-independent, i.e. after the caching described
above); within that plateau it favors second-moment accuracy and worst-case
gap over first-moment accuracy relative to ``psi = 0.5`` (which is itself the
antithetic-mode-favoring choice). Prefer a larger ``psi`` (up to the
standalone instance's ``0.5`` default) if the target's first moments matter
more than its second moments for your use case; ``psi = 1`` collapses the
jittered interval to the single point ``{U}``, discarding the jitter that
otherwise regularizes trajectory-length-driven resonance (section 2.2.3) --
technically still a valid, well-defined instance of the conditional above,
but not recommended.

RNG discipline
---------------
The rng_key this module's Gibbs step receives is split into three
independent subkeys before drawing anything: one for ``(a, b)``, one
reserved (see below), and one for ``L``. The doubling/halving step-size
search itself is deterministic given ``(a, b)`` and the state -- it consumes
no randomness -- so its subkey is not drawn from and not passed to the
selector; it is split off explicitly anyway (rather than only ever deriving
two subkeys) so that a future randomized selection criterion inherits an
independent key without touching this split. Reusing (or implicitly
deriving) one key for two logically distinct draws would correlate them in a
way the conditional-density factorization above assumes they are not; this
module's own regression test checks conditional, not just marginal,
uniformity for exactly this reason.

Prior art
---------
The composition principle behind selecting a local step size and then
building an orbit conditional on that selection is already published:
adaptNUTS [2] composes a GIST step-size selector with NUTS-style path
construction through the same enlarged-space involution and reverse
conditional-density correction this module uses (its eqs. 16-18 and
Algorithm 7); ATLAS [4] independently constructs and proves an analogous
step-size-then-orbit composition with a different (Hessian/lognormal) step
distribution. This module is a new *instance* of that published composition
machinery -- the specific pairing of the [AutoStep] symmetric step-size
selector with the GIST survey's dense one-sided no-U-turn interval selector
-- not a claim to the composition principle itself.

References
----------
.. [1] Bou-Rabee, Carpenter, Marsden, "GIST: Gibbs self-tuning for locally
   adaptive Hamiltonian Monte Carlo", arXiv:2404.15253, Statistical Surveys
   2026, Vol. 20, pp. 135-179.
.. [2] Bou-Rabee, Carpenter, Kleppe, Marsden, "Incorporating Local Step-Size
   Adaptivity into the No-U-Turn Sampler using Gibbs Self Tuning",
   arXiv:2408.08259, Journal of Chemical Physics, eqs. 16-18, Algorithm 7,
   Appendix A.
.. [3] Hoffman, Gelman, "The No-U-Turn Sampler", JMLR 15, 2014 (the
   ``4.4934`` dual-averaging step-size heuristic this module's ``psi``
   default borrows its closed form from).
.. [4] Modi, "Delayed rejection Hamiltonian Monte Carlo for sampling
   multiscale distributions" / ATLAS, arXiv:2410.21587, Appendix D-E
   (ATLAS-Simple).
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
        re-check) hit its iteration budget without terminating -- a
        first-class diagnostic channel, not folded silently into
        ``is_accepted``: divergence count is near-definitional for a
        step-size-selecting instance (the selector drives the very energy
        error the divergence flag thresholds) and is therefore *not* a
        reliable success criterion on its own; ``h_search_exhausted``
        together with ``is_no_return_rejected`` below are the loud,
        structural channels to monitor instead.
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

        # GATHER, not a re-integration: `num_steps` is always <= `forward_uturn`
        # by construction of the draw above, so `rollout.states[num_steps - 1]`
        # was written by the forward rollout (built AT the selected `h`) and is
        # bit-identical to what a fresh integration would (re-)compute here --
        # the same caching seam `gist_trajectory_length` uses.
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

        # h-selection reversibility check: re-run the SAME ladder (same
        # frozen trial length, same (a, b)) at the proposal.
        reverse_step_index, h_search_exhausted_reverse = selector(
            proposal_state, a, b, logdensity_fn, metric
        )
        h_search_exhausted = h_search_exhausted_forward | h_search_exhausted_reverse
        is_h_reversible = reverse_step_index == step_index

        # Reverse no-U-turn rollout AT THE SAME, forward-selected `h` (not a
        # base step size, not the proposal's own re-selected `h'`).
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
        step size (the module docstring's "Integrator assumption").
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
        The step-size ladder's frozen trial trajectory length (``L0`` in the
        module docstring) -- fixed independently of the sampled ``L`` to
        avoid a circular definition. Must be >= 1.
    path_fraction
        ``psi`` in ``[0, 1]``, the no-U-turn interval's jitter fraction, see
        the module docstring's "Estimand-aware psi default". Defaults to
        ``0.2849`` (this instance's own default; the standalone
        ``gist_trajectory_length`` instance defaults to ``0.5``).
    max_num_steps
        Hard cap on each no-U-turn rollout (forward and reverse), both built
        at the selected ``h``; also sizes the forward-rollout buffer (module
        docstring). Must be >= 1.

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
        circular definition (see the module docstring). Default 1
        reproduces the MALA-equivalent single-leapfrog-step trial the
        standalone ``gist_step_size`` instance also defaults to.
    path_fraction
        ``psi`` in ``[0, 1]``, see the module docstring's "Estimand-aware
        psi default". Default ``0.2849``.
    max_num_steps
        Hard cap on each no-U-turn rollout (forward and reverse), both built
        at the selected ``h``. Also sizes the forward-rollout buffer (module
        docstring) -- ``O(max_num_steps)`` leapfrog states are allocated per
        transition.
    divergence_threshold
        The absolute value of the difference in energy between two states
        above which we say that the transition is divergent. Divergence
        count is not a reliable success criterion for this instance on its
        own -- see ``GISTComposedInfo.h_search_exhausted`` docstring.
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
