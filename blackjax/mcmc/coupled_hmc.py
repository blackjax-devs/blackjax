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
"""Two HMC chains advanced from a shared source of randomness.

This module runs two ordinary HMC marginals side by side and couples them
*only through their random inputs*: the standard normal that becomes each
momentum, and the single uniform that drives each Metropolis test.

.. warning::

    The scope of what is implemented here is deliberately narrow.  This
    module provides a **coupling of the random inputs**.  It makes no claim
    that the pair is invariant for the product of the two targets, that the
    chains meet (exactly, maximally, or at all), that any estimator built
    from the pair is unbiased, or that coupling improves sampling efficiency
    in any sense.  None of those properties is implemented or tested.

The contract
------------

The one property this module does assert, and test, is:

    Coupling changes only the *joint law* of the two marginals' random
    inputs.  It never changes either marginal's transition function, its
    Metropolis decision rule, or its cached log-density and gradient.

Concretely, given the same ``(state, standard_normal, uniform)`` triple, each
marginal here performs exactly the transition an uncoupled HMC marginal built
from the same target, metric, step size and integration count would perform,
and returns exactly the same state and info.  The two marginals share a
uniform but each compares it against **its own** acceptance probability, so a
shared uniform is emphatically *not* a shared decision.  Each marginal carries
its own ``logdensity``/``logdensity_grad`` cache.

Mathematical semantics
----------------------

Write :math:`M` for a marginal's mass matrix and :math:`A` for the momentum
square root BlackJAX uses, so that a momentum drawn as :math:`p = A z` with
:math:`z \\sim N(0, I)` has covariance :math:`A A^\\top = M`.  Both couplings
below draw a single :math:`z` and hand each marginal a transformed copy:

``synchronous``
    Both marginals receive the same :math:`z`.

``reflection``
    The first marginal receives :math:`z`; the second receives

    .. math::

        z' = z - 2 e (e^\\top z)

    for a unit vector :math:`e`.  That map is an orthogonal reflection, so in
    exact arithmetic it preserves :math:`N(0, I)` and the second marginal's
    momentum law is unchanged **for any** unit :math:`e` that does not depend
    on :math:`z`.  In floating point the normalisation of :math:`e` carries a
    rounding error of a few units in the last place, so the preservation is
    exact in the mathematics and accurate to that tolerance in the code.  That is the whole of what reflection buys here: marginal
    correctness.  It is not a statement about contraction, meeting, or
    coupling quality, and none is made.

    ``e`` comes from ``direction_fn``, which is fixed when the kernel is
    built and is applied to the *incoming* pair of states.  It is given the two
    states and the first metric, and no innovations.  Note that this is a
    statement about its arguments, not about ordering: the kernel draws ``z``
    and the uniform first and only then runs the transition that evaluates
    ``direction_fn``.  Supplying a ``direction_fn`` that is a pure function of
    its arguments is the caller's part of the contract.  The default direction is the
    difference of the two positions whitened by the **first** marginal's
    metric, :math:`e \\propto A^{-1}(x_1 - x_2)`.  That particular choice is a
    heuristic: it is the direction along which, to leading order, the two
    chains' displacements respond oppositely, since the velocity
    :math:`\\partial K / \\partial p` equals :math:`A^{-\\top} z` and hence
    :math:`\\langle x_1 - x_2, \\partial K/\\partial p\\rangle = \\langle
    A^{-1}(x_1 - x_2), z\\rangle`.  When the two marginals use *different*
    metrics the default still uses the first marginal's metric only; it is
    then a first-metric-based direction with no claimed property for the
    pair.  A zero direction is the identity map, i.e. reflection degenerates
    to synchronous coupling for that transition.

Relation to ``blackjax.hmc``
----------------------------

Each marginal applies the same *mathematical* Metropolis rule as ordinary
HMC: accept with probability :math:`\\min(1, e^{\\Delta})`.  It realises that
rule by comparing a supplied uniform against the acceptance probability,
whereas :func:`~blackjax.mcmc.proposal.static_binomial_sampling` realises it
with :func:`jax.random.bernoulli`.  The two agree as distributions and do
**not** agree draw-for-draw: feeding the same key to this kernel and to
``blackjax.hmc`` will not reproduce the same accept/reject sequence, and no
such parity is claimed or tested.

Usage
-----

There is deliberately no top-level ``blackjax.coupled_hmc``; reach this
module at ``blackjax.mcmc.coupled_hmc``.  Every per-marginal parameter is
passed as an explicit ``(first, second)`` pair -- including when both
marginals share a value, which is written ``(value, value)``.  Nothing is
broadcast or inferred, because positions, metrics and step sizes may
themselves legitimately be tuples.

.. code::

    import blackjax.mcmc.coupled_hmc as coupled_hmc

    algorithm = coupled_hmc.as_top_level_api(
        (logdensity_fn, logdensity_fn),
        step_size=(0.1, 0.1),
        inverse_mass_matrix=(inverse_mass_matrix, inverse_mass_matrix),
        num_integration_steps=(10, 10),
        coupling="reflection",
    )
    state = algorithm.init((position_one, position_two))
    state, info = algorithm.step(rng_key, state)

"""
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import blackjax.mcmc.hmc as hmc
import blackjax.mcmc.integrators as integrators
import blackjax.mcmc.metrics as metrics
from blackjax.base import Info, Position, SamplingAlgorithm, State
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = [
    "CoupledHMCState",
    "CoupledHMCInfo",
    "init",
    "build_kernel",
    "as_top_level_api",
    "validate_marginal_inputs",
]


class CoupledHMCState(NamedTuple):
    """State of a pair of coupled HMC chains.

    The two marginals are held as two complete, independent
    :class:`~blackjax.mcmc.hmc.HMCState` values, so each carries its own
    position and its own cached ``logdensity``/``logdensity_grad``.  Keeping
    them separate makes crossing the two caches an obvious error rather than
    a silent one -- it does not make it impossible, so the test suite checks
    it explicitly.

    first
        State of the first marginal chain.
    second
        State of the second marginal chain.

    """

    first: hmc.HMCState
    second: hmc.HMCState


class CoupledHMCInfo(NamedTuple):
    """Additional information on a coupled HMC transition.

    Both marginals' complete :class:`~blackjax.mcmc.hmc.HMCInfo` objects are
    preserved untouched, so every per-chain diagnostic (momentum, acceptance
    rate, acceptance decision, divergence flag, energy, proposal, integration
    count) stays separately visible.  The remaining three fields describe the
    shared randomness itself.

    first
        Transition information for the first marginal.
    second
        Transition information for the second marginal.
    common_normal
        The flat standard normal vector handed to the first marginal.  The
        second marginal receives this vector under the coupling map, i.e.
        unchanged for ``"synchronous"`` and reflected for ``"reflection"``.
    reflection_unit
        The unit vector used by the reflection, in the same flat coordinates
        as ``common_normal``.  Exactly zero under synchronous coupling, and
        exactly zero for a reflection whose direction was zero (both denote
        the identity map).
    uniform
        The single uniform variate shared by both Metropolis tests.  Each
        marginal compares it against its own acceptance probability.

    """

    first: hmc.HMCInfo
    second: hmc.HMCInfo
    common_normal: Array
    reflection_unit: Array
    uniform: float


# --------------------------------------------------------------------
#                       PAIRS AND INPUT CHECKING
# --------------------------------------------------------------------


def _as_pair(value, name: str) -> tuple:
    """Require an explicit two-element pair.

    Nothing is broadcast: a value shared by both marginals must be written
    ``(value, value)``.  Positions, metrics and step sizes can themselves be
    tuples, so inferring a pair from a bare value would be ambiguous exactly
    where the mistake is most costly -- and the cost is silent.  A bare dense
    ``(d, d)`` inverse mass matrix, if it were unpacked as a pair, would index
    to its first two ROWS: the first marginal would run with row 0 as a
    *diagonal* metric and the second with row 1.  The shapes are plausible, the
    run completes, and the results are wrong with nothing raised anywhere.
    """
    if isinstance(value, tuple) and len(value) == 2:
        return value
    if isinstance(value, list) and len(value) == 2:
        return tuple(value)
    raise TypeError(
        f"`{name}` must be an explicit (first, second) pair. Pass "
        f"`({name}, {name})` when both marginals share a value."
    )


def _check_metric_kind(inverse_mass_matrix):
    """Reject metric inputs this module does not support, without tracing.

    Only the three array-shaped Euclidean metric payloads are accepted: a
    one-dimensional diagonal, a two-dimensional dense matrix, and a
    :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`.  A callable
    (Riemannian) metric and a pre-built :class:`~blackjax.mcmc.metrics.Metric`
    are refused rather than assumed valid: the coupling maps momenta through
    ``Metric.scale``, and that this reproduces ``Metric.sample_momentum``
    cannot be verified for an opaque or position-dependent metric.

    This is a static type check only.  It performs no array conversion and is
    safe on tracers; the numerical checks live in
    :func:`validate_marginal_inputs`.
    """
    if isinstance(inverse_mass_matrix, metrics.LowRankInverseMassMatrix):
        return inverse_mass_matrix
    if isinstance(inverse_mass_matrix, metrics.Metric):
        raise TypeError(
            "coupled_hmc does not accept a pre-built Metric; pass a diagonal "
            "array, a dense array, or a LowRankInverseMassMatrix."
        )
    if callable(inverse_mass_matrix):
        raise TypeError(
            "coupled_hmc does not accept a callable (Riemannian) metric; the "
            "coupling is defined for fixed Gaussian Euclidean metrics only."
        )
    return inverse_mass_matrix


def _flat_position(position: ArrayLikeTree) -> tuple[Array, Callable]:
    """Flatten a position, requiring one nonempty real floating dtype.

    The dtype compared here is the **effective** one -- what the leaf becomes
    once JAX has it, via ``jnp.asarray`` -- not the dtype the caller's object
    declares.  That is deliberate and is the opposite choice from
    :func:`_check_innovations`, so the difference is worth stating.

    A position is the thing everything else is measured against, and
    ``ravel_pytree`` will convert its leaves regardless, so what matters is the
    dtype they end up with.  Under JAX's default configuration a float64 leaf
    becomes float32, and a position mixing float64 and float32 leaves is
    therefore accepted: after conversion they genuinely agree, and there is no
    later step at which they could disagree.

    An innovation is a different case.  There the caller supplies a value that
    is compared against an acceptance probability, so narrowing it silently
    changes the number the Metropolis test uses.  :func:`_check_innovations`
    consequently inspects the *declared* dtype and refuses a mismatch instead
    of converting.  Neither function promises both notions at once.
    """
    leaves = jax.tree.leaves(position)
    if not leaves:
        raise TypeError("positions must have at least one array leaf")
    dtypes = {jnp.asarray(leaf).dtype for leaf in leaves}
    if len(dtypes) != 1:
        raise TypeError(
            f"all position leaves must share one floating dtype, got {sorted(map(str, dtypes))}"
        )
    flat, unravel = ravel_pytree(position)
    if not jnp.issubdtype(flat.dtype, jnp.floating) or flat.size == 0:
        raise TypeError("positions must have nonempty real floating coordinates")
    return flat, unravel


def _check_paired_positions(first_position, second_position) -> None:
    """Require identical tree structure, leaf shapes and dtype across the pair."""
    first_structure = jax.tree.structure(first_position)
    second_structure = jax.tree.structure(second_position)
    if first_structure != second_structure:
        raise ValueError(
            "paired positions must share one pytree structure, got "
            f"{first_structure} and {second_structure}"
        )
    # Per-leaf shapes, not merely equal flattened size.  Two positions can share
    # a tree structure and a total size while splitting it differently -- say
    # {"a": (2,), "b": (3,)} against {"a": (3,), "b": (2,)}.  This enforces the
    # documented contract that the pair has matching leaf shapes.  Each marginal
    # would still be individually valid in that situation, since each unravels
    # the flat innovation consistently with its own position; what breaks is the
    # pairing, because the shared vector lands on different leaf boundaries in
    # each chain and so does not mean the same thing to both.
    first_leaves = jax.tree.leaves(first_position)
    second_leaves = jax.tree.leaves(second_position)
    # `strict=True` states the dependence on the structure check above, which is
    # what guarantees the two leaf lists have equal length.
    for index, (first_leaf, second_leaf) in enumerate(
        zip(first_leaves, second_leaves, strict=True)
    ):
        if jnp.shape(first_leaf) != jnp.shape(second_leaf):
            raise ValueError(
                "paired positions must have matching leaf shapes, leaf "
                f"{index} has {jnp.shape(first_leaf)} and {jnp.shape(second_leaf)}"
            )
    first_flat, _ = _flat_position(first_position)
    second_flat, _ = _flat_position(second_position)
    # No separate flat-shape check: equal tree structure with equal per-leaf
    # shapes already implies equal flattened shape.
    # Effective dtype, per `_flat_position`: this compares what the positions
    # become in the computation, not what the caller's objects declared.
    if first_flat.dtype != second_flat.dtype:
        raise TypeError(
            "paired positions must have matching floating dtypes, got "
            f"{first_flat.dtype} and {second_flat.dtype}"
        )


def _declared_dtype(value):
    """The dtype the value itself declares, or ``None`` if it declares none.

    A NumPy or JAX array carries a dtype and is therefore making a claim about
    its precision.  A bare Python ``float`` or ``int`` carries none and is
    weakly typed, exactly as elsewhere in JAX, so it is adopted into the
    position dtype rather than refused.
    """
    if getattr(value, "weak_type", False):
        # JAX's own weak-typing flag. A weakly typed array is adopted into the
        # surrounding dtype everywhere else in JAX, so refusing it here would
        # contradict the convention this function claims to follow.
        return None
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else np.dtype(dtype)


def _declared_shape(value):
    """The shape of the value as given, without converting it.

    ``np.shape`` rather than a ``.shape`` attribute lookup: the attribute is
    absent on a plain Python list, which would then be reported as a scalar,
    slip past the shape check, and fail later inside ``float()`` with a message
    that never names the argument.
    """
    return tuple(np.shape(value))


def _metric_dimension(inverse_mass_matrix):
    """The dimension the metric is defined on, from static shapes only.

    Read from shapes rather than values, so it is safe on tracers. Returns
    ``None`` when the shape is not one of the supported forms, leaving the
    existing kind checks to speak.
    """
    if isinstance(inverse_mass_matrix, metrics.LowRankInverseMassMatrix):
        shape = jnp.shape(inverse_mass_matrix.sigma)
        return shape[0] if len(shape) == 1 else None
    shape = jnp.shape(inverse_mass_matrix)
    return shape[0] if len(shape) in (1, 2) else None


def _check_innovations(standard_normal, uniform, flat_position) -> None:
    """Require prescribed innovations to match the position shape and dtype.

    The inputs are inspected **as given**, before any conversion.  That
    ordering is the whole point: ``jnp.asarray`` narrows a float64 input to
    float32 under JAX's default configuration, so converting first and
    comparing dtypes afterwards compares two values that already agree and
    checks nothing.

    Narrowing is not cosmetic: a float64 uniform below one becomes exactly
    ``1.0`` in float32 and then rejects a certain acceptance, and one in range
    becomes a *different* float32, so the test would use a value the caller
    never gave.  An array declaring a dtype must match the position exactly; a
    bare Python scalar declares none, is weakly typed and is adopted, so its
    domain is checked after conversion instead.
    """
    if _declared_shape(standard_normal) != flat_position.shape:
        raise ValueError(
            "`standard_normal` must be a flat vector matching the position, got "
            f"{_declared_shape(standard_normal)}, expected {flat_position.shape}"
        )
    declared = _declared_dtype(standard_normal)
    if declared is not None and declared != flat_position.dtype:
        raise TypeError(
            "`standard_normal` dtype must match the position dtype exactly "
            f"(no narrowing), got {declared}, expected {flat_position.dtype}"
        )
    if _declared_shape(uniform) != ():
        raise ValueError(
            f"`uniform` must be a scalar, got shape {_declared_shape(uniform)}"
        )
    declared = _declared_dtype(uniform)
    if declared is not None and declared != flat_position.dtype:
        raise TypeError(
            "`uniform` dtype must match the position dtype exactly (no "
            f"narrowing), got {declared}, expected {flat_position.dtype}"
        )

    # A uniform outside [0, 1) is refused, never clipped and never allowed to
    # masquerade as an ordinary Metropolis rejection.  The value can only be
    # read when it is concrete, so under `jit` or `vmap` this remains a
    # documented precondition rather than a check.  It is read after conversion
    # to the position dtype, because that is the value the Metropolis test will
    # actually compare.
    if not isinstance(uniform, jax.core.Tracer):
        value = float(jnp.asarray(uniform, flat_position.dtype))
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"`uniform` must lie in [0, 1) in the position dtype, got {value!r}. "
                "It is not clipped, because a clipped uniform would silently "
                "become an ordinary rejection."
            )
    if not isinstance(standard_normal, jax.core.Tracer) and not bool(
        np.all(np.isfinite(np.asarray(standard_normal)))
    ):
        raise ValueError("`standard_normal` must be finite")


def _concrete_real_scalar(value, name: str) -> float:
    """Read a concrete real scalar, accepting the array forms BlackJAX produces.

    Warmup returns a step size as a zero-dimensional JAX array, so requiring a
    Python float would force callers to hand-cast routine BlackJAX output.
    Booleans, complex values, non-scalars, non-finite values and tracers are
    still refused.
    """
    if isinstance(value, jax.core.Tracer):
        raise TypeError(
            f"`{name}` must be concrete here -- a traced value cannot be checked "
            "eagerly. Use `build_kernel`, which performs no eager validation."
        )
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"`{name}` must be a real number, not a boolean")
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"`{name}` must be a scalar, got shape {array.shape}")
    if array.dtype == np.bool_ or not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(
            f"`{name}` must have a real floating or integer dtype, got {array.dtype}"
        )
    if not np.isfinite(array):
        raise ValueError(f"`{name}` must be finite, got {array!r}")
    return float(array)


def _concrete_integer_scalar(value, name: str) -> int:
    """Read a concrete integer scalar, accepting NumPy and JAX scalar forms.

    A floating value is refused here rather than left to fail later. Nothing
    truncates it -- ``fori_loop`` rejects a float bound -- but it does so with a
    message about loop bound types that never names the parameter, and only
    once a trajectory is being built. This is a diagnostic improvement, not a
    correctness guard.
    """
    if isinstance(value, jax.core.Tracer):
        raise TypeError(
            f"`{name}` must be concrete here -- a traced value cannot be checked "
            "eagerly. Use `build_kernel`, which performs no eager validation."
        )
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"`{name}` must be an integer, not a boolean")
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"`{name}` must be a scalar, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"`{name}` must have an integer dtype, got {array.dtype}")
    return int(array)


def validate_marginal_inputs(inverse_mass_matrix, step_size, num_integration_steps):
    """Eagerly validate one marginal's concrete metric and integration settings.

    These are host-side numerical checks on *concrete* values: they use NumPy
    and cannot run on tracers, so they are performed once when an algorithm is
    constructed and are deliberately absent from the traced kernel.

    .. important::

        Passing this check says the values supplied *now* are admissible.  It
        says nothing about values that appear later inside a jitted or
        vmapped computation, which are never seen by this function.  A caller
        who builds a kernel with :func:`build_kernel` and feeds it traced
        metrics is responsible for their admissibility.

    Raises
    ------
    TypeError, ValueError
        If the metric is not a supported kind, is not finite, is not positive
        definite, or the integration settings are not positive and finite.

    """
    inverse_mass_matrix = _check_metric_kind(inverse_mass_matrix)

    if _concrete_real_scalar(step_size, "step_size") <= 0:
        raise ValueError("`step_size` must be positive")
    if _concrete_integer_scalar(num_integration_steps, "num_integration_steps") <= 0:
        raise ValueError("`num_integration_steps` must be positive")

    if isinstance(inverse_mass_matrix, metrics.LowRankInverseMassMatrix):
        sigma, basis, eigenvalues = (np.asarray(x) for x in inverse_mass_matrix)
        if sigma.ndim != 1 or not sigma.size:
            raise ValueError("`sigma` must be a nonempty vector")
        if basis.ndim != 2 or basis.shape[0] != sigma.size:
            raise ValueError("`U` must have shape (dimension, rank)")
        if eigenvalues.shape != (basis.shape[1],):
            raise ValueError("`lam` must have shape (rank,)")
        arrays = (sigma, basis, eigenvalues)
        if not all(np.issubdtype(x.dtype, np.floating) for x in arrays):
            raise TypeError("metric arrays must have real floating dtypes")
        if not all(np.isfinite(x).all() for x in arrays):
            raise ValueError("metric arrays must be finite")
        if np.any(sigma <= 0) or np.any(eigenvalues <= 0):
            raise ValueError("`sigma` and `lam` must be positive")
        tolerance = 100 * max(np.finfo(x.dtype).eps for x in arrays)
        # Columns with an eigenvalue of exactly one are neutral, so only the
        # active columns need to be orthonormal.
        active = basis[:, eigenvalues != 1]
        if active.shape[1] and not np.allclose(
            active.T @ active,
            np.eye(active.shape[1]),
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError(
                "active `U` columns must be orthonormal; lam=1 columns are neutral"
            )
        return

    matrix = np.asarray(inverse_mass_matrix)
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError("the inverse mass matrix must have a real floating dtype")
    if matrix.ndim not in (1, 2) or not matrix.size:
        raise ValueError(
            "the inverse mass matrix must be a nonempty vector or square matrix"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("the inverse mass matrix must be finite")
    if matrix.ndim == 1:
        if np.any(matrix <= 0):
            raise ValueError("a diagonal inverse mass matrix must be positive")
        return
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("a dense inverse mass matrix must be square")
    # Do not silently symmetrize a caller's kinetic energy.
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("a dense inverse mass matrix must be exactly symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "a dense inverse mass matrix must be positive definite"
        ) from error


# --------------------------------------------------------------------
#                        THE MARGINAL TRANSITION
# --------------------------------------------------------------------


def _uniform_binomial_sampling(uniform, log_p_accept, proposal, new_proposal):
    """Accept or reject using a supplied uniform rather than a key.

    This is :func:`~blackjax.mcmc.proposal.static_binomial_sampling` with the
    Bernoulli draw replaced by a comparison against a uniform the caller
    provides, which is what lets two marginals share one variate.  The
    mathematical rule is identical -- accept with probability
    ``min(1, exp(log_p_accept))`` -- and the realisation is not: it does not
    reproduce ``jax.random.bernoulli`` draw-for-draw.

    The uniform is required to lie in ``[0, 1)``.  Under that precondition the
    endpoints behave correctly: ``p_accept == 0`` (including a divergence,
    where ``log_p_accept`` is ``-inf``) always rejects, and ``p_accept == 1``
    always accepts.  The value is never clipped, so an out-of-domain uniform
    is not quietly converted into an ordinary Metropolis rejection.
    """
    p_accept = jnp.clip(jnp.exp(log_p_accept), max=1)
    do_accept = uniform < p_accept
    info = do_accept, p_accept, log_p_accept
    return (
        jax.lax.cond(
            do_accept,
            lambda: new_proposal,
            lambda: proposal,
        ),
        info,
    )


def _build_prescribed_marginal(
    logdensity_fn: Callable,
    inverse_mass_matrix: metrics.MetricTypes,
    step_size: float,
    num_integration_steps: int,
    integrator: Callable,
    divergence_threshold: float,
):
    """Build one marginal whose randomness is supplied rather than drawn.

    Returns ``(metric, step)`` where
    ``step(state, standard_normal, uniform) -> (HMCState, HMCInfo)`` is a pure
    function of its arguments.

    The momentum is obtained as ``metric.scale(position, z, inv=False,
    trans=False)``, which is precisely the map ``Metric.sample_momentum``
    applies to a standard normal for the three supported Euclidean metric
    kinds.  Externalising the draw therefore leaves the marginal momentum law
    unchanged; the remainder of the transition is BlackJAX's own trajectory,
    endpoint flip, energy difference and Metropolis test.
    """
    inverse_mass_matrix = _check_metric_kind(inverse_mass_matrix)
    metric = metrics.default_metric(inverse_mass_matrix)
    metric_dimension = _metric_dimension(inverse_mass_matrix)
    symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
    generate = hmc.hmc_proposal(
        symplectic_integrator,
        metric.kinetic_energy,
        step_size,
        num_integration_steps,
        divergence_threshold,
        sample_proposal=_uniform_binomial_sampling,
    )

    def step(
        state: hmc.HMCState, standard_normal: Array, uniform: Array
    ) -> tuple[hmc.HMCState, hmc.HMCInfo]:
        flat, unravel = _flat_position(state.position)
        # A metric of the wrong dimension does not necessarily fail: a
        # length-one diagonal broadcasts silently across any position, so one
        # marginal could run isotropic while its partner runs anisotropic with
        # nothing said. A mismatched-but-non-broadcasting length fails much
        # later, in a message naming neither the metric nor which marginal.
        if metric_dimension is not None and flat.shape[0] != metric_dimension:
            raise ValueError(
                "the inverse mass matrix must have the same dimension as the "
                f"position, got {metric_dimension}, expected {flat.shape[0]}"
            )
        _check_innovations(standard_normal, uniform, flat)
        momentum = metric.scale(
            state.position, unravel(standard_normal), inv=False, trans=False
        )
        integrator_state = integrators.IntegratorState(
            state.position, momentum, state.logdensity, state.logdensity_grad
        )
        selected, info, _ = generate(uniform, integrator_state)
        new_state = hmc.HMCState(
            selected.position, selected.logdensity, selected.logdensity_grad
        )
        return new_state, info

    return metric, step


# --------------------------------------------------------------------
#                          THE COUPLING MAPS
# --------------------------------------------------------------------


def _reflection_unit(direction: Array) -> Array:
    """Normalise a direction robustly; a zero direction gives the identity.

    The direction is first divided by its largest absolute entry, so a
    direction whose norm would overflow or underflow in the working dtype is
    still normalised correctly.  This is load-bearing rather than defensive, and
    the failure it prevents is silent.  At float32 -- BlackJAX's default -- a
    difference like ``[1e20, -2e20, 5e19, 3e20]`` has a norm that overflows, so
    a naive ``d / norm(d)`` yields exactly zero: the reflection would quietly
    become the identity and the pair would report ``reflection_unit = 0`` while
    running synchronous coupling under a reflection label.  The mirrored
    underflow case yields infinities.  Both are reachable for a diverging chain.
    A direction that is exactly zero is returned as zero, which makes the
    reflection the identity map -- that case is intended and documented.

    A direction containing a non-finite entry propagates as non-finite rather
    than silently producing a plausible-looking unit vector.
    """
    maximum = jnp.max(jnp.abs(direction))
    scaled = direction / jnp.where(maximum == 0, 1, maximum)
    norm = jnp.linalg.norm(scaled)
    return scaled / jnp.where(norm == 0, 1, norm)


def _reflect(standard_normal: Array, unit: Array) -> Array:
    """Reflect ``standard_normal`` in the hyperplane orthogonal to ``unit``."""
    return standard_normal - 2 * unit * jnp.dot(unit, standard_normal)


def whitened_difference(
    first_state: hmc.HMCState,
    second_state: hmc.HMCState,
    first_metric: metrics.Metric,
) -> Array:
    """Default reflection direction: the position difference, whitened.

    Returns :math:`A^{-1}(x_1 - x_2)` in flat coordinates, where :math:`A` is
    the first marginal's momentum square root.  ``Metric.scale`` with
    ``inv=True, trans=True`` is exactly that inverse map.

    Only the first marginal's metric is used.  When the two marginals carry
    different metrics this is therefore a first-metric-based direction, and no
    property is claimed for the pair.  A policy needing the second metric can
    close over it.
    """
    delta = jax.tree.map(
        lambda a, b: a - b, first_state.position, second_state.position
    )
    scaled = first_metric.scale(first_state.position, delta, inv=True, trans=True)
    return ravel_pytree(scaled)[0]


def _build_prescribed_pair(
    logdensity_fn: Sequence[Callable],
    inverse_mass_matrix: Sequence[metrics.MetricTypes],
    step_size: Sequence[float],
    num_integration_steps: Sequence[int],
    integrator: Callable,
    divergence_threshold: float,
    coupling: str,
    direction_fn: Callable | None,
):
    """Build the coupled transition whose randomness is supplied, not drawn.

    Returns ``step(state, standard_normal, uniform) -> (CoupledHMCState,
    CoupledHMCInfo)``.  This is the deterministic core of the kernel: given
    the pair of incoming states and one ``(z, u)`` pair it is an ordinary pure
    function, which is what makes the coupling separately checkable without
    also reasoning about key splitting.  It is internal to this module.

    This function passes ``direction_fn`` nothing but the incoming states and
    the first metric -- the innovations it receives are not among its
    arguments, though they do already exist by the time it is called. Whether
    the returned direction is genuinely free of them is a contract the caller
    keeps, since a callable may close over anything.
    """
    logdensity_fns = _as_pair(logdensity_fn, "logdensity_fn")
    step_sizes = _as_pair(step_size, "step_size")
    inverse_mass_matrices = _as_pair(inverse_mass_matrix, "inverse_mass_matrix")
    integration_steps = _as_pair(num_integration_steps, "num_integration_steps")
    if coupling == "reflection" and direction_fn is None:
        raise ValueError("reflection coupling requires a direction function")
    measure_direction = direction_fn

    first_metric, first_step = _build_prescribed_marginal(
        logdensity_fns[0],
        inverse_mass_matrices[0],
        step_sizes[0],
        integration_steps[0],
        integrator,
        divergence_threshold,
    )
    _, second_step = _build_prescribed_marginal(
        logdensity_fns[1],
        inverse_mass_matrices[1],
        step_sizes[1],
        integration_steps[1],
        integrator,
        divergence_threshold,
    )

    def step(
        state: CoupledHMCState, standard_normal: Array, uniform: Array
    ) -> tuple[CoupledHMCState, CoupledHMCInfo]:
        _check_paired_positions(state.first.position, state.second.position)
        flat, _ = _flat_position(state.first.position)

        if coupling == "reflection":
            assert measure_direction is not None  # narrowed above
            direction = jnp.asarray(
                measure_direction(state.first, state.second, first_metric)
            )
            if direction.shape != flat.shape:
                raise ValueError(
                    "`direction_fn` must return a flat direction matching the "
                    f"position, got {direction.shape}, expected {flat.shape}"
                )
            if direction.dtype != flat.dtype:
                # Name the likely cause rather than only the messenger: with
                # the default direction this mismatch comes from a metric whose
                # dtype differs from the position's, not from anything the
                # caller wrote.
                raise TypeError(
                    "the reflection direction must have the position dtype, got "
                    f"{direction.dtype}, expected {flat.dtype}"
                )
            unit = _reflection_unit(direction)
            second_normal = _reflect(standard_normal, unit)
        else:
            unit = jnp.zeros(flat.shape, flat.dtype)
            second_normal = standard_normal

        first_state, first_info = first_step(state.first, standard_normal, uniform)
        second_state, second_info = second_step(state.second, second_normal, uniform)

        return (
            CoupledHMCState(first_state, second_state),
            CoupledHMCInfo(first_info, second_info, standard_normal, unit, uniform),
        )

    return step


# --------------------------------------------------------------------
#                            PUBLIC API
# --------------------------------------------------------------------


def init(position: Position, logdensity_fn: Sequence[Callable]) -> CoupledHMCState:
    """Initialise a coupled pair.

    Parameters
    ----------
    position
        An explicit ``(first_position, second_position)`` pair.  The two
        positions must share a pytree structure, leaf shapes and one floating
        dtype.
    logdensity_fn
        An explicit ``(first_logdensity_fn, second_logdensity_fn)`` pair.  The
        two may target different distributions.

    """
    positions = _as_pair(position, "position")
    logdensity_fns = _as_pair(logdensity_fn, "logdensity_fn")
    _check_paired_positions(*positions)
    return CoupledHMCState(
        hmc.init(positions[0], logdensity_fns[0]),
        hmc.init(positions[1], logdensity_fns[1]),
    )


def build_kernel(
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
    *,
    coupling: str = "synchronous",
    direction_fn: Callable | None = None,
):
    """Build a coupled HMC kernel.

    ``coupling`` and ``direction_fn`` are fixed here, when the kernel is
    built, and not at call time.  ``direction_fn`` is applied to the pair of
    incoming states and the first metric, and is supplied no innovations.

    It is worth being exact about what that does and does not say, because an
    earlier version of this docstring overstated it.  The kernel draws ``z``
    and the uniform *before* running the transition in which ``direction_fn``
    is evaluated, so the guarantee is **not** one of ordering.  It is that the
    innovations are not among the arguments handed over.  A ``direction_fn``
    that closes over innovations, or draws its own randomness, breaks the
    reflection's marginal correctness and nothing here can detect it. Keeping
    it a pure function of the arguments it is given is the caller's part of the
    contract -- the same purity convention every BlackJAX kernel assumes.

    Parameters
    ----------
    integrator
        Symplectic integrator used by both marginals.
    divergence_threshold
        Energy difference above which a marginal transition is flagged
        divergent.  Applied to each marginal separately.
    coupling
        ``"synchronous"``, which gives both marginals the same standard
        normal, or ``"reflection"``, which gives the second marginal a
        reflected copy.
    direction_fn
        Only for ``"reflection"``.  A callable
        ``(first_state, second_state, first_metric) -> Array`` returning a
        flat direction; defaults to :func:`whitened_difference`.
        Passing one under synchronous coupling is an error, since it would
        have no effect.

    Returns
    -------
    A kernel ``(rng_key, state, logdensity_fn, step_size,
    inverse_mass_matrix, num_integration_steps) -> (CoupledHMCState,
    CoupledHMCInfo)`` in which every per-marginal parameter is an explicit
    ``(first, second)`` pair.

    """
    if coupling not in ("synchronous", "reflection"):
        raise ValueError(
            f'`coupling` must be "synchronous" or "reflection", got {coupling!r}'
        )
    if coupling == "synchronous":
        if direction_fn is not None:
            raise ValueError(
                "synchronous coupling does not use a direction; pass "
                'coupling="reflection" to reflect.'
            )
    elif direction_fn is None:
        direction_fn = whitened_difference

    def kernel(
        rng_key: PRNGKey,
        state: CoupledHMCState,
        logdensity_fn: Sequence[Callable],
        step_size: Sequence[float],
        inverse_mass_matrix: Sequence[metrics.MetricTypes],
        num_integration_steps: Sequence[int],
    ) -> tuple[CoupledHMCState, CoupledHMCInfo]:
        """Advance a coupled pair by one transition."""
        step = _build_prescribed_pair(
            logdensity_fn,
            inverse_mass_matrix,
            step_size,
            num_integration_steps,
            integrator,
            divergence_threshold,
            coupling,
            direction_fn,
        )
        flat, _ = _flat_position(state.first.position)

        key_normal, key_uniform = jax.random.split(rng_key, 2)
        standard_normal = jax.random.normal(key_normal, flat.shape, flat.dtype)
        # jax.random.uniform returns a value in [0, 1) in the requested dtype,
        # so the Metropolis precondition holds by construction and no cast --
        # hence no narrowing -- is needed.
        uniform = jax.random.uniform(key_uniform, (), flat.dtype)

        return step(state, standard_normal, uniform)

    return kernel


def as_top_level_api(
    logdensity_fn: Sequence[Callable],
    step_size: Sequence[float],
    inverse_mass_matrix: Sequence[metrics.MetricTypes],
    num_integration_steps: Sequence[int],
    *,
    coupling: str = "synchronous",
    direction_fn: Callable | None = None,
    integrator: Callable = integrators.velocity_verlet,
    divergence_threshold: float = 1000,
) -> SamplingAlgorithm:
    """Build a ``SamplingAlgorithm`` for a coupled HMC pair.

    There is deliberately no top-level ``blackjax.coupled_hmc``; reach this
    through ``blackjax.mcmc.coupled_hmc.as_top_level_api``.

    Every per-marginal argument is an explicit ``(first, second)`` pair, so a
    shared value is written twice.  The two marginals may target different
    distributions and use different fixed Gaussian Euclidean metrics, step
    sizes and integration counts, but their positions must share a pytree
    structure, leaf shapes and one floating dtype.

    Parameters
    ----------
    logdensity_fn
        Pair of log-density functions.
    step_size
        Pair of integration step sizes.
    inverse_mass_matrix
        Pair of inverse mass matrices.  Each is a diagonal array, a dense
        array, or a :class:`~blackjax.mcmc.metrics.LowRankInverseMassMatrix`;
        callable (Riemannian) metrics and pre-built
        :class:`~blackjax.mcmc.metrics.Metric` objects are not supported.
    num_integration_steps
        Pair of integration step counts.
    coupling
        ``"synchronous"`` or ``"reflection"``; see :func:`build_kernel`.
    direction_fn
        Reflection direction policy; see :func:`build_kernel`.
    integrator
        Symplectic integrator used by both marginals.
    divergence_threshold
        Per-marginal divergence threshold.

    Notes
    -----
    Each marginal's concrete metric and integration settings are checked here
    by :func:`validate_marginal_inputs`.  Those checks read concrete values on
    the host, so they cover the arguments given here and say nothing about
    traced values appearing later under ``jit`` or ``vmap``.  Callers who need
    to supply traced metrics use :func:`build_kernel` directly, which performs
    no eager numerical validation.

    Returns
    -------
    A ``SamplingAlgorithm`` whose ``init`` takes a pair of positions.

    """
    step_sizes = _as_pair(step_size, "step_size")
    inverse_mass_matrices = _as_pair(inverse_mass_matrix, "inverse_mass_matrix")
    integration_steps = _as_pair(num_integration_steps, "num_integration_steps")
    _as_pair(logdensity_fn, "logdensity_fn")

    for index in (0, 1):
        validate_marginal_inputs(
            inverse_mass_matrices[index],
            step_sizes[index],
            integration_steps[index],
        )

    kernel = build_kernel(
        integrator,
        divergence_threshold,
        coupling=coupling,
        direction_fn=direction_fn,
    )

    # `build_sampling_algorithm` is typed for a single ``logdensity_fn``; this
    # algorithm takes a pair, so the two-line wrapper is written out rather than
    # forced through a helper whose contract does not cover it.
    def init_fn(position: Position, rng_key: PRNGKey | None = None) -> State:
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state: State) -> tuple[State, Info]:
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_sizes,
            inverse_mass_matrices,
            integration_steps,
        )

    return SamplingAlgorithm(init_fn, step_fn)
