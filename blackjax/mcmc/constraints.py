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
"""Constraint types and intersection solvers for constrained sampling.

Registering custom constraint types
------------------------------------
Users can extend the dispatch system with their own constraint class by calling
:func:`register_constraint`.  The minimum requirement is ``intersect_fn`` plus
the oracle function (``oracle_fn``); the other two can be derived automatically
from the oracle via :func:`ad_barrier_hessian` and :func:`ad_scaled_gradients`
if the custom type stores the oracle as an attribute::

    from blackjax.mcmc.constraints import (
        register_constraint, ad_barrier_hessian, ad_scaled_gradients,
    )

    class MyConstraint:
        def __init__(self, ...):
            self.evaluate = ...   # callable  R^n -> R^m,  feasible iff ≤ 0

    register_constraint(
        MyConstraint,
        intersect_fn=lambda c, eps, n: lambda x, u: my_fast_intersect(c, x, u),
        oracle_fn=lambda c, pos: c.evaluate(pos),
        barrier_hessian_fn=ad_barrier_hessian(attr="evaluate"),
        scaled_gradients_fn=ad_scaled_gradients(attr="evaluate"),
    )
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree

from blackjax.types import Array

__all__ = [
    "LinearConstraint",
    "QuadraticConstraint",
    "GeneralConstraint",
    "ComposedConstraint",
    "ConstraintDict",
    "ellipsoid_constraint",
    "build_intersect_fn",
    "build_oracle_fn",
    "barrier_hessian",
    "vaidya_hessian",
    "register_constraint",
    "ad_barrier_hessian",
    "ad_scaled_gradients",
    "constrain",
]


# ---------------------------------------------------------------------------
# Constraint types
# ---------------------------------------------------------------------------


class LinearConstraint(NamedTuple):
    """Batch of linear constraints  Ax ≤ b.

    A : (m, n)
    b : (m,)
    """

    A: Array
    b: Array


class QuadraticConstraint(NamedTuple):
    """Batch of quadratic constraints  x^T A_i x + b_i^T x + c_i ≤ 0.

    A : (m, n, n)  — symmetric positive-definite matrices
    b : (m, n)     — linear coefficients
    c : (m,)       — scalar constants
    """

    A: Array
    b: Array
    c: Array


class GeneralConstraint(NamedTuple):
    """General constraint given as a callable  fn: R^n → R^m.

    The feasible region is ``{x : fn(x) ≤ 0}``.
    Intersections are found with Newton's method.
    """

    fn: Callable


class ComposedConstraint(NamedTuple):
    """Conjunction of heterogeneous constraints.

    The feasible region is the intersection of all sub-constraints.
    ``constraints`` may be any mix of
    ``LinearConstraint``, ``QuadraticConstraint``, ``GeneralConstraint``,
    or nested ``ComposedConstraint`` objects.
    """

    constraints: list


class ConstraintDict:
    """Per-leaf constraints for an ArrayTree (dict/pytree) position.

    When the sampler position is a pytree (e.g. ``{'linear': x1, 'quad': x2}``),
    this class maps each pytree key to its own constraint object.  Each
    sub-constraint applies only to the flattened slice of ``x`` that corresponds
    to the matching leaf.

    Parameters
    ----------
    constraints : dict
        A mapping ``{key: constraint}`` where each value is a
        ``LinearConstraint``, ``QuadraticConstraint``, ``GeneralConstraint``,
        ``ComposedConstraint``, or another ``ConstraintDict``.
        Keys must match the keys of the position pytree.
    example_position : pytree
        An example position with the same pytree structure as the positions
        that will be passed to the sampler.  This is used at trace time to
        compute the index slices for each leaf.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from blackjax.mcmc.constraints import LinearConstraint, ConstraintDict
    >>> A_lin = jnp.array([[1., 1.], [-1., 0.], [0., -1.]])
    >>> b_lin = jnp.array([1., 0., 0.])
    >>> example_pos = {'linear': jnp.zeros(2), 'quad': jnp.zeros(2)}
    >>> cdict = ConstraintDict(
    ...     {'linear': LinearConstraint(A_lin, b_lin)},
    ...     example_pos,
    ... )
    """

    def __init__(self, constraints: dict, example_position):
        self.constraints = constraints
        self.example_position = example_position

        # Compute, at Python-trace time, the flat-array start/end index for
        # every leaf key.  We rely on JAX's canonical leaf ordering which is
        # the same as ``ravel_pytree``.
        flat, _ = ravel_pytree(example_position)
        self._total_dim = flat.shape[0]

        # Ordered leaves with their paths
        leaves_with_paths = jax.tree_util.tree_leaves_with_path(example_position)
        offset = 0
        self._slices = {}
        for path, leaf in leaves_with_paths:
            size = jnp.size(leaf)
            key = path[0].key if len(path) == 1 else tuple(p.key for p in path)
            self._slices[key] = (offset, offset + size)
            offset += size


def ellipsoid_constraint(
    M: Array, center: Array, radius: float = 1.0
) -> QuadraticConstraint:
    """Build a ``QuadraticConstraint`` for the ellipsoid
    ``(x − center)^T M (x − center) ≤ radius²``.

    Parameters
    ----------
    M
        Shape matrix, ``(n, n)`` symmetric positive definite.
    center
        Centre of the ellipsoid, shape ``(n,)``.
    radius
        Radius (default 1).
    """
    A = M[None]  # (1, n, n)
    b = (-2.0 * M @ center)[None]  # (1, n)
    c = jnp.array([center @ M @ center - radius**2])  # (1,)
    return QuadraticConstraint(A=A, b=b, c=c)


# ---------------------------------------------------------------------------
# Internal math kernels
# ---------------------------------------------------------------------------


def _linear_intersect(
    A: Array, b: Array, x: Array, u: Array, eps: float = 1e-8
) -> Array:
    """Closed-form intersection for  Ax ≤ b  along the ray x + su."""
    Au = A @ u
    s = (b - A @ x) / Au
    mask_pos = Au > eps
    s_max = jnp.min(jnp.where(mask_pos, s, jnp.inf))
    return lax.select(s_max > 0.0, s_max, 0.0)


def _quadratic_intersect(
    constraint: QuadraticConstraint, x: Array, u: Array, eps: float = 1e-8
) -> Array:
    """Exact one-step intersection for a batch of quadratic constraints.

    For each constraint i the ray gives the quadratic
        f_i(s) = α_i s² + β_i s + f0_i = 0
    which is solved analytically with the quadratic formula.
    """
    Au = jnp.tensordot(constraint.A, u, axes=[[2], [0]])  # (m, n)
    Ax = jnp.tensordot(constraint.A, x, axes=[[2], [0]])  # (m, n)

    alpha = (Au * u).sum(-1)  # (m,)
    beta = 2.0 * (Ax * u).sum(-1) + (constraint.b * u).sum(-1)  # (m,)
    f0 = (Ax * x).sum(-1) + (constraint.b * x).sum(-1) + constraint.c  # (m,)

    disc = jnp.clip(beta**2 - 4.0 * alpha * f0, 0.0)
    s = (-beta + jnp.sqrt(disc)) / (2.0 * alpha)

    mask_pos = s > eps
    s_max = jnp.min(jnp.where(mask_pos, s, jnp.inf))
    return lax.select(s_max > 0.0, s_max, 0.0)


def _newton_intersect(
    fn: Callable, x: Array, u: Array, eps: float = 1e-8, max_iter: int = 100
) -> Array:
    """Newton's method intersection for a general constraint  fn: R^n → R^m.

    Each constraint i maintains an independent scalar iterate s_i.
    Only the JVP of fn along u is required per step (forward-mode AD).
    """
    cx, gcx_u = jax.jvp(fn, (x,), (u,))
    s_init = -cx / gcx_u

    def eval_at_s_vec(s_vec):
        def eval_one(i, s_i):
            c_val, dcu_val = jax.jvp(fn, (x + s_i * u,), (u,))
            return c_val[i], dcu_val[i]

        return jax.vmap(eval_one)(jnp.arange(s_vec.shape[0]), s_vec)

    def body(s_vec, _):
        c_diag, dcu_diag = eval_at_s_vec(s_vec)
        s_new = s_vec - c_diag / dcu_diag
        s_new = jnp.where(jnp.abs(c_diag) < eps, s_vec, s_new)
        return s_new, None

    s_final, _ = lax.scan(body, s_init, None, length=max_iter)
    mask_pos = s_final > eps
    s_max = jnp.min(jnp.where(mask_pos, s_final, jnp.inf))
    return lax.select(s_max > 0.0, s_max, 0.0)


def _linear_barrier_hessian(A: Array, b: Array, x: Array) -> Array:
    """Hessian of the log-barrier for  Ax ≤ b."""
    s = (b - A @ x).reshape(-1, 1)  # (m, 1)
    As = A / s  # (m, n)  — a_i / s_i
    return As.T @ As  # (n, n)


def _quadratic_barrier_hessian(constraint: QuadraticConstraint, x: Array) -> Array:
    """Hessian of the log-barrier for a batch of quadratic constraints."""
    Ax = jnp.tensordot(constraint.A, x, axes=[[2], [0]])  # (m, n)
    gc = 2.0 * Ax + constraint.b  # (m, n)  ∇c_i(x)
    c = (Ax * x).sum(-1) + (constraint.b * x).sum(-1) + constraint.c  # (m,)

    # outer product term: Σ_i (∇c_i)(∇c_i)^T / c_i²
    gc_over_c = gc / c[:, None]  # (m, n)
    H = gc_over_c.T @ gc_over_c  # (n, n)

    # Hessian correction term: −Σ_i 2 A_i / c_i
    H = H - 2.0 * jnp.einsum("i,ijk->jk", 1.0 / c, constraint.A)  # (n, n)

    return H


def _general_barrier_hessian(fn: Callable, x: Array) -> Array:
    """Hessian of the log-barrier for a general constraint  fn: R^n → R^m."""
    c = fn(x)  # (m,)
    J = jax.jacobian(fn)(x)  # (m, n)  — ∇c_i as rows

    # Per-output Hessians via vmap over hessian of each scalar c_i
    H_all = jax.vmap(lambda i: jax.hessian(lambda z: fn(z)[i])(x))(
        jnp.arange(c.shape[0])
    )  # (m, n, n)

    # outer product term
    J_over_c = J / c[:, None]  # (m, n)
    H = J_over_c.T @ J_over_c  # (n, n)

    # Hessian correction term
    H = H - jnp.einsum("i,ijk->jk", 1.0 / c, H_all)  # (n, n)

    return H


# ---------------------------------------------------------------------------
# AD-based fallback factories for custom constraint types
# ---------------------------------------------------------------------------


def ad_barrier_hessian(attr: str = "fn") -> Callable:
    """Return a ``barrier_hessian_fn`` handler that uses JAX autodiff.

    Reads the oracle callable from ``getattr(constraint, attr)`` and
    delegates to the general AD-based barrier Hessian computation.  Use
    this when registering a custom constraint type whose oracle is stored
    as an attribute but for which you haven't derived an analytic Hessian.

    Parameters
    ----------
    attr
        Name of the attribute on the constraint object that holds the oracle
        callable ``fn: R^n → R^m`` (feasible iff ``fn(x) ≤ 0``).
        Defaults to ``"fn"``, matching :class:`GeneralConstraint`.

    Returns
    -------
    handler : Callable
        A function with signature ``(constraint, x) -> Array`` suitable for
        the ``barrier_hessian_fn`` argument of :func:`register_constraint`.
    """
    return lambda constraint, x: _general_barrier_hessian(getattr(constraint, attr), x)


def ad_scaled_gradients(attr: str = "fn") -> Callable:
    """Return a ``scaled_gradients_fn`` handler that uses JAX autodiff.

    Reads the oracle callable from ``getattr(constraint, attr)`` and
    computes ``∇c_i(x) / c_i(x)`` rows via ``jax.jacobian``.  Use this
    when registering a custom constraint type for :func:`vaidya_hessian`
    support without a hand-written gradient formula.

    Parameters
    ----------
    attr
        Name of the attribute on the constraint object that holds the oracle
        callable ``fn: R^n → R^m`` (feasible iff ``fn(x) ≤ 0``).
        Defaults to ``"fn"``, matching :class:`GeneralConstraint`.

    Returns
    -------
    handler : Callable
        A function with signature ``(constraint, x) -> Array`` suitable for
        the ``scaled_gradients_fn`` argument of :func:`register_constraint`.
    """

    def _handler(constraint, x):
        fn = getattr(constraint, attr)
        cv = fn(x)  # (m,)
        J = jax.jacobian(fn)(x)  # (m, n)
        return J / cv[:, None]

    return _handler


# ---------------------------------------------------------------------------
# Dispatch registries
# ---------------------------------------------------------------------------
# Each registry maps  constraint_type -> handler.
#
# _INTERSECT_REGISTRY[T](constraint, eps, max_iter) -> Callable[(x, u) -> s]
# _ORACLE_REGISTRY[T](constraint, position)         -> Array (m,)
# _BARRIER_HESSIAN_REGISTRY[T](constraint, x)       -> Array (n, n)
# _SCALED_GRADIENTS_REGISTRY[T](constraint, x)      -> Array (m, n)
#
# Users add custom types via register_constraint().

_INTERSECT_REGISTRY: dict[type, Callable] = {}
_ORACLE_REGISTRY: dict[type, Callable] = {}
_BARRIER_HESSIAN_REGISTRY: dict[type, Callable] = {}
_SCALED_GRADIENTS_REGISTRY: dict[type, Callable] = {}


def register_constraint(
    constraint_type: type,
    *,
    intersect_fn: Callable,
    oracle_fn: Callable,
    barrier_hessian_fn: Callable,
    scaled_gradients_fn: Callable,
) -> None:
    """Register handlers for a custom constraint type.

    After calling this, :func:`build_intersect_fn`, :func:`build_oracle_fn`,
    :func:`barrier_hessian`, and :func:`vaidya_hessian` will all recognise
    ``constraint_type``.

    Parameters
    ----------
    constraint_type
        The Python class to register.
    intersect_fn
        Callable with signature
        ``(constraint, eps: float, max_iter: int) -> Callable[(x, u) -> s_max]``.
        ``x`` and ``u`` are **flat** 1-D arrays.  Returns a JIT-compatible
        ``(x, u) -> s_max`` closure.
    oracle_fn
        Callable with signature ``(constraint, position) -> Array`` that
        evaluates the constraint function at ``position`` (a pytree or flat
        array) and returns an ``(m,)`` array whose entries are ``≤ 0`` iff
        feasible.  Used by :func:`build_oracle_fn` for rejection-based
        sampling.
    barrier_hessian_fn
        Callable with signature ``(constraint, x: Array) -> Array`` that
        returns the ``(n, n)`` log-barrier Hessian at flat position ``x``.
        Use :func:`ad_barrier_hessian` for an autodiff fallback.
    scaled_gradients_fn
        Callable with signature ``(constraint, x: Array) -> Array`` that
        returns the ``(m, n)`` matrix of ``∇c_i(x) / c_i(x)`` rows at flat
        position ``x``, used internally by :func:`vaidya_hessian`.
        Use :func:`ad_scaled_gradients` for an autodiff fallback.

    Examples
    --------
    >>> register_constraint(
    ...     MyConstraint,
    ...     intersect_fn=lambda c, eps, n: lambda x, u: my_intersect(c, x, u),
    ...     oracle_fn=lambda c, pos: c.evaluate(pos),
    ...     barrier_hessian_fn=ad_barrier_hessian(attr="evaluate"),
    ...     scaled_gradients_fn=ad_scaled_gradients(attr="evaluate"),
    ... )
    """
    _INTERSECT_REGISTRY[constraint_type] = intersect_fn
    _ORACLE_REGISTRY[constraint_type] = oracle_fn
    _BARRIER_HESSIAN_REGISTRY[constraint_type] = barrier_hessian_fn
    _SCALED_GRADIENTS_REGISTRY[constraint_type] = scaled_gradients_fn


# ---------------------------------------------------------------------------
# Built-in type registrations  (no intermediate shim functions)
# ---------------------------------------------------------------------------

register_constraint(
    LinearConstraint,
    intersect_fn=lambda c, eps, max_iter: (
        lambda x, u: _linear_intersect(c.A, c.b, x, u, eps)
    ),
    oracle_fn=lambda c, x: c.A @ x - c.b,
    barrier_hessian_fn=lambda c, x: _linear_barrier_hessian(c.A, c.b, x),
    scaled_gradients_fn=lambda c, x: c.A / (c.b - c.A @ x).reshape(-1, 1),
)


def _quadratic_oracle(c, x):
    Ax = jnp.tensordot(c.A, x, axes=[[2], [0]])  # (m, n)
    return (Ax * x).sum(-1) + (c.b * x).sum(-1) + c.c  # (m,)


def _quadratic_scaled_gradients(c, x):
    Ax = jnp.tensordot(c.A, x, axes=[[2], [0]])  # (m, n)
    gc = 2.0 * Ax + c.b  # (m, n)
    cv = (Ax * x).sum(-1) + (c.b * x).sum(-1) + c.c  # (m,)
    return gc / cv[:, None]


register_constraint(
    QuadraticConstraint,
    intersect_fn=lambda c, eps, max_iter: (
        lambda x, u: _quadratic_intersect(c, x, u, eps)
    ),
    oracle_fn=_quadratic_oracle,
    barrier_hessian_fn=lambda c, x: _quadratic_barrier_hessian(c, x),
    scaled_gradients_fn=_quadratic_scaled_gradients,
)

register_constraint(
    GeneralConstraint,
    intersect_fn=lambda c, eps, max_iter: (
        lambda x, u: _newton_intersect(c.fn, x, u, eps, max_iter)
    ),
    oracle_fn=lambda c, x: c.fn(x),
    barrier_hessian_fn=lambda c, x: _general_barrier_hessian(c.fn, x),
    scaled_gradients_fn=ad_scaled_gradients(attr="fn"),
)

register_constraint(
    ComposedConstraint,
    intersect_fn=lambda c, eps, max_iter: (
        lambda x, u: jnp.min(
            jnp.stack(
                [build_intersect_fn(sc, eps, max_iter)(x, u) for sc in c.constraints]
            )
        )
    ),
    oracle_fn=lambda c, pos: jnp.concatenate(
        [build_oracle_fn(sc)(pos) for sc in c.constraints]
    ),
    barrier_hessian_fn=lambda c, x: sum(barrier_hessian(sc, x) for sc in c.constraints),
    scaled_gradients_fn=lambda c, x: jnp.concatenate(
        [_scaled_gradients(sc, x) for sc in c.constraints], axis=0
    ),
)


def _constraint_dict_barrier_hessian(c, x):
    H = jnp.zeros((c._total_dim, c._total_dim))
    for key, sub_c in c.constraints.items():
        start, end = c._slices[key]
        H = H.at[start:end, start:end].add(barrier_hessian(sub_c, x[start:end]))
    return H


register_constraint(
    ConstraintDict,
    intersect_fn=lambda c, eps, max_iter: (
        lambda x, u: jnp.min(
            jnp.stack(
                [
                    build_intersect_fn(sub_c, eps, max_iter)(
                        x[c._slices[key][0] : c._slices[key][1]],
                        u[c._slices[key][0] : c._slices[key][1]],
                    )
                    for key, sub_c in c.constraints.items()
                ]
            )
        )
    ),
    oracle_fn=lambda c, pos: jnp.concatenate(
        [build_oracle_fn(sub_c)(pos[key]) for key, sub_c in c.constraints.items()]
    ),
    barrier_hessian_fn=_constraint_dict_barrier_hessian,
    scaled_gradients_fn=lambda c, x: jnp.concatenate(
        [
            (
                lambda G_leaf, start, end, n_full: (
                    jnp.zeros((G_leaf.shape[0], n_full)).at[:, start:end].set(G_leaf)
                )
            )(
                _scaled_gradients(sub_c, x[c._slices[key][0] : c._slices[key][1]]),
                c._slices[key][0],
                c._slices[key][1],
                c._total_dim,
            )
            for key, sub_c in c.constraints.items()
        ],
        axis=0,
    ),
)


# ---------------------------------------------------------------------------
# Internal helper used by vaidya_hessian + ComposedConstraint oracle
# ---------------------------------------------------------------------------


def _scaled_gradients(constraint, x: Array) -> Array:
    """Return ``(m, n)`` matrix of ``∇c_i(x) / c_i(x)`` rows."""
    handler = _SCALED_GRADIENTS_REGISTRY.get(type(constraint))
    if handler is None:
        raise TypeError(
            f"Unknown constraint type: {type(constraint)}.  "
            f"Register it with register_constraint()."
        )
    return handler(constraint, x)


# ---------------------------------------------------------------------------
# Public dispatch functions
# ---------------------------------------------------------------------------


def build_intersect_fn(constraint, eps: float = 1e-8, max_iter: int = 100) -> Callable:
    """Return an ``(x, u) → s_max`` closure for the given constraint.

    ``x`` and ``u`` are **flat** 1-D arrays.  Dispatches via the
    constraint-type registry.  Built-in types:

    * ``LinearConstraint``    — closed-form, exact.
    * ``QuadraticConstraint`` — quadratic formula, exact in one step.
    * ``GeneralConstraint``   — Newton's method.
    * ``ComposedConstraint``  — recursively builds sub-solvers and takes the
                                minimum intersection distance.
    * ``ConstraintDict``      — per-leaf sub-solvers on flat-array slices.

    Custom types can be added with :func:`register_constraint`.
    """
    handler = _INTERSECT_REGISTRY.get(type(constraint))
    if handler is None:
        raise TypeError(
            f"Unknown constraint type: {type(constraint)}.  "
            f"Register it with register_constraint()."
        )
    return handler(constraint, eps, max_iter)


def build_oracle_fn(constraint) -> Callable:
    """Return a ``position → c(position)`` oracle for the given constraint.

    The oracle evaluates the constraint function at ``position`` (a pytree or
    flat array, matching whatever the sampler uses) and returns an ``(m,)``
    array.  The position is **feasible** iff all entries are ``≤ 0``.

    Useful for rejection-based sampling: draw a proposal, call the oracle,
    and reject if ``jnp.any(oracle(proposal) > 0)``.

    Built-in types:

    * ``LinearConstraint``    — ``Ax − b``.
    * ``QuadraticConstraint`` — ``[x^T A_i x + b_i^T x + c_i for each i]``.
    * ``GeneralConstraint``   — ``fn(x)`` directly.
    * ``ComposedConstraint``  — concatenation of sub-oracles.
    * ``ConstraintDict``      — concatenation of per-leaf sub-oracles; each
                                leaf oracle receives the corresponding pytree
                                sub-position (e.g. ``position['key']``).

    Custom types can be added with :func:`register_constraint`.

    Parameters
    ----------
    constraint
        Any registered constraint type.

    Returns
    -------
    oracle : Callable
        ``(position) -> Array`` of shape ``(m,)``.
    """
    handler = _ORACLE_REGISTRY.get(type(constraint))
    if handler is None:
        raise TypeError(
            f"Unknown constraint type: {type(constraint)}.  "
            f"Register it with register_constraint()."
        )
    return lambda position: handler(constraint, position)


def barrier_hessian(constraint, x: Array) -> Array:
    """Hessian of the log-barrier  ``−Σ_i log(−c_i(x))``  at ``x``.

    This is used by interior-point metrics (Dikin, Vaidya) to define the
    local geometry of the feasible region.  Each constraint type uses the
    most efficient available formula:

    * ``LinearConstraint``    — closed-form  ``A^T diag(s)^{-2} A``.
    * ``QuadraticConstraint`` — exact formula using the gradient and Hessian
                                of each quadratic.
    * ``GeneralConstraint``   — autodiff (``jax.jacobian`` + per-output
                                ``jax.hessian``).
    * ``ComposedConstraint``  — sum of sub-constraint contributions.
    * ``ConstraintDict``      — block-diagonal embedding of per-leaf Hessians.

    Custom types can be added with :func:`register_constraint`.

    Parameters
    ----------
    constraint
        Any registered constraint type.
    x
        Current position, shape ``(n,)``.

    Returns
    -------
    H : Array, shape ``(n, n)``
    """
    handler = _BARRIER_HESSIAN_REGISTRY.get(type(constraint))
    if handler is None:
        raise TypeError(
            f"Unknown constraint type: {type(constraint)}.  "
            f"Register it with register_constraint()."
        )
    return handler(constraint, x)


def vaidya_hessian(constraint, x: Array) -> Array:
    """Hessian of the Vaidya metric at ``x``.

    The Vaidya metric re-weights the Dikin (log-barrier) metric by leverage
    scores.  For each constraint i with normalised gradient
    ``g_i = ∇c_i(x) / c_i(x)``, the leverage score is
    ``σ_i = g_i^T H^{-1} g_i`` where ``H`` is the Dikin metric.  The Vaidya
    metric then adds ``σ_i × (d/n) × g_i g_i^T`` on top of the Dikin metric
    (here ``d`` = state-space dimension, ``n`` = total number of constraints).

    Dispatches via the constraint-type registry.  Custom types can be added
    with :func:`register_constraint`.

    Parameters
    ----------
    constraint
        Any registered constraint type.
    x
        Current position, shape ``(n,)``.

    Returns
    -------
    V : Array, shape ``(n, n)``
    """
    G = _scaled_gradients(constraint, x)  # (m_total, n)
    n_constraints, d = G.shape

    H = barrier_hessian(constraint, x)  # Dikin metric (n, n)

    # leverage scores: σ_i = g_i^T H^{-1} g_i
    Hinv_G = jnp.linalg.solve(H, G.T)  # (d, m_total)
    sigma = (G * Hinv_G.T).sum(-1)  # (m_total,)

    # Vaidya metric: H + Σ_i σ_i * (d / n_constraints) * g_i g_i^T
    weights = sigma * (d / n_constraints)  # (m_total,)
    V = H + (G * weights[:, None]).T @ G  # (d, d)
    return V


def constrain(logdensity_fn: Callable, constraint) -> Callable:
    """Wrap a log-density so it returns ``-inf`` outside the feasible region.

    Parameters
    ----------
    logdensity_fn
        Unconstrained log-density ``position -> scalar``.
    constraint
        Any registered constraint type.  The feasible region is
        ``{position : oracle(position) ≤ 0}`` for all constraint outputs.

    Returns
    -------
    Callable
        A new log-density with the same signature as ``logdensity_fn`` that
        returns ``logdensity_fn(position)`` when feasible and ``-inf``
        otherwise.
    """
    oracle = build_oracle_fn(constraint)

    def _constrained(position):
        return lax.cond(
            jnp.all(oracle(position) <= 0),
            lambda: logdensity_fn(position),
            lambda: -jnp.inf,
        )

    return _constrained
