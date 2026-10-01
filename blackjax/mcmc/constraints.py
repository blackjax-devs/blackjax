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
"""Constraint types and intersection solvers for constrained sampling."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from blackjax.types import Array

__all__ = [
    "LinearConstraint",
    "QuadraticConstraint",
    "GeneralConstraint",
    "ComposedConstraint",
    "ellipsoid_constraint",
    "build_intersect_fn",
    "barrier_hessian",
    "vaidya_hessian",
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
# Internal per-type intersection solvers
# ---------------------------------------------------------------------------


def _linear_intersect(
    A: Array, b: Array, x: Array, u: Array, eps: float = 1e-8
) -> float:
    """Closed-form intersection for  Ax ≤ b  along the ray x + su."""
    Au = A @ u
    s = (b - A @ x) / Au
    mask_pos = Au > eps
    s_max = jnp.min(jnp.where(mask_pos, s, jnp.inf))
    return lax.select(s_max > 0.0, s_max, 0.0)


def _quadratic_intersect(
    constraint: QuadraticConstraint, x: Array, u: Array, eps: float = 1e-8
) -> float:
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
) -> float:
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


# ---------------------------------------------------------------------------
# Python-level dispatch  (runs at trace time, not inside JAX-traced code)
# ---------------------------------------------------------------------------


def build_intersect_fn(constraint, eps: float = 1e-8, max_iter: int = 100) -> Callable:
    """Return an ``(x, u) → s_max`` closure for the given constraint.

    Dispatches to the optimal solver for each constraint type:

    * ``LinearConstraint``    — closed-form, exact.
    * ``QuadraticConstraint`` — quadratic formula, exact in one step.
    * ``GeneralConstraint``   — Newton's method.
    * ``ComposedConstraint``  — recursively builds sub-solvers and takes the
                                minimum intersection distance.
    """
    if isinstance(constraint, LinearConstraint):
        return lambda x, u: _linear_intersect(constraint.A, constraint.b, x, u, eps)

    elif isinstance(constraint, QuadraticConstraint):
        return lambda x, u: _quadratic_intersect(constraint, x, u, eps)

    elif isinstance(constraint, GeneralConstraint):
        return lambda x, u: _newton_intersect(constraint.fn, x, u, eps, max_iter)

    elif isinstance(constraint, ComposedConstraint):
        sub_fns = [build_intersect_fn(c, eps, max_iter) for c in constraint.constraints]
        return lambda x, u: jnp.min(jnp.stack([f(x, u) for f in sub_fns]))

    else:
        raise TypeError(
            f"Unknown constraint type: {type(constraint)}.  "
            f"Expected one of LinearConstraint, QuadraticConstraint, "
            f"GeneralConstraint, or ComposedConstraint."
        )


# ---------------------------------------------------------------------------
# Log-barrier Hessian  ∇²(−Σ_i log(−c_i(x)))
# ---------------------------------------------------------------------------
#
# For each constraint i the contribution is:
#   (∇c_i)(∇c_i)^T / c_i²  −  ∇²c_i / c_i
#
# Per type:
#   Linear    : ∇c_i = a_i (constant),  ∇²c_i = 0   → a_i a_i^T / s_i²
#   Quadratic : ∇c_i = 2 A_i x + b_i,  ∇²c_i = 2 A_i
#   General   : both terms via jax.jacobian / jax.hessian


def _linear_barrier_hessian(A: Array, b: Array, x: Array) -> Array:
    """Hessian of the log-barrier for  Ax ≤ b."""
    s  = (b - A @ x).reshape(-1, 1)   # (m, 1)
    As = A / s                         # (m, n)  — a_i / s_i
    return As.T @ As                   # (n, n)


def _quadratic_barrier_hessian(constraint: QuadraticConstraint, x: Array) -> Array:
    """Hessian of the log-barrier for a batch of quadratic constraints."""
    Ax  = jnp.tensordot(constraint.A, x, axes=[[2], [0]])          # (m, n)
    gc  = 2.0 * Ax + constraint.b                                   # (m, n)  ∇c_i(x)
    c   = (Ax * x).sum(-1) + (constraint.b * x).sum(-1) + constraint.c  # (m,)

    # outer product term: Σ_i (∇c_i)(∇c_i)^T / c_i²
    gc_over_c = gc / c[:, None]                                     # (m, n)
    H = gc_over_c.T @ gc_over_c                                     # (n, n)

    # Hessian correction term: −Σ_i 2 A_i / c_i
    H = H - 2.0 * jnp.einsum("i,ijk->jk", 1.0 / c, constraint.A)  # (n, n)

    return H


def _general_barrier_hessian(fn: Callable, x: Array) -> Array:
    """Hessian of the log-barrier for a general constraint  fn: R^n → R^m."""
    c  = fn(x)                         # (m,)
    J  = jax.jacobian(fn)(x)           # (m, n)  — ∇c_i as rows

    # Per-output Hessians via vmap over hessian of each scalar c_i
    H_all = jax.vmap(lambda i: jax.hessian(lambda z: fn(z)[i])(x))(
        jnp.arange(c.shape[0])
    )  # (m, n, n)

    # outer product term
    J_over_c = J / c[:, None]          # (m, n)
    H = J_over_c.T @ J_over_c         # (n, n)

    # Hessian correction term
    H = H - jnp.einsum("i,ijk->jk", 1.0 / c, H_all)  # (n, n)

    return H


def vaidya_hessian(constraint, x: Array) -> Array:
    """Hessian of the Vaidya metric at ``x``.

    The Vaidya metric re-weights the Dikin (log-barrier) metric by leverage
    scores.  For each constraint i with normalised gradient
    ``g_i = ∇c_i(x) / c_i(x)``, the leverage score is
    ``σ_i = g_i^T H^{-1} g_i`` where ``H`` is the Dikin metric.  The Vaidya
    metric then adds ``σ_i × (d/n) × g_i g_i^T`` on top of the Dikin metric
    (here ``d`` = state-space dimension, ``n`` = total number of constraints).

    Parameters
    ----------
    constraint
        Any constraint type from this module.
    x
        Current position, shape ``(n,)``.

    Returns
    -------
    V : Array, shape ``(n, n)``
    """
    # Collect (∇c_i / c_i) as rows — the "scaled gradient" for each constraint.
    # For composed constraints we concatenate rows from sub-constraints.
    def _scaled_gradients(c, z):
        """Return (m, n) matrix of ∇c_i(z) / c_i(z) for each sub-constraint."""
        if isinstance(c, LinearConstraint):
            s = (c.b - c.A @ z).reshape(-1, 1)   # (m, 1)
            return c.A / s                         # (m, n)
        elif isinstance(c, QuadraticConstraint):
            Ax  = jnp.tensordot(c.A, z, axes=[[2], [0]])       # (m, n)
            gc  = 2.0 * Ax + c.b                                # (m, n)
            cv  = (Ax * z).sum(-1) + (c.b * z).sum(-1) + c.c   # (m,)
            return gc / cv[:, None]
        elif isinstance(c, GeneralConstraint):
            cv = c.fn(z)                                        # (m,)
            J  = jax.jacobian(c.fn)(z)                         # (m, n)
            return J / cv[:, None]
        elif isinstance(c, ComposedConstraint):
            return jnp.concatenate([_scaled_gradients(sc, z) for sc in c.constraints], axis=0)
        else:
            raise TypeError(f"Unknown constraint type: {type(c)}")

    G = _scaled_gradients(constraint, x)   # (m_total, n)
    n_constraints, d = G.shape

    H = barrier_hessian(constraint, x)     # Dikin metric, (n, n)  (outer-product part only for linear)

    # leverage scores: σ_i = g_i^T H^{-1} g_i
    Hinv_G = jnp.linalg.solve(H, G.T)     # (d, m_total)
    sigma  = (G * Hinv_G.T).sum(-1)       # (m_total,)

    # Vaidya metric: H + Σ_i σ_i * (d / n_constraints) * g_i g_i^T
    weights = sigma * (d / n_constraints)  # (m_total,)
    V = H + (G * weights[:, None]).T @ G  # (d, d)
    return V


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

    Parameters
    ----------
    constraint
        Any constraint type from this module.
    x
        Current position, shape ``(n,)``.

    Returns
    -------
    H : Array, shape ``(n, n)``
    """
    if isinstance(constraint, LinearConstraint):
        return _linear_barrier_hessian(constraint.A, constraint.b, x)

    elif isinstance(constraint, QuadraticConstraint):
        return _quadratic_barrier_hessian(constraint, x)

    elif isinstance(constraint, GeneralConstraint):
        return _general_barrier_hessian(constraint.fn, x)

    elif isinstance(constraint, ComposedConstraint):
        return sum(barrier_hessian(c, x) for c in constraint.constraints)

    else:
        raise TypeError(
            f"Unknown constraint type: {type(constraint)}.  "
            f"Expected one of LinearConstraint, QuadraticConstraint, "
            f"GeneralConstraint, or ComposedConstraint."
        )
