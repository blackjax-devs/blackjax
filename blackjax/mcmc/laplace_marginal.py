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
"""Adjoint-differentiated Laplace marginal log-density.

Provides a differentiable approximation to the marginal log-density obtained
by integrating out latent Gaussian variables via the Laplace approximation.
Intended for use in hierarchical models where sampling the joint posterior
over latent variables and hyperparameters is geometrically difficult.

Typical model structure::

    phi   ~ p(phi)                  # hyperparameters (small dimension)
    theta ~ N(0, K(phi))            # latent Gaussian variables (large dimension)
    y     ~ p(y | theta, phi)       # observations (any C³ likelihood)

``laplace_marginal_factory`` returns a ``LaplaceMarginal`` object whose
``__call__`` method evaluates the Laplace-approximated marginal log-density
``log p̂(phi | y)`` with correct gradients via the implicit function theorem.

References
----------
Margossian et al., "Hamiltonian Monte Carlo using an adjoint-differentiated
Laplace approximation", NeurIPS 2020. arXiv:2004.12550.

Margossian, "General adjoint-differentiated Laplace approximation", 2023.
arXiv:2306.14976.
"""
import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.optimizers.lbfgs import minimize_lbfgs
from blackjax.types import ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["LaplaceMarginal", "laplace_marginal_factory"]


@dataclasses.dataclass(frozen=True)
class LaplaceMarginal:
    """Bundle of pure functions for the Laplace-approximated marginal density.

    Each attribute is a plain callable, testable and reusable independently.
    The dataclass is a named container — there is no mutable state.

    The four callables are:

    - ``solve_theta(phi, theta_prev=None) -> theta_star``: finds the mode of
      ``p(theta | phi, y)`` via L-BFGS.  No custom VJP; useful for warm-starting.
    - ``get_theta_star(phi, theta_prev=None) -> theta_star``: same as
      ``solve_theta`` but wrapped in ``jax.lax.custom_root`` for IFT gradients.
    - ``log_marginal(phi, theta_prev=None) -> (lp, theta_star)``: evaluates the
      Laplace log-marginal and returns ``theta_star`` as auxiliary output.
      Use with ``jax.value_and_grad(..., has_aux=True)``.
    - ``sample_theta(rng_key, phi, theta_star) -> theta_sample``: draws one
      sample from ``p(theta | phi, y) ≈ N(theta_star, H(phi)^{-1})``.
    """

    solve_theta: Callable
    get_theta_star: Callable
    log_marginal: Callable
    sample_theta: Callable

    def __call__(
        self, phi: ArrayLikeTree, theta_prev: ArrayTree | None = None
    ) -> tuple[float, ArrayTree]:
        """Evaluate log p̂(phi | y).  Alias for ``log_marginal``.

        Compatible with ``jax.value_and_grad(..., has_aux=True)``.
        """
        return self.log_marginal(phi, theta_prev)


def laplace_marginal_factory(
    log_joint_fn: Callable,
    theta_init: ArrayLikeTree,
    **optimizer_kwargs,
) -> LaplaceMarginal:
    """Build a Laplace-approximated marginal log-density over hyperparameters.

    For a model ``log_joint_fn(theta, phi) = log p(theta, phi, y)``, returns a
    ``LaplaceMarginal`` whose ``__call__`` evaluates the Laplace approximation::

        log p̂(phi | y) ≈ log p(theta*(phi), phi, y)
                        - 1/2 log|det(-H(theta*(phi), phi))|
                        + d/2 log(2π)

    where ``theta*(phi) = argmax_theta log_joint_fn(theta, phi)`` is found via
    L-BFGS and ``H = d²/dtheta² log_joint_fn`` is the Hessian at the mode.

    Gradients w.r.t. ``phi`` are computed via the implicit function theorem
    (``jax.lax.custom_root``): the L-BFGS iterations are *not* unrolled.
    The log-determinant gradient uses JAX's built-in VJP for
    ``jnp.linalg.slogdet``.

    Parameters
    ----------
    log_joint_fn
        ``(theta, phi) -> float``.  Both ``theta`` and ``phi`` may be
        arbitrary PyTrees.  Must be at least C³ smooth in ``theta``.
    theta_init
        Initial guess for ``theta``.  Fixes the PyTree structure and shape of
        the latent variable space for all subsequent calls.  Used as cold-start
        fallback when ``theta_prev=None``.
    **optimizer_kwargs
        Passed through to ``blackjax.optimizers.lbfgs.minimize_lbfgs``.
        Useful keys: ``maxiter`` (default 30), ``gtol``, ``ftol``, ``maxls``.

    Returns
    -------
    A ``LaplaceMarginal`` instance.

    Examples
    --------
    .. code::

        def log_joint(theta, phi):
            log_p_phi   = jax.scipy.stats.halfnorm.logpdf(phi, 0, 1)
            log_p_theta = jax.scipy.stats.norm.logpdf(theta, 0, phi).sum()
            log_lik     = jax.scipy.stats.norm.logpdf(y_obs, theta, 1).sum()
            return log_p_phi + log_p_theta + log_lik

        laplace = laplace_marginal_factory(log_joint, jnp.zeros(n))

        # Evaluate with gradient (for use in any sampler):
        (lp, theta_star), grad = jax.value_and_grad(
            laplace, has_aux=True
        )(phi)

        # Individual components are testable in isolation:
        theta_star = laplace.solve_theta(phi, theta_prev=prev_theta_star)

    Notes
    -----
    Applicability:

    - The Laplace approximation is accurate when ``p(theta | phi, y)`` is
      approximately Gaussian (unimodal, log-concave near the mode).
    - The Hessian ``-d²/dtheta² log_joint_fn`` must be positive-definite at
      ``theta*(phi)`` for all ``phi`` encountered during sampling.
    - Memory is O(d²) and log-determinant computation is O(d³) where
      ``d = dim(theta)``.
    """
    theta_flat_init, unravel_theta = ravel_pytree(theta_init)
    d = theta_flat_init.shape[0]

    # ------------------------------------------------------------------
    # solve_theta: pure L-BFGS mode-finding, no custom VJP
    # ------------------------------------------------------------------
    def solve_theta(
        phi: ArrayLikeTree, theta_prev: ArrayTree | None = None
    ) -> ArrayTree:
        """Find theta*(phi) via L-BFGS.  Warm-starts from theta_prev if given."""
        initial = theta_prev if theta_prev is not None else theta_init

        def objective(theta):
            return -log_joint_fn(theta, phi)

        result, _ = minimize_lbfgs(objective, initial, **optimizer_kwargs)
        return result.params

    # ------------------------------------------------------------------
    # get_theta_star: same solve, wrapped in custom_root for IFT gradient
    # ------------------------------------------------------------------
    def get_theta_star(
        phi: ArrayLikeTree, theta_prev: ArrayTree | None = None
    ) -> ArrayTree:
        """Find theta*(phi) with correct implicit-function-theorem gradient."""

        def f_residual(theta_flat):
            # Gradient of log_joint w.r.t. theta (= 0 at the mode).
            theta = unravel_theta(theta_flat)
            grad_theta = jax.grad(log_joint_fn, argnums=0)(theta, phi)
            grad_flat, _ = ravel_pytree(grad_theta)
            return grad_flat

        def solve_root(f, x0):
            del f  # custom_root requires this signature; we use solve_theta
            theta_star = solve_theta(phi, theta_prev)
            theta_star_flat, _ = ravel_pytree(theta_star)
            return theta_star_flat

        def tangent_solve(g, y):
            # g is the linearised residual at theta_flat_star (a linear map).
            # jax.jacobian(g)(zeros) extracts the Jacobian matrix (= Hessian
            # of log_joint at theta*) without needing to know theta_flat_star.
            J = jax.jacobian(g)(jnp.zeros_like(theta_flat_init))
            return jnp.linalg.solve(J, y)

        theta_flat_star = jax.lax.custom_root(
            f_residual, theta_flat_init, solve_root, tangent_solve
        )
        return unravel_theta(theta_flat_star)

    # ------------------------------------------------------------------
    # log_marginal: assemble the Laplace log-marginal value
    # ------------------------------------------------------------------
    def log_marginal(
        phi: ArrayLikeTree, theta_prev: ArrayTree | None = None
    ) -> tuple[float, ArrayTree]:
        """Evaluate log p̂(phi | y).  Returns (lp, theta_star) for has_aux=True."""
        theta_star = get_theta_star(phi, theta_prev)
        theta_flat_star, _ = ravel_pytree(theta_star)

        def log_joint_flat(t_flat):
            return log_joint_fn(unravel_theta(t_flat), phi)

        log_p_star = log_joint_flat(theta_flat_star)

        # H = -Hessian_theta log_joint (positive-definite at the mode)
        neg_hess = jax.hessian(lambda t: -log_joint_flat(t))(theta_flat_star)
        _, log_abs_det = jnp.linalg.slogdet(neg_hess)

        lp = log_p_star - 0.5 * log_abs_det + 0.5 * d * jnp.log(2.0 * jnp.pi)
        return lp, theta_star

    # ------------------------------------------------------------------
    # sample_theta: draw from the Laplace-approximate conditional posterior
    # ------------------------------------------------------------------
    def sample_theta(
        rng_key: PRNGKey,
        phi: ArrayLikeTree,
        theta_star: ArrayTree,
    ) -> ArrayTree:
        """Sample theta ~ N(theta_star, H(phi)^{-1}).

        Parameters
        ----------
        rng_key
            JAX PRNG key.
        phi
            Hyperparameter value (a single MCMC sample).
        theta_star
            MAP of theta at ``phi``, e.g. ``state.theta_star`` from a
            :class:`~blackjax.mcmc.laplace_hmc.LaplaceHMCState`.

        Returns
        -------
        A sample from the Laplace-approximate posterior
        ``p(theta | phi, y) ≈ N(theta_star, H^{-1})``.
        """
        theta_flat_star, _ = ravel_pytree(theta_star)

        def log_joint_flat(t_flat):
            return log_joint_fn(unravel_theta(t_flat), phi)

        # H = -Hessian_theta log_joint at theta_star (positive-definite)
        neg_hess = jax.hessian(lambda t: -log_joint_flat(t))(theta_flat_star)

        # Cholesky: H = L L^T.  Sample x = theta_star + L^{-T} z, z ~ N(0, I).
        # This is equivalent to x ~ N(theta_star, H^{-1}).
        L = jnp.linalg.cholesky(neg_hess)
        z = jax.random.normal(rng_key, (d,))
        # Solve L^T x_flat = z  =>  x_flat = (L^T)^{-1} z
        x_flat = jax.lax.linalg.triangular_solve(
            L, z, left_side=True, lower=True, transpose_a=True
        )
        return unravel_theta(theta_flat_star + x_flat)

    return LaplaceMarginal(
        solve_theta=solve_theta,
        get_theta_star=get_theta_star,
        log_marginal=log_marginal,
        sample_theta=sample_theta,
    )
