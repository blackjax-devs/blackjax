blackjax.mcmc.laplace_marginal
==============================

.. py:module:: blackjax.mcmc.laplace_marginal

.. autoapi-nested-parse::

   Adjoint-differentiated Laplace marginal log-density.

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

   .. rubric:: References

   Margossian et al., "Hamiltonian Monte Carlo using an adjoint-differentiated
   Laplace approximation", NeurIPS 2020. arXiv:2004.12550.

   Margossian, "General adjoint-differentiated Laplace approximation", 2023.
   arXiv:2306.14976.



Classes
-------

.. autoapisummary::

   blackjax.mcmc.laplace_marginal.LaplaceHMCInfo
   blackjax.mcmc.laplace_marginal.LaplaceMarginal


Functions
---------

.. autoapisummary::

   blackjax.mcmc.laplace_marginal.laplace_marginal_factory


Module Contents
---------------

.. py:class:: LaplaceHMCInfo



   Info returned by any ``laplace_*hmc`` kernel step.

   Contains all standard :class:`~blackjax.mcmc.hmc.HMCInfo` fields for
   backward compatibility, plus L-BFGS convergence diagnostics from the
   ``theta*`` refresh that occurs after each accept/reject step.

   The L-BFGS diagnostics reflect the **post-accept-reject warm-started solve**
   — i.e., the single explicit :meth:`~LaplaceMarginal.solve_theta_with_info`
   call at the end of each kernel step, not the leapfrog-interior solves (those
   happen inside ``jax.lax.custom_root`` and are not directly accessible).

   momentum
       Momentum sampled at the start of the trajectory.
   acceptance_rate
       Metropolis acceptance probability for this transition.
   is_accepted
       Whether the proposed position was accepted.
   is_divergent
       Whether the energy difference exceeded the divergence threshold.
   energy
       Total energy (kinetic + potential) of the transition.
   proposal
       The proposed integrator state (position + momentum at trajectory end).
   num_integration_steps
       Number of leapfrog steps taken.
   lbfgs_iter_num
       Number of L-BFGS iterations at the post-accept ``theta*`` refresh.
   lbfgs_error
       Final gradient norm ``||∇f(theta*)||₂`` at the post-accept refresh.
       Large values (>> ``gtol``) indicate a non-converged inner solve.
   lbfgs_converged
       ``True`` iff ``lbfgs_error <= gtol``.  May be ``False`` for well-behaved
       warm-started solves that land near (but not below) ``gtol``; prefer
       ``lbfgs_hit_maxiter`` as the primary non-convergence alarm.
   lbfgs_hit_maxiter
       ``True`` iff the L-BFGS solver exhausted its iteration budget
       (``iter_num >= maxiter``).  **This is the direct signal for the
       silent-non-convergence bug diagnosed in blackjax issue #925.**
       When ``True``, ``theta*`` may be a poor MAP estimate and the Laplace
       log-marginal (and its gradient) is unreliable for this step.


   .. py:attribute:: momentum
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: acceptance_rate
      :type:  float


   .. py:attribute:: is_accepted
      :type:  bool


   .. py:attribute:: is_divergent
      :type:  bool


   .. py:attribute:: energy
      :type:  float


   .. py:attribute:: proposal
      :type:  Any


   .. py:attribute:: num_integration_steps
      :type:  int


   .. py:attribute:: lbfgs_iter_num
      :type:  blackjax.types.Array


   .. py:attribute:: lbfgs_error
      :type:  blackjax.types.Array


   .. py:attribute:: lbfgs_converged
      :type:  blackjax.types.Array


   .. py:attribute:: lbfgs_hit_maxiter
      :type:  blackjax.types.Array


.. py:class:: LaplaceMarginal

   Bundle of pure functions for the Laplace-approximated marginal density.

   Each attribute is a plain callable, testable and reusable independently.
   The dataclass is a named container — there is no mutable state.

   The five callables are:

   - ``solve_theta(phi, theta_prev=None) -> theta_star``: finds the mode of
     ``p(theta | phi, y)`` via L-BFGS.  No custom VJP; useful for warm-starting.
   - ``solve_theta_with_info(phi, theta_prev=None) -> (theta_star, LBFGSDiagnostics)``:
     same as ``solve_theta`` but also returns per-call L-BFGS diagnostics
     (``iter_num``, ``error``, ``converged``, ``hit_maxiter``).  Used inside
     the laplace kernel to populate :class:`LaplaceHMCInfo`.
   - ``get_theta_star(phi, theta_prev=None) -> theta_star``: same as
     ``solve_theta`` but wrapped in ``jax.lax.custom_root`` for IFT gradients.
   - ``log_marginal(phi, theta_prev=None) -> (lp, theta_star)``: evaluates the
     Laplace log-marginal and returns ``theta_star`` as auxiliary output.
     Use with ``jax.value_and_grad(..., has_aux=True)``.
   - ``sample_theta(rng_key, phi, theta_star) -> theta_sample``: draws one
     sample from ``p(theta | phi, y) ≈ N(theta_star, H(phi)^{-1})``.


   .. py:attribute:: solve_theta
      :type:  Callable


   .. py:attribute:: solve_theta_with_info
      :type:  Callable


   .. py:attribute:: get_theta_star
      :type:  Callable


   .. py:attribute:: log_marginal
      :type:  Callable


   .. py:attribute:: sample_theta
      :type:  Callable


.. py:function:: laplace_marginal_factory(log_joint_fn: Callable, theta_init: blackjax.types.ArrayLikeTree, **optimizer_kwargs) -> LaplaceMarginal

   Build a Laplace-approximated marginal log-density over hyperparameters.

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

   :param log_joint_fn: ``(theta, phi) -> float``.  Both ``theta`` and ``phi`` may be
                        arbitrary PyTrees.  Must be at least C³ smooth in ``theta``.
   :param theta_init: Initial guess for ``theta``.  Fixes the PyTree structure and shape of
                      the latent variable space for all subsequent calls.  Used as cold-start
                      fallback when ``theta_prev=None``.
   :param \*\*optimizer_kwargs: Passed through to ``blackjax.optimizers.lbfgs.minimize_lbfgs``.
                                Useful keys: ``maxiter`` (default 30), ``gtol``, ``ftol``, ``maxls``.

   :rtype: A ``LaplaceMarginal`` instance.

   .. rubric:: Examples

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

   .. rubric:: Notes

   Applicability:

   - The Laplace approximation is accurate when ``p(theta | phi, y)`` is
     approximately Gaussian (unimodal, log-concave near the mode).
   - The Hessian ``-d²/dtheta² log_joint_fn`` must be positive-definite at
     ``theta*(phi)`` for all ``phi`` encountered during sampling.
   - Memory is O(d²) and log-determinant computation is O(d³) where
     ``d = dim(theta)``.


