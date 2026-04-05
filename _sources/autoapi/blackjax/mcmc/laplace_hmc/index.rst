blackjax.mcmc.laplace_hmc
=========================

.. py:module:: blackjax.mcmc.laplace_hmc

.. autoapi-nested-parse::

   HMC on the Laplace-approximated marginal log-density with warm-starting.

   Wraps the composable :func:`~blackjax.mcmc.laplace_marginal.laplace_marginal_factory`
   in a standard BlackJAX three-layer sampler that carries the MAP latent variables
   ``theta_star`` through the chain.  At each step ``theta_star`` is used as the
   warm-start hint for the L-BFGS solver at every leapfrog evaluation, so the
   optimizer needs only a handful of iterations when ``phi`` moves by a small amount.

   Typical usage::

       sampler = blackjax.laplace_hmc(
           log_joint, theta_init=jnp.zeros(n),
           step_size=0.1, inverse_mass_matrix=jnp.ones(d),
           num_integration_steps=10,
       )
       state = sampler.init(phi_init)
       new_state, info = jax.jit(sampler.step)(rng_key, state)
       # new_state.theta_star: MAP of theta at the accepted phi



Classes
-------

.. autoapisummary::

   blackjax.mcmc.laplace_hmc.LaplaceHMCState


Functions
---------

.. autoapisummary::

   blackjax.mcmc.laplace_hmc.init
   blackjax.mcmc.laplace_hmc.build_kernel
   blackjax.mcmc.laplace_hmc.as_top_level_api


Module Contents
---------------

.. py:class:: LaplaceHMCState



   State of the Laplace-HMC sampler.

   position
       Current hyperparameter position ``phi``.  Can be any PyTree.
   logdensity
       Current value of the Laplace log-marginal ``log p̂(phi | y)``.
   logdensity_grad
       Gradient of ``log p̂(phi | y)`` w.r.t. ``phi``.  Same PyTree structure
       as ``position``.
   theta_star
       MAP of the latent variables at the current ``phi``, i.e.
       ``theta*(phi) = argmax_theta log_joint(theta, phi)``.  Carried through
       the chain so the next L-BFGS solve can warm-start from here.


   .. py:attribute:: position
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: logdensity
      :type:  float


   .. py:attribute:: logdensity_grad
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: theta_star
      :type:  blackjax.types.ArrayTree


.. py:function:: init(position: blackjax.types.ArrayLikeTree, laplace: blackjax.mcmc.laplace_marginal.LaplaceMarginal) -> LaplaceHMCState

   Create an initial :class:`LaplaceHMCState`.

   Runs L-BFGS from cold start to find ``theta*(position)``, then evaluates
   the Laplace log-marginal and its gradient.

   :param position: Initial hyperparameter value ``phi``.
   :param laplace: A :class:`~blackjax.mcmc.laplace_marginal.LaplaceMarginal` instance
                   returned by :func:`~blackjax.mcmc.laplace_marginal.laplace_marginal_factory`.


.. py:function:: build_kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000) -> Callable

   Build the Laplace-HMC kernel.

   :param integrator: Symplectic integrator used for the HMC trajectory.
   :param divergence_threshold: Energy difference above which a transition is declared divergent.

   :returns: * A kernel ``(rng_key, state, laplace, step_size, inverse_mass_matrix,
             * num_integration_steps) -> (LaplaceHMCState, HMCInfo)``.


.. py:function:: as_top_level_api(log_joint_fn: Callable, theta_init: blackjax.types.ArrayLikeTree, step_size: float, inverse_mass_matrix: blackjax.mcmc.metrics.MetricTypes, num_integration_steps: int, *, divergence_threshold: int = 1000, integrator: Callable = integrators.velocity_verlet, **optimizer_kwargs) -> blackjax.base.SamplingAlgorithm

   HMC on the Laplace-approximated marginal log-density.

   For a hierarchical model ``log p(theta, phi, y)``, integrates out the
   latent variables ``theta`` via the Laplace approximation and runs HMC on
   the resulting marginal over the hyperparameters ``phi``.

   Gradients w.r.t. ``phi`` are computed via the implicit function theorem
   (:func:`jax.lax.custom_root`) — the L-BFGS iterations are *not* unrolled.
   ``theta*(phi)`` is warm-started from the previous MCMC state, reducing the
   number of L-BFGS iterations needed at each leapfrog step.

   :param log_joint_fn: ``(theta, phi) -> float``.  The full log joint ``log p(theta, phi, y)``.
                        Both arguments may be arbitrary PyTrees.  Must be at least C³ in theta.
   :param theta_init: Initial guess for theta.  Fixes the PyTree structure for all calls.
   :param step_size: HMC leapfrog step size.
   :param inverse_mass_matrix: Inverse mass matrix for HMC (1-D array for diagonal, scalar for isotropic).
   :param num_integration_steps: Number of leapfrog steps per HMC transition.
   :param divergence_threshold: Absolute energy difference above which a transition is declared divergent.
                                Default 1000.
   :param integrator: Symplectic integrator.  Default: velocity Verlet.
   :param \*\*optimizer_kwargs: Forwarded to :func:`~blackjax.optimizers.lbfgs.minimize_lbfgs`.
                                Useful keys: ``maxiter`` (default 30), ``gtol``, ``ftol``.

   :returns: * A :class:`~blackjax.base.SamplingAlgorithm` whose ``step`` returns a
             * :class:`LaplaceHMCState` (with ``theta_star`` field) and
             * :class:`~blackjax.mcmc.hmc.HMCInfo`.

   .. rubric:: Examples

   .. code::

       sampler = blackjax.laplace_hmc(
           log_joint, theta_init=jnp.zeros(n_latent),
           step_size=0.1, inverse_mass_matrix=jnp.ones(d_phi),
           num_integration_steps=10, maxiter=100,
       )
       state = sampler.init(phi_init)
       new_state, info = jax.jit(sampler.step)(rng_key, state)
       print(new_state.theta_star)   # MAP latent at the new phi


