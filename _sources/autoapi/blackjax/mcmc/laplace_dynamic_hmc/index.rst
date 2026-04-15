blackjax.mcmc.laplace_dynamic_hmc
=================================

.. py:module:: blackjax.mcmc.laplace_dynamic_hmc

.. autoapi-nested-parse::

   Dynamic HMC on the Laplace-approximated marginal log-density.

   Combines the warm-started Laplace marginalisation of
   :mod:`~blackjax.mcmc.laplace_hmc` with the quasi-random integration-step
   schedule of :mod:`~blackjax.mcmc.dynamic_hmc`.

   The state carries both extra fields:

   - ``theta_star``: MAP of latent variables at the current ``phi``, used to
     warm-start L-BFGS at every leapfrog step.
   - ``random_generator_arg``: Halton index (or PRNG key) used by
     ``integration_steps_fn`` to draw the number of leapfrog steps each
     transition.

   Two variants are available at the top level:

   +---------------------------+------------------+------------------------------+
   | Alias                     | Proposal         | Notes                        |
   +===========================+==================+==============================+
   | ``blackjax.laplace_dhmc`` | endpoint + M-H   | default                      |
   | ``blackjax.laplace_dmhmc``| full trajectory  | multinomial, no rejection    |
   +---------------------------+------------------+------------------------------+

   Typical usage::

       sampler = blackjax.laplace_dhmc(
           log_joint, theta_init=jnp.zeros(n),
           step_size=0.1, inverse_mass_matrix=jnp.ones(d),
       )
       state = sampler.init(phi_init, rng_key)
       new_state, info = jax.jit(sampler.step)(rng_key, state)
       # new_state.theta_star  — MAP latent at accepted phi
       # new_state.random_generator_arg  — advanced Halton index



Classes
-------

.. autoapisummary::

   blackjax.mcmc.laplace_dynamic_hmc.LaplaceDynamicHMCState


Functions
---------

.. autoapisummary::

   blackjax.mcmc.laplace_dynamic_hmc.init
   blackjax.mcmc.laplace_dynamic_hmc.build_kernel
   blackjax.mcmc.laplace_dynamic_hmc.as_top_level_api


Module Contents
---------------

.. py:class:: LaplaceDynamicHMCState



   State of the Laplace dynamic HMC sampler.

   position
       Current hyperparameter position ``phi``.
   logdensity
       Current value of the Laplace log-marginal ``log p̂(phi | y)``.
   logdensity_grad
       Gradient of ``log p̂(phi | y)`` w.r.t. ``phi``.
   theta_star
       MAP of the latent variables at the current ``phi``.  Warm-starts
       L-BFGS at every leapfrog step.
   random_generator_arg
       Halton index or PRNG key consumed by ``integration_steps_fn`` to
       draw the number of leapfrog steps for the next transition.


   .. py:attribute:: position
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: logdensity
      :type:  float


   .. py:attribute:: logdensity_grad
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: theta_star
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: random_generator_arg
      :type:  blackjax.types.Array


.. py:function:: init(position: blackjax.types.ArrayLikeTree, laplace: blackjax.mcmc.laplace_marginal.LaplaceMarginal, random_generator_arg: blackjax.types.Array) -> LaplaceDynamicHMCState

   Create an initial :class:`LaplaceDynamicHMCState`.

   :param position: Initial hyperparameter value ``phi``.
   :param laplace: A :class:`~blackjax.mcmc.laplace_marginal.LaplaceMarginal` instance.
   :param random_generator_arg: Initial value for the quasi-random step-count generator (e.g. a
                                PRNG key or Halton index).  When called via the top-level API this
                                is seeded automatically from the ``rng_key`` passed to ``.init``.


.. py:function:: build_kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000, next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1], integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10), build_proposal: Callable = hmc.hmc_proposal) -> Callable

   Build the Laplace dynamic HMC kernel.

   :param integrator: Symplectic integrator for the leapfrog trajectory.
   :param divergence_threshold: Energy difference above which a transition is declared divergent.
   :param next_random_arg_fn: Advances ``random_generator_arg`` each step.
   :param integration_steps_fn: Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
                                that draws the number of leapfrog steps for a single transition.
                                Extra positional arguments are supplied at call time via
                                ``integration_steps_params`` on the inner kernel.
   :param build_proposal: Proposal builder.  Defaults to :func:`~blackjax.mcmc.hmc.hmc_proposal`
                          (endpoint + M-H).  Pass :func:`~blackjax.mcmc.hmc.multinomial_hmc_proposal`
                          for multinomial trajectory sampling (``blackjax.laplace_dmhmc``).

   :returns: * *A kernel*
             * ``(rng_key, state, laplace, step_size, inverse_mass_matrix) -> (LaplaceDynamicHMCState, HMCInfo)``.


.. py:function:: as_top_level_api(log_joint_fn: Callable, theta_init: blackjax.types.ArrayLikeTree, step_size: float, inverse_mass_matrix: blackjax.mcmc.metrics.MetricTypes, *, divergence_threshold: int = 1000, integrator: Callable = integrators.velocity_verlet, next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1], integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10), integration_steps_params: tuple = (), build_proposal: Callable = hmc.hmc_proposal, **optimizer_kwargs) -> blackjax.base.SamplingAlgorithm

   Dynamic HMC on the Laplace-approximated marginal log-density.

   Combines Laplace marginalisation over latent variables with a
   quasi-random number of leapfrog steps per transition, reducing
   periodic-orbit sensitivity while retaining the computational benefits
   of operating on the low-dimensional hyperparameter marginal.

   :param log_joint_fn: ``(theta, phi) -> float``.  Full log joint ``log p(theta, phi, y)``.
   :param theta_init: Initial guess for theta; fixes the latent PyTree structure.
   :param step_size: Leapfrog step size.
   :param inverse_mass_matrix: Inverse mass matrix (1-D array for diagonal, scalar for isotropic).
   :param divergence_threshold: Absolute energy difference above which a transition is divergent.
   :param integrator: Symplectic integrator.  Default: velocity Verlet.
   :param next_random_arg_fn: Advances ``random_generator_arg`` each step.
   :param integration_steps_fn: Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
                                that draws the number of leapfrog steps for a single transition.
   :param integration_steps_params: Extra positional arguments unpacked into ``integration_steps_fn`` after
                                    ``random_generator_arg`` on every step.  Defaults to ``()`` so that a
                                    plain 1-arg ``integration_steps_fn`` works unchanged.
   :param build_proposal: Proposal builder.  Defaults to :func:`~blackjax.mcmc.hmc.hmc_proposal`
                          (``blackjax.laplace_dhmc``).  Pass
                          :func:`~blackjax.mcmc.hmc.multinomial_hmc_proposal` for
                          ``blackjax.laplace_dmhmc``.
   :param \*\*optimizer_kwargs: Forwarded to :func:`~blackjax.optimizers.lbfgs.minimize_lbfgs`.
                                Useful keys: ``maxiter`` (default 30), ``gtol``, ``ftol``.

   :returns: * A :class:`~blackjax.base.SamplingAlgorithm` whose ``step`` returns a
             * :class:`LaplaceDynamicHMCState` and :class:`~blackjax.mcmc.hmc.HMCInfo`.

   .. rubric:: Examples

   .. code::

       sampler = blackjax.laplace_dhmc(
           log_joint, theta_init=jnp.zeros(n_latent),
           step_size=0.1, inverse_mass_matrix=jnp.ones(d_phi),
           maxiter=100,
       )
       state = sampler.init(phi_init, rng_key)
       new_state, info = jax.jit(sampler.step)(rng_key, state)
       print(new_state.theta_star)          # MAP latent at new phi
       print(new_state.random_generator_arg)  # advanced Halton index


