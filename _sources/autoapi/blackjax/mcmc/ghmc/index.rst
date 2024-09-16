blackjax.mcmc.ghmc
==================

.. py:module:: blackjax.mcmc.ghmc

.. autoapi-nested-parse::

   Public API for the Generalized (Non-reversible w/ persistent momentum) HMC Kernel



Classes
-------

.. autoapisummary::

   blackjax.mcmc.ghmc.GHMCState


Functions
---------

.. autoapisummary::

   blackjax.mcmc.ghmc.init
   blackjax.mcmc.ghmc.build_kernel
   blackjax.mcmc.ghmc.as_top_level_api


Module Contents
---------------

.. py:class:: GHMCState



   State of the Generalized HMC algorithm.

   The Generalized HMC algorithm is persistent on its momentum, hence
   taking as input a position and momentum pair, updating and returning
   it for the next iteration. The algorithm also uses a persistent slice
   to perform a non-reversible Metropolis Hastings update, thus we also
   store the current slice variable and return its updated version after
   each iteration. To make computations more efficient, we also store
   the current logdensity as well as the current gradient of the
   logdensity.



   .. py:attribute:: position
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: momentum
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: logdensity
      :type:  float


   .. py:attribute:: logdensity_grad
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: slice
      :type:  float


.. py:function:: init(position: blackjax.types.ArrayLikeTree, rng_key: blackjax.types.PRNGKey, logdensity_fn: Callable) -> GHMCState

.. py:function:: build_kernel(noise_fn: Callable = lambda _: 0.0, divergence_threshold: float = 1000)

   Build a Generalized HMC kernel.

   The Generalized HMC kernel performs a similar procedure to the standard HMC
   kernel with the difference of a persistent momentum variable and a non-reversible
   Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
   step. This means that; apart from momentum and slice variables that are dependent
   on the previous momentum and slice variables, and a Metropolis-Hastings step
   performed (equivalently) as slice sampling; the standard HMC's implementation can
   be re-used to perform Generalized HMC sampling.

   :param noise_fn: A function that takes as input the slice variable and outputs a random
                    variable used as a noise correction of the persistent slice update.
                    The parameter defaults to a random variable with a single atom at 0.
   :param divergence_threshold: Value of the difference in energy above which we consider that the
                                transition is divergent.

   :returns: * *A kernel that takes a rng_key, a Pytree that contains the current state*
             * *of the chain, and free parameters of the sampling mechanism; and that*
             * *returns a new state of the chain along with information about the transition.*


.. py:function:: as_top_level_api(logdensity_fn: Callable, step_size: float, momentum_inverse_scale: blackjax.types.ArrayLikeTree, alpha: float, delta: float, *, divergence_threshold: int = 1000, noise_gn: Callable = lambda _: 0.0) -> blackjax.base.SamplingAlgorithm

   Implements the (basic) user interface for the Generalized HMC kernel.

   The Generalized HMC kernel performs a similar procedure to the standard HMC
   kernel with the difference of a persistent momentum variable and a non-reversible
   Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
   step.

   This means that the sampling of the momentum variable depends on the previous
   momentum, the rate of persistence depends on the alpha parameter, and that the
   Metropolis-Hastings accept/reject step is done through slice sampling with a
   non-reversible slice variable also dependent on the previous slice, the determinisitc
   transformation is defined by the delta parameter.

   The Generalized HMC does not have a trajectory length parameter, it always performs
   one iteration of the velocity verlet integrator with a given step size, making
   the algorithm a good candiate for running many chains in parallel.

   .. rubric:: Examples

   A new Generalized HMC kernel can be initialized and used with the following code:

   .. code::

       ghmc_kernel = blackjax.ghmc(logdensity_fn, step_size, alpha, delta)
       state = ghmc_kernel.init(rng_key, position)
       new_state, info = ghmc_kernel.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(ghmc_kernel.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: A PyTree of the same structure as the target PyTree (position) with the
                     values used for as a step size for each dimension of the target space in
                     the velocity verlet integrator.
   :param momentum_inverse_scale: Pytree with the same structure as the targeted position variable
                                  specifying the per dimension inverse scaling transformation applied
                                  to the persistent momentum variable prior to the integration step.
   :param alpha: The value defining the persistence of the momentum variable.
   :param delta: The value defining the deterministic translation of the slice variable.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param noise_gn: A function that takes as input the slice variable and outputs a random
                    variable used as a noise correction of the persistent slice update.
                    The parameter defaults to a random variable with a single atom at 0.

   :rtype: A ``SamplingAlgorithm``.


