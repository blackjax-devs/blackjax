:py:mod:`blackjax.mcmc.hmc`
===========================

.. py:module:: blackjax.mcmc.hmc

.. autoapi-nested-parse::

   Public API for the HMC Kernel



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.hmc.HMCState
   blackjax.mcmc.hmc.HMCInfo
   blackjax.mcmc.hmc.hmc



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.hmc.init
   blackjax.mcmc.hmc.build_kernel



.. py:class:: HMCState




   State of the HMC algorithm.

   The HMC algorithm takes one position of the chain and returns another
   position. In order to make computations more efficient, we also store
   the current logdensity as well as the current gradient of the logdensity.


   .. py:attribute:: position
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: logdensity_grad
      :type: blackjax.types.ArrayTree

      


.. py:class:: HMCInfo




   Additional information on the HMC transition.

   This additional information can be used for debugging or computing
   diagnostics.

   momentum:
       The momentum that was sampled and used to integrate the trajectory.
   acceptance_rate
       The acceptance probability of the transition, linked to the energy
       difference between the original and the proposed states.
   is_accepted
       Whether the proposed position was accepted or the original position
       was returned.
   is_divergent
       Whether the difference in energy between the original and the new state
       exceeded the divergence threshold.
   energy:
       Total energy of the transition.
   proposal
       The state proposed by the proposal. Typically includes the position and
       momentum.
   step_size
       Size of the integration step.
   num_integration_steps
       Number of times we run the symplectic integrator to build the trajectory


   .. py:attribute:: momentum
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: acceptance_rate
      :type: float

      

   .. py:attribute:: is_accepted
      :type: bool

      

   .. py:attribute:: is_divergent
      :type: bool

      

   .. py:attribute:: energy
      :type: float

      

   .. py:attribute:: proposal
      :type: blackjax.mcmc.integrators.IntegratorState

      

   .. py:attribute:: num_integration_steps
      :type: int

      


.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable)


.. py:function:: build_kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000)

   Build a HMC kernel.

   :param integrator: The symplectic integrator to use to integrate the Hamiltonian dynamics.
   :param divergence_threshold: Value of the difference in energy above which we consider that the transition is
                                divergent.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: hmc


   Implements the (basic) user interface for the HMC kernel.

   The general hmc kernel builder (:meth:`blackjax.mcmc.hmc.build_kernel`, alias
   `blackjax.hmc.build_kernel`) can be cumbersome to manipulate. Since most users only
   need to specify the kernel parameters at initialization time, we provide a helper
   function that specializes the general kernel.

   We also add the general kernel and state generator as an attribute to this class so
   users only need to pass `blackjax.hmc` to SMC, adaptation, etc. algorithms.

   .. rubric:: Examples

   A new HMC kernel can be initialized and used with the following code:

   .. code::

       hmc = blackjax.hmc(
           logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps
       )
       state = hmc.init(position)
       new_state, info = hmc.step(rng_key, state)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(hmc.step)
      new_state, info = step(rng_key, state)

   Should you need to you can always use the base kernel directly:

   .. code::

      import blackjax.mcmc.integrators as integrators

      kernel = blackjax.hmc.build_kernel(integrators.mclachlan)
      state = blackjax.hmc.init(position, logdensity_fn)
      state, info = kernel(
          rng_key,
          state,
          logdensity_fn,
          step_size,
          inverse_mass_matrix,
          num_integration_steps,
      )

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value for
                               the momentum and computing the kinetic energy. This argument will be
                               passed to the ``metrics.default_metric`` function so it supports the
                               full interface presented there.
   :param num_integration_steps: The number of steps we take with the symplectic integrator at each
                                 sample step before returning a sample.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate the
                      trajectory.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


