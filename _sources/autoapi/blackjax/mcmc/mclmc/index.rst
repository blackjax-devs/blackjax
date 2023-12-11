:py:mod:`blackjax.mcmc.mclmc`
=============================

.. py:module:: blackjax.mcmc.mclmc

.. autoapi-nested-parse::

   Public API for the MCLMC Kernel



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.mclmc.MCLMCInfo
   blackjax.mcmc.mclmc.mclmc



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.mclmc.init
   blackjax.mcmc.mclmc.build_kernel



.. py:class:: MCLMCInfo




   Additional information on the MCLMC transition.

   logdensity
       The log-density of the distribution at the current step of the MCLMC chain.
   kinetic_change
       The difference in kinetic energy between the current and previous step.
   energy_change
       The difference in energy between the current and previous step.

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: kinetic_change
      :type: float

      

   .. py:attribute:: energy_change
      :type: float

      


.. py:function:: init(position: blackjax.types.ArrayLike, logdensity_fn, rng_key)


.. py:function:: build_kernel(logdensity_fn, integrator)

   Build a HMC kernel.

   :param integrator: The symplectic integrator to use to integrate the Hamiltonian dynamics.
   :param L: the momentum decoherence rate.
   :param step_size: step size of the integrator.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: mclmc


   The general mclmc kernel builder (:meth:`blackjax.mcmc.mclmc.build_kernel`, alias `blackjax.mclmc.build_kernel`) can be
   cumbersome to manipulate. Since most users only need to specify the kernel
   parameters at initialization time, we provide a helper function that
   specializes the general kernel.

   We also add the general kernel and state generator as an attribute to this class so
   users only need to pass `blackjax.mclmc` to SMC, adaptation, etc. algorithms.

   .. rubric:: Examples

   A new mclmc kernel can be initialized and used with the following code:

   .. code::

       mclmc = blackjax.mcmc.mclmc.mclmc(
           logdensity_fn=logdensity_fn,
           L=L,
           step_size=step_size
       )
       state = mclmc.init(position)
       new_state, info = mclmc.step(rng_key, state)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

       step = jax.jit(mclmc.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param L: the momentum decoherence rate
   :param step_size: step size of the integrator
   :param integrator: an integrator. We recommend using the default here.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


