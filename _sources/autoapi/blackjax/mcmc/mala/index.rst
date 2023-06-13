:py:mod:`blackjax.mcmc.mala`
============================

.. py:module:: blackjax.mcmc.mala

.. autoapi-nested-parse::

   Public API for Metropolis Adjusted Langevin kernels.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.mala.MALAState
   blackjax.mcmc.mala.MALAInfo
   blackjax.mcmc.mala.mala



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.mala.init
   blackjax.mcmc.mala.build_kernel



.. py:class:: MALAState




   State of the MALA algorithm.

   The MALA algorithm takes one position of the chain and returns another
   position. In order to make computations more efficient, we also store
   the current log-probability density as well as the current gradient of the
   log-probability density.


   .. py:attribute:: position
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: logdensity_grad
      :type: blackjax.types.ArrayTree

      


.. py:class:: MALAInfo




   Additional information on the MALA transition.

   This additional information can be used for debugging or computing
   diagnostics.

   acceptance_rate
       The acceptance rate of the transition.
   is_accepted
       Whether the proposed position was accepted or the original position
       was returned.


   .. py:attribute:: acceptance_rate
      :type: float

      

   .. py:attribute:: is_accepted
      :type: bool

      


.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable) -> MALAState


.. py:function:: build_kernel()

   Build a MALA kernel.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: mala


   Implements the (basic) user interface for the MALA kernel.

   The general mala kernel builder (:meth:`blackjax.mcmc.mala.build_kernel`, alias `blackjax.mala.build_kernel`) can be
   cumbersome to manipulate. Since most users only need to specify the kernel
   parameters at initialization time, we provide a helper function that
   specializes the general kernel.

   We also add the general kernel and state generator as an attribute to this class so
   users only need to pass `blackjax.mala` to SMC, adaptation, etc. algorithms.

   .. rubric:: Examples

   A new MALA kernel can be initialized and used with the following code:

   .. code::

       mala = blackjax.mala(logdensity_fn, step_size)
       state = mala.init(position)
       new_state, info = mala.step(rng_key, state)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(mala.step)
      new_state, info = step(rng_key, state)

   Should you need to you can always use the base kernel directly:

   .. code::

      kernel = blackjax.mala.build_kernel(logdensity_fn)
      state = blackjax.mala.init(position, logdensity_fn)
      state, info = kernel(rng_key, state, logdensity_fn, step_size)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


