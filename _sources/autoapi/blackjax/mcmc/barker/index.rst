:py:mod:`blackjax.mcmc.barker`
==============================

.. py:module:: blackjax.mcmc.barker

.. autoapi-nested-parse::

   Public API for Barker's proposal with a Gaussian base kernel.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.barker.BarkerState
   blackjax.mcmc.barker.BarkerInfo
   blackjax.mcmc.barker.barker_proposal



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.barker.init
   blackjax.mcmc.barker.build_kernel



.. py:class:: BarkerState




   State of the Barker's proposal algorithm.

   The Barker algorithm takes one position of the chain and returns another
   position. In order to make computations more efficient, we also store
   the current log-probability density as well as the current gradient of the
   log-probability density.


   .. py:attribute:: position
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: logdensity_grad
      :type: blackjax.types.ArrayTree

      


.. py:class:: BarkerInfo




   Additional information on the Barker's proposal kernel transition.

   This additional information can be used for debugging or computing
   diagnostics.

   proposal
       The proposal that was sampled.
   acceptance_rate
       The acceptance rate of the transition.
   is_accepted
       Whether the proposed position was accepted or the original position
       was returned.


   .. py:attribute:: acceptance_rate
      :type: float

      

   .. py:attribute:: is_accepted
      :type: bool

      

   .. py:attribute:: proposal
      :type: BarkerState

      


.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable) -> BarkerState


.. py:function:: build_kernel()

   Build a Barker's proposal kernel.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: barker_proposal


   Implements the (basic) user interface for the Barker's proposal :cite:p:`Livingstone2022Barker` kernel with a
   Gaussian base kernel.

   The general Barker kernel builder (:meth:`blackjax.mcmc.barker.build_kernel`, alias `blackjax.barker.build_kernel`) can be
   cumbersome to manipulate. Since most users only need to specify the kernel
   parameters at initialization time, we provide a helper function that
   specializes the general kernel.

   We also add the general kernel and state generator as an attribute to this class so
   users only need to pass `blackjax.barker` to SMC, adaptation, etc. algorithms.

   .. rubric:: Examples

   A new Barker kernel can be initialized and used with the following code:

   .. code::

       barker = blackjax.barker(logdensity_fn, step_size)
       state = barker.init(position)
       new_state, info = barker.step(rng_key, state)

   Kernels are not jit-compiled by default so you will need to do it manually:

   .. code::

      step = jax.jit(barker.step)
      new_state, info = step(rng_key, state)

   Should you need to you can always use the base kernel directly:

   .. code::

      kernel = blackjax.barker.build_kernel(logdensity_fn)
      state = blackjax.barker.init(position, logdensity_fn)
      state, info = kernel(rng_key, state, logdensity_fn, step_size)

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


