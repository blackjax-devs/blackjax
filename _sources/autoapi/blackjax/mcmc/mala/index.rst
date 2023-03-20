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



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.mala.init
   blackjax.mcmc.mala.kernel



.. py:class:: MALAState



   State of the MALA algorithm.

   The MALA algorithm takes one position of the chain and returns another
   position. In order to make computations more efficient, we also store
   the current log-probability density as well as the current gradient of the
   log-probability density.


   .. py:attribute:: position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: logdensity_grad
      :type: blackjax.types.PyTree

      


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

      


.. py:function:: init(position: blackjax.types.PyTree, logdensity_fn: Callable) -> MALAState


.. py:function:: kernel()

   Build a MALA kernel.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


