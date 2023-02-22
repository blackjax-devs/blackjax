:py:mod:`blackjax.mcmc.rmh`
===========================

.. py:module:: blackjax.mcmc.rmh

.. autoapi-nested-parse::

   Public API for Rosenbluth-Metropolis-Hastings kernels.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.rmh.RMHState
   blackjax.mcmc.rmh.RMHInfo



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.rmh.init
   blackjax.mcmc.rmh.kernel



.. py:class:: RMHState



   State of the RMH chain.

   position
       Current position of the chain.
   log_density
       Current value of the log-density


   .. py:attribute:: position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: log_density
      :type: float

      


.. py:class:: RMHInfo



   Additional information on the RMH chain.

   This additional information can be used for debugging or computing
   diagnostics.

   acceptance_rate
       The acceptance probability of the transition, linked to the energy
       difference between the original and the proposed states.
   is_accepted
       Whether the proposed position was accepted or the original position
       was returned.
   proposal
       The state proposed by the proposal.


   .. py:attribute:: acceptance_rate
      :type: float

      

   .. py:attribute:: is_accepted
      :type: bool

      

   .. py:attribute:: proposal
      :type: RMHState

      


.. py:function:: init(position: blackjax.types.PyTree, logdensity_fn: Callable) -> RMHState

   Create a chain state from a position.

   :param position: The initial position of the chain
   :type position: PyTree
   :param logdensity_fn: Log-probability density function of the distribution we wish to sample
                         from.
   :type logdensity_fn: Callable


.. py:function:: kernel()

   Build a Random Walk Rosenbluth-Metropolis-Hastings kernel with a gaussian
   proposal distribution.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


